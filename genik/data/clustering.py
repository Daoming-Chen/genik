"""Pose Space Clustering for GeNIK.

This module provides functionality for clustering joint configurations
by their end-effector poses to identify multi-solution regions in the workspace.
"""

from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import warnings

import numpy as np
from scipy.spatial import KDTree
import torch
from tqdm import tqdm


def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compute the geodesic distance between quaternions.
    
    Parameters
    ----------
    q1 : np.ndarray
        First quaternion(s) of shape (4,) or (N, 4) in (w, x, y, z) format.
    q2 : np.ndarray
        Second quaternion(s) of shape (4,) or (N, 4) in (w, x, y, z) format.
        
    Returns
    -------
    np.ndarray
        Angular distance(s) in radians.
    """
    # Ensure 2D
    if q1.ndim == 1:
        q1 = q1[np.newaxis, :]
    if q2.ndim == 1:
        q2 = q2[np.newaxis, :]
    
    # Handle antipodal ambiguity: q and -q represent same rotation
    dot = np.sum(q1 * q2, axis=1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    
    # Angular distance: 2 * arccos(|q1 · q2|)
    angle = 2.0 * np.arccos(dot)
    
    return angle


def pose_distance(
    pose1: np.ndarray,
    pose2: np.ndarray,
    w_pos: float = 1.0,
    w_ori: float = 1.0,
) -> np.ndarray:
    """Compute weighted distance between poses.
    
    Parameters
    ----------
    pose1 : np.ndarray
        First pose(s) of shape (7,) or (N, 7) containing [x, y, z, qw, qx, qy, qz].
    pose2 : np.ndarray
        Second pose(s) of shape (7,) or (N, 7).
    w_pos : float
        Weight for position distance (in meters).
    w_ori : float
        Weight for orientation distance (in radians).
        
    Returns
    -------
    np.ndarray
        Weighted distance(s).
    """
    # Ensure 2D
    if pose1.ndim == 1:
        pose1 = pose1[np.newaxis, :]
    if pose2.ndim == 1:
        pose2 = pose2[np.newaxis, :]
    
    # Position distance (Euclidean)
    pos_dist = np.linalg.norm(pose1[:, :3] - pose2[:, :3], axis=1)
    
    # Orientation distance (geodesic)
    ori_dist = quaternion_distance(pose1[:, 3:], pose2[:, 3:])
    
    # Weighted combination
    return np.sqrt(w_pos * pos_dist**2 + w_ori * ori_dist**2)


class PoseClusterer:
    """Pose Space Clusterer for identifying multi-solution regions.
    
    This class groups joint configurations by their end-effector poses
    to identify regions where multiple IK solutions exist.
    
    Parameters
    ----------
    epsilon_pos : float
        Position epsilon for clustering in meters (default: 0.001 = 1mm).
    epsilon_ori : float
        Orientation epsilon for clustering in radians (default: 0.0175 ≈ 1°).
        
    Attributes
    ----------
    kdtree : KDTree
        KD-tree for efficient nearest neighbor search.
    poses : np.ndarray
        Stored poses for lookup.
    joint_configs : np.ndarray
        Stored joint configurations.
    cluster_map : dict
        Mapping from pose index to list of joint config indices.
        
    Examples
    --------
    >>> clusterer = PoseClusterer(epsilon_pos=0.001, epsilon_ori=0.0175)
    >>> clusterer.fit(poses, joint_configs)
    >>> candidates = clusterer.query(target_pose)
    """
    
    def __init__(
        self,
        epsilon_pos: float = 0.001,  # 1mm
        epsilon_ori: float = 0.0175,  # ~1 degree
    ):
        self.epsilon_pos = epsilon_pos
        self.epsilon_ori = epsilon_ori
        
        # Data storage
        self.kdtree: Optional[KDTree] = None
        self.poses: Optional[np.ndarray] = None
        self.joint_configs: Optional[np.ndarray] = None
        
        # Cluster mapping
        self.cluster_map: Dict[int, List[int]] = {}
        self._cluster_representatives: Optional[np.ndarray] = None
        self._cluster_indices: Optional[List[List[int]]] = None
    
    def fit(
        self,
        poses: np.ndarray,
        joint_configs: np.ndarray,
        show_progress: bool = True,
    ) -> 'PoseClusterer':
        """Fit the clusterer to the data.
        
        Parameters
        ----------
        poses : np.ndarray
            End-effector poses of shape (N, 7).
        joint_configs : np.ndarray
            Joint configurations of shape (N, n_joints).
        show_progress : bool
            Whether to show progress bar.
            
        Returns
        -------
        PoseClusterer
            Self for method chaining.
        """
        self.poses = poses.astype(np.float32)
        self.joint_configs = joint_configs.astype(np.float32)
        
        # Build KD-tree using only position for fast queries
        # We'll filter by orientation separately
        self.kdtree = KDTree(self.poses[:, :3])
        
        # Build cluster map
        self._build_cluster_map(show_progress)
        
        return self
    
    def _build_cluster_map(self, show_progress: bool = True):
        """Build mapping from poses to candidate joint configs."""
        n_samples = len(self.poses)
        
        # For each pose, find all poses within epsilon
        self.cluster_map = {}
        
        # Use a union-find approach for efficiency
        visited = np.zeros(n_samples, dtype=bool)
        cluster_representatives = []
        cluster_indices = []
        
        iterator = range(n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Building clusters")
        
        for i in iterator:
            if visited[i]:
                continue
            
            # Query nearby poses
            pos_indices = self.kdtree.query_ball_point(
                self.poses[i, :3], self.epsilon_pos
            )
            
            # Filter by orientation
            ori_distances = quaternion_distance(
                self.poses[i, 3:], self.poses[pos_indices, 3:]
            )
            valid_indices = np.array(pos_indices)[ori_distances < self.epsilon_ori]
            
            # Mark all as visited
            visited[valid_indices] = True
            
            # Create cluster
            cluster_idx = len(cluster_representatives)
            cluster_representatives.append(i)
            cluster_indices.append(valid_indices.tolist())
            
            # Map each sample to its cluster
            for idx in valid_indices:
                self.cluster_map[idx] = cluster_idx
        
        self._cluster_representatives = np.array(cluster_representatives)
        self._cluster_indices = cluster_indices
    
    def query(
        self,
        target_pose: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query candidate joint configurations for a target pose.
        
        Parameters
        ----------
        target_pose : np.ndarray
            Target pose of shape (7,) or (N, 7).
        k : int
            Number of nearest neighbors to query initially.
            
        Returns
        -------
        tuple
            (joint_configs, indices) where:
            - joint_configs: np.ndarray of candidate configurations
            - indices: np.ndarray of indices into original data
        """
        if self.kdtree is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")
        
        if target_pose.ndim == 1:
            target_pose = target_pose[np.newaxis, :]
        
        all_candidates = []
        all_indices = []
        
        for pose in target_pose:
            # Query nearby poses by position
            distances, indices = self.kdtree.query(pose[:3], k=k)
            
            # Handle case where k=1
            if np.isscalar(indices):
                indices = np.array([indices])
                distances = np.array([distances])
            
            # Filter by position epsilon
            pos_valid = distances < self.epsilon_pos
            indices = indices[pos_valid]
            
            if len(indices) == 0:
                all_candidates.append(np.array([]))
                all_indices.append(np.array([]))
                continue
            
            # Filter by orientation epsilon
            ori_distances = quaternion_distance(pose[3:], self.poses[indices, 3:])
            ori_valid = ori_distances < self.epsilon_ori
            valid_indices = indices[ori_valid]
            
            if len(valid_indices) == 0:
                all_candidates.append(np.array([]))
                all_indices.append(np.array([]))
                continue
            
            all_candidates.append(self.joint_configs[valid_indices])
            all_indices.append(valid_indices)
        
        if len(target_pose) == 1:
            return all_candidates[0], all_indices[0]
        
        return all_candidates, all_indices
    
    def query_radius(
        self,
        target_pose: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query all candidate joint configurations within epsilon of target pose.
        
        Parameters
        ----------
        target_pose : np.ndarray
            Target pose of shape (7,).
            
        Returns
        -------
        tuple
            (joint_configs, indices) of candidates within epsilon.
        """
        if self.kdtree is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")
        
        # Query all poses within position epsilon
        indices = self.kdtree.query_ball_point(target_pose[:3], self.epsilon_pos)
        
        if len(indices) == 0:
            return np.array([]), np.array([])
        
        indices = np.array(indices)
        
        # Filter by orientation epsilon
        ori_distances = quaternion_distance(target_pose[3:], self.poses[indices, 3:])
        valid_mask = ori_distances < self.epsilon_ori
        valid_indices = indices[valid_mask]
        
        if len(valid_indices) == 0:
            return np.array([]), np.array([])
        
        return self.joint_configs[valid_indices], valid_indices
    
    def get_multi_solution_stats(self) -> Dict[str, Any]:
        """Get statistics about multi-solution regions.
        
        Returns
        -------
        dict
            Statistics including:
            - 'n_clusters': Number of unique pose clusters
            - 'avg_solutions_per_cluster': Average number of solutions
            - 'max_solutions': Maximum number of solutions for a pose
            - 'multi_solution_percentage': % of clusters with >1 solution
        """
        if self._cluster_indices is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")
        
        n_clusters = len(self._cluster_indices)
        solutions_per_cluster = [len(c) for c in self._cluster_indices]
        
        return {
            'n_clusters': n_clusters,
            'avg_solutions_per_cluster': np.mean(solutions_per_cluster),
            'max_solutions': np.max(solutions_per_cluster),
            'min_solutions': np.min(solutions_per_cluster),
            'multi_solution_percentage': 100 * np.mean(np.array(solutions_per_cluster) > 1),
        }
    
    def save(self, output_path: Union[str, Path]):
        """Save the clusterer state to disk.
        
        Parameters
        ----------
        output_path : str or Path
            Output file path (.pt format).
        """
        if self.poses is None:
            raise ValueError("Clusterer not fitted. Call fit() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'poses': torch.from_numpy(self.poses),
            'joint_configs': torch.from_numpy(self.joint_configs),
            'cluster_representatives': torch.from_numpy(self._cluster_representatives),
            'cluster_indices': self._cluster_indices,
            'cluster_map': self.cluster_map,
            'epsilon_pos': self.epsilon_pos,
            'epsilon_ori': self.epsilon_ori,
        }
        
        torch.save(data, output_path)
    
    @classmethod
    def load(cls, input_path: Union[str, Path]) -> 'PoseClusterer':
        """Load a clusterer from disk.
        
        Parameters
        ----------
        input_path : str or Path
            Input file path.
            
        Returns
        -------
        PoseClusterer
            Loaded clusterer.
        """
        data = torch.load(input_path, weights_only=False)
        
        clusterer = cls(
            epsilon_pos=data['epsilon_pos'],
            epsilon_ori=data['epsilon_ori'],
        )
        
        clusterer.poses = data['poses'].numpy()
        clusterer.joint_configs = data['joint_configs'].numpy()
        clusterer._cluster_representatives = data['cluster_representatives'].numpy()
        clusterer._cluster_indices = data['cluster_indices']
        clusterer.cluster_map = data['cluster_map']
        
        # Rebuild KD-tree
        clusterer.kdtree = KDTree(clusterer.poses[:, :3])
        
        return clusterer
