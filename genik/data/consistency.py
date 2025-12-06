"""Consistency Matching for GeNIK Training Sample Generation.

This module provides functionality for generating training triplets
(q_ref, x_target, q*) using consistency-based matching.
"""

from typing import Optional, Tuple, Dict, Any, Union, List, Literal
from pathlib import Path
from enum import Enum

import numpy as np
import torch
from tqdm import tqdm

from .clustering import PoseClusterer


class SamplingStrategy(Enum):
    """Sampling strategy for reference configuration generation."""
    PERTURBATION = "perturbation"
    RANDOM = "random"
    MIXED = "mixed"


def joint_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance in joint space.
    
    Parameters
    ----------
    q1 : np.ndarray
        First configuration(s) of shape (n_joints,) or (N, n_joints).
    q2 : np.ndarray
        Second configuration(s) of shape (n_joints,) or (M, n_joints).
        
    Returns
    -------
    np.ndarray
        Distance(s) between configurations.
    """
    if q1.ndim == 1:
        q1 = q1[np.newaxis, :]
    if q2.ndim == 1:
        q2 = q2[np.newaxis, :]
    
    # Broadcast if needed for pairwise distances
    if q1.shape[0] == 1 or q2.shape[0] == 1:
        return np.linalg.norm(q1 - q2, axis=-1)
    
    # Compute pairwise distances
    return np.linalg.norm(q1[:, np.newaxis, :] - q2[np.newaxis, :, :], axis=-1)


class ConsistencyMatcher:
    """Consistency Matcher for generating training triplets.
    
    This class generates training samples using consistency-based matching,
    where the ground truth solution is the one closest to the reference input.
    
    Parameters
    ----------
    clusterer : PoseClusterer
        Fitted pose clusterer for candidate lookup.
    joint_limits : np.ndarray
        Joint limits of shape (n_joints, 2) containing [lower, upper] bounds.
    strategy : str
        Sampling strategy: 'perturbation', 'random', or 'mixed'.
    perturbation_std : float
        Standard deviation for perturbation noise.
    mixed_weights : tuple
        Weights for mixed strategy (perturbation, random, near-boundary).
        
    Examples
    --------
    >>> matcher = ConsistencyMatcher(clusterer, joint_limits)
    >>> triplets = matcher.generate(num_samples=10000)
    """
    
    def __init__(
        self,
        clusterer: PoseClusterer,
        joint_limits: np.ndarray,
        strategy: str = "mixed",
        perturbation_std: float = 0.3,
        mixed_weights: Tuple[float, float, float] = (0.6, 0.3, 0.1),
    ):
        self.clusterer = clusterer
        self.joint_limits = np.array(joint_limits, dtype=np.float32)
        self.n_joints = self.joint_limits.shape[0]
        
        # Validate strategy
        if strategy not in ["perturbation", "random", "mixed"]:
            raise ValueError(f"Invalid strategy: {strategy}")
        self.strategy = SamplingStrategy(strategy)
        
        self.perturbation_std = perturbation_std
        self.mixed_weights = mixed_weights
    
    def _sample_perturbation(
        self,
        q_base: np.ndarray,
        n_samples: int = 1,
    ) -> np.ndarray:
        """Generate reference configs by perturbing a base configuration.
        
        Parameters
        ----------
        q_base : np.ndarray
            Base joint configuration of shape (n_joints,).
        n_samples : int
            Number of perturbations to generate.
            
        Returns
        -------
        np.ndarray
            Perturbed configurations of shape (n_samples, n_joints).
        """
        noise = np.random.normal(0, self.perturbation_std, (n_samples, self.n_joints))
        q_perturbed = q_base + noise
        
        # Clip to joint limits
        q_perturbed = np.clip(
            q_perturbed,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1]
        )
        
        return q_perturbed.astype(np.float32)
    
    def _sample_random(self, n_samples: int = 1) -> np.ndarray:
        """Generate random reference configurations.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
            
        Returns
        -------
        np.ndarray
            Random configurations of shape (n_samples, n_joints).
        """
        lower = self.joint_limits[:, 0]
        upper = self.joint_limits[:, 1]
        
        q_random = np.random.uniform(
            lower, upper, size=(n_samples, self.n_joints)
        )
        
        return q_random.astype(np.float32)
    
    def _sample_near_boundary(self, n_samples: int = 1) -> np.ndarray:
        """Generate reference configurations near joint limits.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
            
        Returns
        -------
        np.ndarray
            Near-boundary configurations of shape (n_samples, n_joints).
        """
        lower = self.joint_limits[:, 0]
        upper = self.joint_limits[:, 1]
        range_ = upper - lower
        
        # Sample near boundaries (within 10% of limits)
        boundary_margin = 0.1 * range_
        
        q_boundary = np.zeros((n_samples, self.n_joints), dtype=np.float32)
        
        for j in range(self.n_joints):
            # Choose lower or upper boundary randomly
            choose_lower = np.random.random(n_samples) < 0.5
            
            # Sample near lower boundary
            q_boundary[choose_lower, j] = np.random.uniform(
                lower[j],
                lower[j] + boundary_margin[j],
                size=np.sum(choose_lower)
            )
            
            # Sample near upper boundary
            q_boundary[~choose_lower, j] = np.random.uniform(
                upper[j] - boundary_margin[j],
                upper[j],
                size=np.sum(~choose_lower)
            )
        
        return q_boundary
    
    def _find_closest_solution(
        self,
        q_ref: np.ndarray,
        candidates: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Find the candidate solution closest to the reference.
        
        Parameters
        ----------
        q_ref : np.ndarray
            Reference configuration of shape (n_joints,).
        candidates : np.ndarray
            Candidate solutions of shape (K, n_joints).
            
        Returns
        -------
        tuple
            (closest_solution, index) where closest_solution is the
            candidate with minimum distance to q_ref.
        """
        if len(candidates) == 0:
            return None, -1
        
        distances = np.linalg.norm(candidates - q_ref, axis=1)
        closest_idx = np.argmin(distances)
        
        return candidates[closest_idx], closest_idx
    
    def generate_single(
        self,
        pose_idx: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate a single training triplet.
        
        Parameters
        ----------
        pose_idx : int
            Index of the target pose in the clusterer.
            
        Returns
        -------
        tuple or None
            (q_ref, x_target, q_star) triplet, or None if no candidates.
        """
        # Get target pose and candidates
        target_pose = self.clusterer.poses[pose_idx]
        candidates, _ = self.clusterer.query_radius(target_pose)
        
        if len(candidates) == 0:
            return None
        
        # Generate reference configuration based on strategy
        if self.strategy == SamplingStrategy.PERTURBATION:
            # Pick a random candidate and perturb it
            base_idx = np.random.randint(len(candidates))
            q_ref = self._sample_perturbation(candidates[base_idx], n_samples=1)[0]
            
        elif self.strategy == SamplingStrategy.RANDOM:
            q_ref = self._sample_random(n_samples=1)[0]
            
        else:  # MIXED
            strategy_choice = np.random.random()
            if strategy_choice < self.mixed_weights[0]:
                # Perturbation
                base_idx = np.random.randint(len(candidates))
                q_ref = self._sample_perturbation(candidates[base_idx], n_samples=1)[0]
            elif strategy_choice < self.mixed_weights[0] + self.mixed_weights[1]:
                # Random
                q_ref = self._sample_random(n_samples=1)[0]
            else:
                # Near-boundary
                q_ref = self._sample_near_boundary(n_samples=1)[0]
        
        # Find closest solution
        q_star, _ = self._find_closest_solution(q_ref, candidates)
        
        if q_star is None:
            return None
        
        return q_ref, target_pose, q_star
    
    def generate(
        self,
        num_samples: int,
        samples_per_pose: int = 3,
        seed: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Generate training triplets.
        
        Parameters
        ----------
        num_samples : int
            Target number of triplets to generate.
        samples_per_pose : int
            Number of samples to generate per unique pose.
        seed : int, optional
            Random seed for reproducibility.
        show_progress : bool
            Whether to show progress bar.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'q_ref': Reference configurations (N, n_joints)
            - 'x_target': Target poses (N, 7)
            - 'q_star': Ground truth solutions (N, n_joints)
            - 'metadata': Generation parameters
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_poses = len(self.clusterer.poses)
        
        # Allocate storage
        q_refs = []
        x_targets = []
        q_stars = []
        
        # Calculate how many poses to sample
        # Need to sample enough poses to generate num_samples triplets
        poses_needed = (num_samples + samples_per_pose - 1) // samples_per_pose
        poses_to_sample = min(poses_needed, n_poses)
        
        # Sample poses (with replacement if we need more samples than poses)
        replace = poses_needed > n_poses
        pose_indices = np.random.choice(n_poses, poses_to_sample, replace=replace)
        
        iterator = pose_indices
        if show_progress:
            iterator = tqdm(iterator, desc="Generating triplets")
        
        for pose_idx in iterator:
            for _ in range(samples_per_pose):
                result = self.generate_single(pose_idx)
                
                if result is not None:
                    q_ref, x_target, q_star = result
                    q_refs.append(q_ref)
                    x_targets.append(x_target)
                    q_stars.append(q_star)
                
                if len(q_refs) >= num_samples:
                    break
            
            if len(q_refs) >= num_samples:
                break
        
        # Convert to arrays
        q_refs = np.array(q_refs, dtype=np.float32)
        x_targets = np.array(x_targets, dtype=np.float32)
        q_stars = np.array(q_stars, dtype=np.float32)
        
        metadata = {
            'num_samples': len(q_refs),
            'samples_per_pose': samples_per_pose,
            'strategy': self.strategy.value,
            'perturbation_std': self.perturbation_std,
            'mixed_weights': self.mixed_weights,
            'n_joints': self.n_joints,
            'seed': seed,
        }
        
        return {
            'q_ref': q_refs,
            'x_target': x_targets,
            'q_star': q_stars,
            'metadata': metadata,
        }
    
    @staticmethod
    def save(
        triplets: Dict[str, Any],
        output_path: Union[str, Path],
    ):
        """Save triplets to disk.
        
        Parameters
        ----------
        triplets : dict
            Triplets dictionary from generate() method.
        output_path : str or Path
            Output file path (.pt format).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'q_ref': torch.from_numpy(triplets['q_ref']),
            'x_target': torch.from_numpy(triplets['x_target']),
            'q_star': torch.from_numpy(triplets['q_star']),
            'metadata': triplets['metadata'],
        }
        
        torch.save(data, output_path)
    
    @staticmethod
    def load(input_path: Union[str, Path]) -> Dict[str, Any]:
        """Load triplets from disk.
        
        Parameters
        ----------
        input_path : str or Path
            Input file path.
            
        Returns
        -------
        dict
            Triplets dictionary.
        """
        data = torch.load(input_path, weights_only=False)
        
        return {
            'q_ref': data['q_ref'].numpy(),
            'x_target': data['x_target'].numpy(),
            'q_star': data['q_star'].numpy(),
            'metadata': data['metadata'],
        }
