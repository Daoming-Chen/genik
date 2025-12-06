"""Forward Kinematics Sampler for GeNIK.

This module provides functionality for sampling joint configurations and computing
their corresponding end-effector poses using forward kinematics.
"""

from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..urdf import URDF


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to quaternion (w, x, y, z).
    
    Parameters
    ----------
    R : np.ndarray
        Rotation matrix of shape (3, 3) or batch of rotation matrices (N, 3, 3).
        
    Returns
    -------
    np.ndarray
        Quaternion of shape (4,) or batch of quaternions (N, 4) in (w, x, y, z) format.
    """
    if R.ndim == 2:
        R = R[np.newaxis, :, :]
        squeeze = True
    else:
        squeeze = False
    
    batch_size = R.shape[0]
    q = np.zeros((batch_size, 4), dtype=R.dtype)
    
    # Compute trace
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s = np.sqrt(trace[mask1] + 1.0) * 2  # s = 4*w
    q[mask1, 0] = 0.25 * s  # w
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s  # x
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s  # y
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s  # z
    
    # Case 2: R[0,0] is largest diagonal
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s = np.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2  # s = 4*x
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s  # w
    q[mask2, 1] = 0.25 * s  # x
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s  # y
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s  # z
    
    # Case 3: R[1,1] is largest diagonal
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s = np.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2  # s = 4*y
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s  # w
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s  # x
    q[mask3, 2] = 0.25 * s  # y
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s  # z
    
    # Case 4: R[2,2] is largest diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s = np.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2  # s = 4*z
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s  # w
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s  # x
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s  # y
    q[mask4, 3] = 0.25 * s  # z
    
    # Normalize quaternions
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    
    if squeeze:
        q = q.squeeze(0)
    
    return q


def pose_from_transform(T: np.ndarray) -> np.ndarray:
    """Extract position and quaternion from a 4x4 transformation matrix.
    
    Parameters
    ----------
    T : np.ndarray
        Transformation matrix of shape (4, 4) or batch (N, 4, 4).
        
    Returns
    -------
    np.ndarray
        Pose vector of shape (7,) or (N, 7) containing [x, y, z, qw, qx, qy, qz].
    """
    if T.ndim == 2:
        T = T[np.newaxis, :, :]
        squeeze = True
    else:
        squeeze = False
    
    position = T[:, :3, 3]
    rotation = T[:, :3, :3]
    quaternion = rotation_matrix_to_quaternion(rotation)
    
    pose = np.concatenate([position, quaternion], axis=1)
    
    if squeeze:
        pose = pose.squeeze(0)
    
    return pose


class FKSampler:
    """Forward Kinematics Sampler for generating joint-pose datasets.
    
    This class provides functionality for uniform sampling of joint configurations
    within joint limits and computing their corresponding end-effector poses.
    
    Parameters
    ----------
    urdf_path : str or Path
        Path to the robot URDF file.
    end_link : str, optional
        Name of the end-effector link. If None, uses the last link in the chain.
        
    Attributes
    ----------
    robot : URDF
        The loaded robot model.
    n_joints : int
        Number of actuated joints.
    joint_limits : np.ndarray
        Joint limits of shape (n_joints, 2) containing [lower, upper] bounds.
    joint_names : list
        Names of actuated joints.
        
    Examples
    --------
    >>> sampler = FKSampler("robots/panda_arm.urdf")
    >>> samples = sampler.sample(num_samples=1000, seed=42)
    >>> print(samples['joint_configs'].shape)  # (1000, 7)
    >>> print(samples['poses'].shape)  # (1000, 7)
    """
    
    def __init__(
        self,
        urdf_path: Union[str, Path],
        end_link: Optional[str] = None,
    ):
        self.urdf_path = Path(urdf_path)
        self.robot = URDF.load(str(self.urdf_path))
        
        # Get end-effector link
        if end_link is None:
            self.end_link = self.robot.end_links[0]
        else:
            self.end_link = self.robot._link_map[end_link]
        
        # Get joint information
        self.n_joints = len(self.robot.actuated_joints)
        self.joint_limits = self.robot.joint_limits
        self.joint_names = self.robot.actuated_joint_names
        
        # Replace infinite limits with reasonable defaults
        self._process_joint_limits()
    
    def _process_joint_limits(self):
        """Process joint limits, replacing infinite bounds with defaults."""
        for i in range(self.n_joints):
            if np.isinf(self.joint_limits[i, 0]):
                self.joint_limits[i, 0] = -2 * np.pi
            if np.isinf(self.joint_limits[i, 1]):
                self.joint_limits[i, 1] = 2 * np.pi
    
    def sample_joint_configs(
        self,
        num_samples: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Sample random joint configurations within joint limits.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        seed : int, optional
            Random seed for reproducibility.
            
        Returns
        -------
        np.ndarray
            Joint configurations of shape (num_samples, n_joints).
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Uniform sampling within joint limits
        lower = self.joint_limits[:, 0]
        upper = self.joint_limits[:, 1]
        
        joint_configs = np.random.uniform(
            lower, upper, size=(num_samples, self.n_joints)
        ).astype(np.float32)
        
        return joint_configs
    
    def compute_fk(
        self,
        joint_configs: np.ndarray,
        batch_size: int = 1000,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """Compute forward kinematics for joint configurations.
        
        Parameters
        ----------
        joint_configs : np.ndarray
            Joint configurations of shape (N, n_joints).
        batch_size : int
            Batch size for processing.
        use_gpu : bool
            Whether to use GPU acceleration.
        device : torch.device, optional
            PyTorch device for GPU computation.
            
        Returns
        -------
        np.ndarray
            End-effector poses of shape (N, 7) containing [x, y, z, qw, qx, qy, qz].
        """
        num_samples = len(joint_configs)
        poses = np.zeros((num_samples, 7), dtype=np.float32)
        
        if use_gpu and device is not None:
            # GPU batch FK
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)
                batch_configs = joint_configs[i:end]
                
                transforms = self.robot.link_fk_batch_gpu(
                    batch_configs, device, link=self.end_link
                )
                # Convert to numpy and extract poses
                transforms_np = transforms.cpu().numpy()
                poses[i:end] = pose_from_transform(transforms_np)
        else:
            # CPU batch FK
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)
                batch_configs = joint_configs[i:end]
                
                transforms = self.robot.link_fk_batch(
                    batch_configs, link=self.end_link
                )
                poses[i:end] = pose_from_transform(transforms)
        
        return poses
    
    def sample(
        self,
        num_samples: int,
        seed: Optional[int] = None,
        batch_size: int = 1000,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Sample joint configurations and compute their FK poses.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        seed : int, optional
            Random seed for reproducibility.
        batch_size : int
            Batch size for FK computation.
        use_gpu : bool
            Whether to use GPU acceleration.
        device : torch.device, optional
            PyTorch device for GPU computation.
        show_progress : bool
            Whether to show progress bar.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'joint_configs': np.ndarray of shape (N, n_joints)
            - 'poses': np.ndarray of shape (N, 7)
            - 'metadata': dict with sampling parameters
        """
        # Sample joint configurations
        joint_configs = self.sample_joint_configs(num_samples, seed)
        
        # Compute FK
        if show_progress:
            poses = np.zeros((num_samples, 7), dtype=np.float32)
            pbar = tqdm(total=num_samples, desc="Computing FK")
            
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)
                batch_configs = joint_configs[i:end]
                
                if use_gpu and device is not None:
                    transforms = self.robot.link_fk_batch_gpu(
                        batch_configs, device, link=self.end_link
                    )
                    transforms_np = transforms.cpu().numpy()
                else:
                    transforms = self.robot.link_fk_batch(
                        batch_configs, link=self.end_link
                    )
                    transforms_np = transforms
                
                poses[i:end] = pose_from_transform(transforms_np)
                pbar.update(end - i)
            
            pbar.close()
        else:
            poses = self.compute_fk(
                joint_configs, batch_size, use_gpu, device
            )
        
        metadata = {
            'robot_name': self.robot.name,
            'urdf_path': str(self.urdf_path),
            'end_link': self.end_link.name,
            'n_joints': self.n_joints,
            'joint_names': self.joint_names,
            'joint_limits': self.joint_limits.tolist(),
            'num_samples': num_samples,
            'seed': seed,
        }
        
        return {
            'joint_configs': joint_configs,
            'poses': poses,
            'metadata': metadata,
        }
    
    def validate_sample(
        self,
        joint_config: np.ndarray,
        pose: np.ndarray,
        tolerance: float = 1e-6,
    ) -> Tuple[bool, float]:
        """Validate that a joint configuration produces the expected pose.
        
        Parameters
        ----------
        joint_config : np.ndarray
            Joint configuration of shape (n_joints,).
        pose : np.ndarray
            Expected pose of shape (7,).
        tolerance : float
            Position tolerance in meters.
            
        Returns
        -------
        tuple
            (is_valid, error) where error is the position error.
        """
        # Compute FK
        transforms = self.robot.link_fk_batch(
            joint_config[np.newaxis, :], link=self.end_link
        )
        computed_pose = pose_from_transform(transforms[0])
        
        # Compare positions
        pos_error = np.linalg.norm(computed_pose[:3] - pose[:3])
        
        return pos_error < tolerance, pos_error
    
    @staticmethod
    def save(
        samples: Dict[str, Any],
        output_path: Union[str, Path],
    ):
        """Save samples to a .pt file.
        
        Parameters
        ----------
        samples : dict
            Samples dictionary from sample() method.
        output_path : str or Path
            Output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to tensors for saving
        data = {
            'joint_configs': torch.from_numpy(samples['joint_configs']),
            'poses': torch.from_numpy(samples['poses']),
            'metadata': samples['metadata'],
        }
        
        torch.save(data, output_path)
    
    @staticmethod
    def load(
        input_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Load samples from a .pt file.
        
        Parameters
        ----------
        input_path : str or Path
            Input file path.
            
        Returns
        -------
        dict
            Samples dictionary.
        """
        data = torch.load(input_path, weights_only=False)
        
        return {
            'joint_configs': data['joint_configs'].numpy(),
            'poses': data['poses'].numpy(),
            'metadata': data['metadata'],
        }
