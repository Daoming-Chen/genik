"""Differentiable Forward Kinematics Layer for GeNIK.

This module provides a PyTorch-compatible forward kinematics layer that
supports gradient backpropagation for end-to-end training.
"""

from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def rotation_matrix_to_quaternion_torch(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions (w, x, y, z).
    
    Parameters
    ----------
    R : torch.Tensor
        Rotation matrices of shape (N, 3, 3).
        
    Returns
    -------
    torch.Tensor
        Quaternions of shape (N, 4) in (w, x, y, z) format.
    """
    batch_size = R.shape[0]
    device = R.device
    dtype = R.dtype
    
    q = torch.zeros(batch_size, 4, device=device, dtype=dtype)
    
    # Compute trace
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s
    
    # Case 2: R[0,0] is largest diagonal
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s
    
    # Case 3: R[1,1] is largest diagonal
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s
    
    # Case 4: R[2,2] is largest diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        q[mask4, 3] = 0.25 * s
    
    # Normalize
    q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)
    
    return q


def rodrigues_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Compute rotation matrix using Rodrigues' formula.
    
    Parameters
    ----------
    axis : torch.Tensor
        Rotation axis of shape (3,) or (N, 3), normalized.
    angle : torch.Tensor
        Rotation angle in radians of shape () or (N,).
        
    Returns
    -------
    torch.Tensor
        Rotation matrix of shape (3, 3) or (N, 3, 3).
    """
    # Ensure proper dimensions
    if axis.dim() == 1:
        axis = axis.unsqueeze(0)
    if angle.dim() == 0:
        angle = angle.unsqueeze(0)
    
    batch_size = max(axis.shape[0], angle.shape[0])
    device = axis.device
    dtype = axis.dtype
    
    # Broadcast if needed
    if axis.shape[0] == 1 and batch_size > 1:
        axis = axis.expand(batch_size, -1)
    if angle.shape[0] == 1 and batch_size > 1:
        angle = angle.expand(batch_size)
    
    # Skew-symmetric matrix
    K = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    
    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
    sin_theta = torch.sin(angle).view(-1, 1, 1)
    cos_theta = torch.cos(angle).view(-1, 1, 1)
    
    R = eye + sin_theta * K + (1 - cos_theta) * torch.bmm(K, K)
    
    if batch_size == 1:
        R = R.squeeze(0)
    
    return R


class JointTransform:
    """Represents a joint transformation for differentiable FK."""
    
    def __init__(
        self,
        joint_type: str,
        axis: np.ndarray,
        origin: np.ndarray,
    ):
        """Initialize joint transform.
        
        Parameters
        ----------
        joint_type : str
            Type of joint: 'revolute', 'continuous', 'prismatic', or 'fixed'.
        axis : np.ndarray
            Joint axis of shape (3,).
        origin : np.ndarray
            Origin transform of shape (4, 4).
        """
        self.joint_type = joint_type
        self.axis = axis.astype(np.float32)
        self.origin = origin.astype(np.float32)
    
    def compute_transform(
        self,
        angle: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the joint transformation for given angle(s).
        
        Parameters
        ----------
        angle : torch.Tensor
            Joint angle(s) of shape () or (N,).
        device : torch.device
            Device for computation.
            
        Returns
        -------
        torch.Tensor
            Transformation matrix of shape (4, 4) or (N, 4, 4).
        """
        is_batched = angle.dim() > 0 and angle.shape[0] > 1
        batch_size = angle.shape[0] if is_batched else 1
        
        # Origin transform
        origin = torch.tensor(self.origin, device=device, dtype=torch.float32)
        
        if self.joint_type == 'fixed':
            if is_batched:
                return origin.unsqueeze(0).expand(batch_size, -1, -1)
            return origin
        
        axis = torch.tensor(self.axis, device=device, dtype=torch.float32)
        
        if self.joint_type in ['revolute', 'continuous']:
            # Rotation about axis
            R = rodrigues_rotation(axis, angle)
            
            if is_batched:
                T = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1).clone()
                T[:, :3, :3] = R
            else:
                T = torch.eye(4, device=device, dtype=torch.float32)
                T[:3, :3] = R
            
        elif self.joint_type == 'prismatic':
            # Translation along axis
            if is_batched:
                T = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1).clone()
                T[:, :3, 3] = axis.unsqueeze(0) * angle.unsqueeze(1)
            else:
                T = torch.eye(4, device=device, dtype=torch.float32)
                T[:3, 3] = axis * angle
        else:
            raise ValueError(f"Unsupported joint type: {self.joint_type}")
        
        # Apply origin transform
        if is_batched:
            origin = origin.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.bmm(origin, T)
        else:
            return torch.mm(origin, T)


class DifferentiableFKLayer(nn.Module):
    """Differentiable Forward Kinematics Layer.
    
    This layer computes forward kinematics in a fully differentiable manner,
    enabling gradient-based optimization of joint angles.
    
    Parameters
    ----------
    urdf_path : str or Path
        Path to the robot URDF file.
    end_link : str, optional
        Name of the end-effector link.
        
    Examples
    --------
    >>> fk_layer = DifferentiableFKLayer("robots/panda_arm.urdf")
    >>> joint_angles = torch.randn(32, 7, requires_grad=True)
    >>> poses = fk_layer(joint_angles)
    >>> loss = poses[:, :3].sum()  # Position loss
    >>> loss.backward()  # Gradients flow to joint_angles
    """
    
    def __init__(
        self,
        urdf_path: Union[str, Path],
        end_link: Optional[str] = None,
    ):
        super().__init__()
        
        self.urdf_path = str(urdf_path)
        
        # Import URDF here to avoid circular imports
        from ..urdf import URDF
        
        self.robot = URDF.load(self.urdf_path)
        
        # Get end-effector link
        if end_link is None:
            self.end_link = self.robot.end_links[0]
        else:
            self.end_link = self.robot._link_map[end_link]
        self.end_link_name = self.end_link.name
        
        # Extract kinematic chain
        self.joint_transforms, self.joint_indices = self._extract_kinematic_chain()
        self.n_joints = len(self.robot.actuated_joints)
        
        # Register joint limits as buffer
        limits = self.robot.joint_limits
        self.register_buffer('joint_lower', torch.tensor(limits[:, 0], dtype=torch.float32))
        self.register_buffer('joint_upper', torch.tensor(limits[:, 1], dtype=torch.float32))
    
    def _extract_kinematic_chain(self) -> Tuple[List[JointTransform], List[int]]:
        """Extract the kinematic chain from base to end-effector."""
        joint_transforms = []
        joint_indices = []  # Index into actuated joints, -1 for fixed
        
        # Get path from end-effector to base
        path = self.robot._paths_to_base[self.end_link]
        
        # Build transforms from base to end-effector (reverse order)
        for i in range(len(path) - 1, 0, -1):
            child = path[i - 1]
            parent = path[i]
            joint = self.robot._G.get_edge_data(child, parent)['joint']
            
            # Get joint properties
            axis = joint.axis if joint.axis is not None else np.array([1, 0, 0])
            origin = joint.origin if joint.origin is not None else np.eye(4)
            
            jt = JointTransform(
                joint_type=joint.joint_type,
                axis=axis,
                origin=origin,
            )
            joint_transforms.append(jt)
            
            # Find index in actuated joints
            if joint.joint_type in ['fixed']:
                joint_indices.append(-1)
            else:
                try:
                    idx = self.robot.actuated_joints.index(joint)
                    joint_indices.append(idx)
                except ValueError:
                    joint_indices.append(-1)
        
        return joint_transforms, joint_indices
    
    def forward(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """Compute forward kinematics.
        
        Parameters
        ----------
        joint_angles : torch.Tensor
            Joint angles of shape (N, n_joints) or (n_joints,).
            
        Returns
        -------
        torch.Tensor
            End-effector poses of shape (N, 7) or (7,) containing
            [x, y, z, qw, qx, qy, qz].
        """
        squeeze = False
        if joint_angles.dim() == 1:
            joint_angles = joint_angles.unsqueeze(0)
            squeeze = True
        
        batch_size = joint_angles.shape[0]
        device = joint_angles.device
        
        # Start with identity transform
        T = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        
        # Apply each joint transform
        for jt, joint_idx in zip(self.joint_transforms, self.joint_indices):
            if joint_idx == -1:
                # Fixed joint
                angle = torch.zeros(batch_size, device=device, dtype=torch.float32)
            else:
                angle = joint_angles[:, joint_idx]
            
            joint_T = jt.compute_transform(angle, device)
            T = torch.bmm(T, joint_T)
        
        # Extract pose
        position = T[:, :3, 3]
        rotation = T[:, :3, :3]
        quaternion = rotation_matrix_to_quaternion_torch(rotation)
        
        pose = torch.cat([position, quaternion], dim=1)
        
        if squeeze:
            pose = pose.squeeze(0)
        
        return pose
    
    def compute_jacobian(
        self,
        joint_angles: torch.Tensor,
        method: str = 'autograd',
    ) -> torch.Tensor:
        """Compute the Jacobian of the FK function.
        
        Parameters
        ----------
        joint_angles : torch.Tensor
            Joint angles of shape (N, n_joints) or (n_joints,).
        method : str
            Method for Jacobian computation: 'autograd' or 'finite_diff'.
            
        Returns
        -------
        torch.Tensor
            Jacobian of shape (N, 7, n_joints) or (7, n_joints).
        """
        squeeze = False
        if joint_angles.dim() == 1:
            joint_angles = joint_angles.unsqueeze(0)
            squeeze = True
        
        if method == 'autograd':
            # Use PyTorch autograd for Jacobian computation
            joint_angles = joint_angles.requires_grad_(True)
            poses = self(joint_angles)
            
            jacobians = []
            for i in range(poses.shape[1]):  # For each output dimension
                grad = torch.autograd.grad(
                    poses[:, i].sum(),
                    joint_angles,
                    retain_graph=True,
                )[0]
                jacobians.append(grad)
            
            jacobian = torch.stack(jacobians, dim=1)
        
        elif method == 'finite_diff':
            # Finite difference approximation
            eps = 1e-6
            batch_size = joint_angles.shape[0]
            
            base_pose = self(joint_angles)
            jacobian = torch.zeros(batch_size, 7, self.n_joints, device=joint_angles.device)
            
            for j in range(self.n_joints):
                angles_plus = joint_angles.clone()
                angles_plus[:, j] += eps
                pose_plus = self(angles_plus)
                jacobian[:, :, j] = (pose_plus - base_pose) / eps
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if squeeze:
            jacobian = jacobian.squeeze(0)
        
        return jacobian
    
    def get_joint_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get joint limits.
        
        Returns
        -------
        tuple
            (lower_limits, upper_limits) tensors of shape (n_joints,).
        """
        return self.joint_lower, self.joint_upper
    
    def clip_to_limits(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """Clip joint angles to limits.
        
        Parameters
        ----------
        joint_angles : torch.Tensor
            Joint angles to clip.
            
        Returns
        -------
        torch.Tensor
            Clipped joint angles.
        """
        return torch.clamp(joint_angles, self.joint_lower, self.joint_upper)


def validate_diff_fk(
    urdf_path: Union[str, Path],
    num_samples: int = 100,
    eps: float = 1e-4,
) -> Dict[str, Any]:
    """Validate the differentiable FK layer against analytical FK.
    
    Parameters
    ----------
    urdf_path : str or Path
        Path to robot URDF file.
    num_samples : int
        Number of random configurations to test.
    eps : float
        Tolerance for validation.
        
    Returns
    -------
    dict
        Validation results including max errors and gradient check.
    """
    from ..urdf import URDF
    from ..data.sampler import pose_from_transform
    
    # Load robot and create FK layer
    robot = URDF.load(str(urdf_path))
    fk_layer = DifferentiableFKLayer(urdf_path)
    
    # Get joint limits
    joint_limits = robot.joint_limits
    n_joints = len(robot.actuated_joints)
    
    # Generate random configurations
    np.random.seed(42)
    configs = np.random.uniform(
        joint_limits[:, 0],
        joint_limits[:, 1],
        size=(num_samples, n_joints),
    ).astype(np.float32)
    
    # Compare FK results
    max_pos_error = 0.0
    max_ori_error = 0.0
    
    for config in configs:
        # Analytical FK
        transform = robot.link_fk(config, link=fk_layer.end_link)
        analytical_pose = pose_from_transform(transform)
        
        # Differentiable FK
        config_tensor = torch.from_numpy(config)
        diff_pose = fk_layer(config_tensor).detach().numpy()
        
        # Compare
        pos_error = np.linalg.norm(diff_pose[:3] - analytical_pose[:3])
        ori_error = 1 - np.abs(np.dot(diff_pose[3:], analytical_pose[3:]))
        
        max_pos_error = max(max_pos_error, pos_error)
        max_ori_error = max(max_ori_error, ori_error)
    
    # Gradient check
    config_tensor = torch.from_numpy(configs[0]).requires_grad_(True)
    pose = fk_layer(config_tensor)
    loss = pose[:3].sum()  # Position loss
    loss.backward()
    
    grad_exists = config_tensor.grad is not None and config_tensor.grad.abs().sum() > 0
    
    return {
        'max_position_error': max_pos_error,
        'max_orientation_error': max_ori_error,
        'position_valid': max_pos_error < eps,
        'orientation_valid': max_ori_error < eps,
        'gradient_exists': grad_exists,
    }
