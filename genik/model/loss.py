"""Mixed Physics Loss Functions for GeNIK Training.

This module provides loss functions that combine joint-space guidance
with task-space physical constraints.
"""

from typing import Optional, Tuple, Dict, Any, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diff_fk import DifferentiableFKLayer


def quaternion_distance_torch(
    q1: torch.Tensor,
    q2: torch.Tensor,
) -> torch.Tensor:
    """Compute geodesic distance between quaternions.
    
    Parameters
    ----------
    q1 : torch.Tensor
        First quaternion(s) of shape (N, 4) in (w, x, y, z) format.
    q2 : torch.Tensor
        Second quaternion(s) of shape (N, 4) in (w, x, y, z) format.
        
    Returns
    -------
    torch.Tensor
        Angular distance(s) in radians of shape (N,).
    """
    # Handle antipodal ambiguity
    dot = (q1 * q2).sum(dim=-1)
    dot = torch.clamp(torch.abs(dot), -1.0, 1.0)
    
    # Angular distance
    angle = 2.0 * torch.acos(dot)
    
    return angle


def pose_error_torch(
    pose_pred: torch.Tensor,
    pose_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute position and orientation errors between poses.
    
    Parameters
    ----------
    pose_pred : torch.Tensor
        Predicted poses of shape (N, 7) or (7,).
    pose_target : torch.Tensor
        Target poses of shape (N, 7) or (7,).
        
    Returns
    -------
    tuple
        (position_error, orientation_error) tensors.
    """
    # Position error (Euclidean distance)
    pos_error = torch.norm(pose_pred[..., :3] - pose_target[..., :3], dim=-1)
    
    # Orientation error (geodesic distance)
    ori_error = quaternion_distance_torch(pose_pred[..., 3:], pose_target[..., 3:])
    
    return pos_error, ori_error


class JointLoss(nn.Module):
    """Joint-space guidance loss.
    
    Parameters
    ----------
    loss_type : str
        Type of loss: 'mse' or 'l1'.
    reduction : str
        Reduction method: 'mean', 'sum', or 'none'.
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',
        reduction: str = 'mean',
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        q_pred: torch.Tensor,
        q_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute joint-space loss.
        
        Parameters
        ----------
        q_pred : torch.Tensor
            Predicted joint configurations of shape (N, n_joints).
        q_target : torch.Tensor
            Target joint configurations of shape (N, n_joints).
            
        Returns
        -------
        torch.Tensor
            Loss value.
        """
        return self.loss_fn(q_pred, q_target)


class PoseLoss(nn.Module):
    """Task-space pose loss using differentiable FK.
    
    Parameters
    ----------
    fk_layer : DifferentiableFKLayer
        Differentiable FK layer for pose computation.
    w_pos : float
        Weight for position error.
    w_ori : float
        Weight for orientation error.
    reduction : str
        Reduction method: 'mean', 'sum', or 'none'.
    """
    
    def __init__(
        self,
        fk_layer: DifferentiableFKLayer,
        w_pos: float = 1.0,
        w_ori: float = 1.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        
        self.fk_layer = fk_layer
        self.w_pos = w_pos
        self.w_ori = w_ori
        self.reduction = reduction
    
    def forward(
        self,
        q_pred: torch.Tensor,
        x_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute pose loss.
        
        Parameters
        ----------
        q_pred : torch.Tensor
            Predicted joint configurations of shape (N, n_joints).
        x_target : torch.Tensor
            Target poses of shape (N, 7).
            
        Returns
        -------
        tuple
            (loss, metrics) where metrics contains position and orientation errors.
        """
        # Compute FK for predicted joints
        x_pred = self.fk_layer(q_pred)
        
        # Compute errors
        pos_error, ori_error = pose_error_torch(x_pred, x_target)
        
        # Combined loss
        pos_loss = self.w_pos * (pos_error ** 2)
        ori_loss = self.w_ori * (ori_error ** 2)
        loss = pos_loss + ori_loss
        
        # Reduce
        if self.reduction == 'mean':
            loss = loss.mean()
            pos_loss = pos_loss.mean()
            ori_loss = ori_loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            pos_loss = pos_loss.sum()
            ori_loss = ori_loss.sum()
        
        metrics = {
            'pos_loss': pos_loss,
            'ori_loss': ori_loss,
            'pos_error': pos_error.mean(),
            'ori_error': ori_error.mean(),
        }
        
        return loss, metrics


class AdaptiveWeightScheduler:
    """Scheduler for adaptive loss weight lambda.
    
    Implements linear warmup: λ(t) = λ_min + (λ_max - λ_min) × min(t / T, 1.0)
    
    Parameters
    ----------
    lambda_min : float
        Minimum weight value (start of training).
    lambda_max : float
        Maximum weight value (end of warmup).
    warmup_epochs : int
        Number of epochs for warmup.
    """
    
    def __init__(
        self,
        lambda_min: float = 0.1,
        lambda_max: float = 1.0,
        warmup_epochs: int = 50,
    ):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """Update current epoch."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
    
    def get_lambda(self) -> float:
        """Get current lambda value."""
        t = min(self.current_epoch / self.warmup_epochs, 1.0)
        return self.lambda_min + (self.lambda_max - self.lambda_min) * t
    
    @property
    def current_lambda(self) -> float:
        """Current lambda value."""
        return self.get_lambda()


class GeNIKLoss(nn.Module):
    """Combined GeNIK loss with joint guidance and pose constraint.
    
    L_total = L_joint + λ × L_pose
    
    Parameters
    ----------
    fk_layer : DifferentiableFKLayer
        Differentiable FK layer for pose computation.
    lambda_min : float
        Minimum pose loss weight.
    lambda_max : float
        Maximum pose loss weight.
    warmup_epochs : int
        Number of warmup epochs.
    joint_loss_type : str
        Type of joint loss: 'mse' or 'l1'.
    w_pos : float
        Weight for position error in pose loss.
    w_ori : float
        Weight for orientation error in pose loss.
        
    Examples
    --------
    >>> fk_layer = DifferentiableFKLayer("robots/panda_arm.urdf")
    >>> criterion = GeNIKLoss(fk_layer)
    >>> loss, metrics = criterion(q_pred, q_star, x_target, epoch=10)
    """
    
    def __init__(
        self,
        fk_layer: DifferentiableFKLayer,
        lambda_min: float = 0.1,
        lambda_max: float = 1.0,
        warmup_epochs: int = 50,
        joint_loss_type: str = 'mse',
        w_pos: float = 1.0,
        w_ori: float = 1.0,
    ):
        super().__init__()
        
        self.joint_loss = JointLoss(loss_type=joint_loss_type)
        self.pose_loss = PoseLoss(fk_layer, w_pos=w_pos, w_ori=w_ori)
        self.scheduler = AdaptiveWeightScheduler(
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            warmup_epochs=warmup_epochs,
        )
    
    def forward(
        self,
        q_pred: torch.Tensor,
        q_star: torch.Tensor,
        x_target: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss.
        
        Parameters
        ----------
        q_pred : torch.Tensor
            Predicted joint configurations of shape (N, n_joints).
        q_star : torch.Tensor
            Ground truth joint configurations of shape (N, n_joints).
        x_target : torch.Tensor
            Target poses of shape (N, 7).
        epoch : int, optional
            Current epoch for adaptive weighting.
            
        Returns
        -------
        tuple
            (loss, metrics) where metrics contains component losses and errors.
        """
        # Update scheduler
        if epoch is not None:
            self.scheduler.step(epoch)
        
        # Compute losses
        joint_loss = self.joint_loss(q_pred, q_star)
        pose_loss, pose_metrics = self.pose_loss(q_pred, x_target)
        
        # Get current lambda
        lambda_val = self.scheduler.current_lambda
        
        # Combined loss
        total_loss = joint_loss + lambda_val * pose_loss
        
        # Collect metrics
        metrics = {
            'joint_loss': joint_loss,
            'pose_loss': pose_loss,
            'lambda': torch.tensor(lambda_val),
            **pose_metrics,
        }
        
        return total_loss, metrics
    
    def get_current_lambda(self) -> float:
        """Get current lambda value."""
        return self.scheduler.current_lambda


class SimplePoseLoss(nn.Module):
    """Simple pose loss without differentiable FK.
    
    Used when FK is computed externally (e.g., for evaluation).
    
    Parameters
    ----------
    w_pos : float
        Weight for position error.
    w_ori : float
        Weight for orientation error.
    """
    
    def __init__(
        self,
        w_pos: float = 1.0,
        w_ori: float = 1.0,
    ):
        super().__init__()
        self.w_pos = w_pos
        self.w_ori = w_ori
    
    def forward(
        self,
        pose_pred: torch.Tensor,
        pose_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute pose loss from precomputed poses.
        
        Parameters
        ----------
        pose_pred : torch.Tensor
            Predicted poses of shape (N, 7).
        pose_target : torch.Tensor
            Target poses of shape (N, 7).
            
        Returns
        -------
        tuple
            (loss, metrics).
        """
        pos_error, ori_error = pose_error_torch(pose_pred, pose_target)
        
        loss = (self.w_pos * pos_error ** 2 + self.w_ori * ori_error ** 2).mean()
        
        metrics = {
            'pos_error': pos_error.mean(),
            'ori_error': ori_error.mean(),
        }
        
        return loss, metrics
