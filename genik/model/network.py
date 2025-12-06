"""IK Network Architecture for GeNIK.

This module provides the neural network architectures for the GeNIK
inverse kinematics solver.
"""

from typing import Optional, Tuple, Dict, Any, Union, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with optional dropout.
    
    Parameters
    ----------
    hidden_size : int
        Size of hidden layers.
    dropout : float
        Dropout probability.
    use_batch_norm : bool
        Whether to use batch normalization.
    """
    
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        
        layers = [nn.Linear(hidden_size, hidden_size)]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return x + self.block(x)


class IKNetwork(nn.Module):
    """Conditional IK Network for GeNIK.
    
    This network predicts joint configurations given a reference configuration
    and target end-effector pose.
    
    Parameters
    ----------
    n_joints : int
        Number of robot joints (DOF).
    pose_dim : int
        Dimension of pose vector (default: 7 for position + quaternion).
    hidden_size : int
        Size of hidden layers.
    num_blocks : int
        Number of residual blocks.
    dropout : float
        Dropout probability.
    use_batch_norm : bool
        Whether to use batch normalization.
    use_residual_output : bool
        Whether to use residual connection from input to output.
    joint_limits : tuple, optional
        (lower, upper) joint limits for output clamping.
        
    Examples
    --------
    >>> model = IKNetwork(n_joints=7)
    >>> q_ref = torch.randn(32, 7)
    >>> x_target = torch.randn(32, 7)
    >>> q_pred = model(q_ref, x_target)
    >>> print(q_pred.shape)  # (32, 7)
    """
    
    def __init__(
        self,
        n_joints: int,
        pose_dim: int = 7,
        hidden_size: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.1,
        use_batch_norm: bool = False,
        use_residual_output: bool = True,
        joint_limits: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        super().__init__()
        
        self.n_joints = n_joints
        self.pose_dim = pose_dim
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.use_residual_output = use_residual_output
        
        # Input dimension: q_ref + x_target
        input_dim = n_joints + pose_dim
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout, use_batch_norm)
            for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden_size, n_joints)
        
        # Joint limits
        if joint_limits is not None:
            self.register_buffer('joint_lower', joint_limits[0])
            self.register_buffer('joint_upper', joint_limits[1])
        else:
            self.joint_lower = None
            self.joint_upper = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Small initialization for output layer (for residual learning)
        if self.use_residual_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
    
    def forward(
        self,
        q_ref: torch.Tensor,
        x_target: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        q_ref : torch.Tensor
            Reference joint configuration of shape (N, n_joints) or (n_joints,).
        x_target : torch.Tensor
            Target pose of shape (N, pose_dim) or (pose_dim,).
            
        Returns
        -------
        torch.Tensor
            Predicted joint configuration of shape (N, n_joints) or (n_joints,).
        """
        # Handle single sample
        squeeze = False
        if q_ref.dim() == 1:
            q_ref = q_ref.unsqueeze(0)
            x_target = x_target.unsqueeze(0)
            squeeze = True
        
        # Concatenate inputs
        x = torch.cat([q_ref, x_target], dim=1)
        
        # Embedding
        h = self.embedding(x)
        
        # Residual blocks
        for block in self.blocks:
            h = block(h)
        
        # Output
        delta_q = self.output(h)
        
        # Residual connection
        if self.use_residual_output:
            q_pred = q_ref + delta_q
        else:
            q_pred = delta_q
        
        # Clamp to joint limits
        if self.joint_lower is not None and self.joint_upper is not None:
            q_pred = torch.clamp(q_pred, self.joint_lower, self.joint_upper)
        
        if squeeze:
            q_pred = q_pred.squeeze(0)
        
        return q_pred
    
    def get_param_count(self) -> Dict[str, int]:
        """Get parameter counts.
        
        Returns
        -------
        dict
            Dictionary with 'total', 'trainable', and per-layer counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        layer_counts = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                count = sum(p.numel() for p in module.parameters())
                layer_counts[name] = count
        
        return {
            'total': total,
            'trainable': trainable,
            'layers': layer_counts,
        }
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Save model checkpoint.
        
        Parameters
        ----------
        path : str or Path
            Output file path.
        optimizer : Optimizer, optional
            Optimizer to save state from.
        epoch : int, optional
            Current epoch number.
        metrics : dict, optional
            Training metrics to save.
        config : dict, optional
            Model configuration to save.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'n_joints': self.n_joints,
            'pose_dim': self.pose_dim,
            'hidden_size': self.hidden_size,
            'num_blocks': self.num_blocks,
            'use_residual_output': self.use_residual_output,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics
        if config is not None:
            checkpoint['config'] = config
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> Tuple['IKNetwork', Dict[str, Any]]:
        """Load model from checkpoint.
        
        Parameters
        ----------
        path : str or Path
            Checkpoint file path.
        device : torch.device, optional
            Device to load model to.
        strict : bool
            Whether to enforce strict state dict matching.
            
        Returns
        -------
        tuple
            (model, checkpoint_dict) where checkpoint_dict contains
            optimizer state, epoch, metrics, etc.
        """
        if device is None:
            device = torch.device('cpu')
        
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Create model
        model = cls(
            n_joints=checkpoint['n_joints'],
            pose_dim=checkpoint.get('pose_dim', 7),
            hidden_size=checkpoint['hidden_size'],
            num_blocks=checkpoint['num_blocks'],
            use_residual_output=checkpoint.get('use_residual_output', True),
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        model.to(device)
        
        return model, checkpoint


class LargeIKNetwork(nn.Module):
    """Larger IK Network with more capacity for complex robots.
    
    This network uses a deeper architecture with skip connections
    across multiple levels.
    
    Parameters
    ----------
    n_joints : int
        Number of robot joints.
    pose_dim : int
        Dimension of pose vector.
    hidden_sizes : list
        List of hidden layer sizes.
    dropout : float
        Dropout probability.
    """
    
    def __init__(
        self,
        n_joints: int,
        pose_dim: int = 7,
        hidden_sizes: List[int] = [512, 512, 256, 256],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_joints = n_joints
        self.pose_dim = pose_dim
        
        input_dim = n_joints + pose_dim
        
        # Build layers
        layers = []
        prev_size = input_dim
        
        for i, size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size
        
        layers.append(nn.Linear(prev_size, n_joints))
        
        self.network = nn.Sequential(*layers)
        
        # Skip connection
        self.skip = nn.Linear(input_dim, n_joints, bias=False)
        nn.init.zeros_(self.skip.weight)
    
    def forward(
        self,
        q_ref: torch.Tensor,
        x_target: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with skip connection."""
        x = torch.cat([q_ref, x_target], dim=-1)
        
        # Main path
        out = self.network(x)
        
        # Skip connection
        skip = self.skip(x)
        
        # Residual output
        return q_ref + out + skip


def create_model(
    n_joints: int,
    model_type: str = 'standard',
    **kwargs,
) -> nn.Module:
    """Factory function for creating IK models.
    
    Parameters
    ----------
    n_joints : int
        Number of robot joints.
    model_type : str
        Type of model: 'standard', 'large'.
    **kwargs
        Additional arguments passed to model constructor.
        
    Returns
    -------
    nn.Module
        IK network model.
    """
    if model_type == 'standard':
        return IKNetwork(n_joints, **kwargs)
    elif model_type == 'large':
        return LargeIKNetwork(n_joints, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
