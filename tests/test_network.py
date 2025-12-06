"""Tests for neural network architecture."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def panda_urdf_path():
    """Path to Panda robot URDF."""
    urdf_path = Path(__file__).parent.parent / "robots" / "panda_arm.urdf"
    if not urdf_path.exists():
        pytest.skip(f"URDF not found: {urdf_path}")
    return str(urdf_path)


class TestIKNetwork:
    """Tests for IK network architecture."""
    
    def test_network_init(self):
        """Test network initialization."""
        from genik.model.network import IKNetwork
        
        network = IKNetwork(input_dim=7, output_dim=7, hidden_dim=256, n_layers=4)
        
        assert network.input_dim == 7
        assert network.output_dim == 7
    
    def test_network_forward(self):
        """Test network forward pass."""
        from genik.model.network import IKNetwork
        
        network = IKNetwork(input_dim=7, output_dim=7, hidden_dim=256, n_layers=4)
        
        # Random pose input
        x = torch.randn(16, 7)
        
        # Forward pass
        y = network(x)
        
        assert y.shape == (16, 7)
        assert not torch.isnan(y).any()
    
    def test_network_gradient(self):
        """Test network gradient flow."""
        from genik.model.network import IKNetwork
        
        network = IKNetwork(input_dim=7, output_dim=7, hidden_dim=256, n_layers=4)
        
        x = torch.randn(16, 7, requires_grad=True)
        y = network(x)
        
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        
        # Check all parameters have gradients
        for param in network.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_network_different_sizes(self):
        """Test network with different configurations."""
        from genik.model.network import IKNetwork
        
        configs = [
            (7, 6, 128, 2),   # 7D pose, 6 joints
            (7, 7, 256, 4),   # 7D pose, 7 joints
            (7, 9, 512, 6),   # 7D pose, 9 joints
        ]
        
        for input_dim, output_dim, hidden_dim, n_layers in configs:
            network = IKNetwork(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
            )
            
            x = torch.randn(8, input_dim)
            y = network(x)
            
            assert y.shape == (8, output_dim)


class TestResidualIKNetwork:
    """Tests for residual IK network."""
    
    def test_residual_network_init(self):
        """Test residual network initialization."""
        from genik.model.network import ResidualIKNetwork
        
        network = ResidualIKNetwork(
            input_dim=7, output_dim=7, hidden_dim=256, n_blocks=4
        )
        
        assert network is not None
    
    def test_residual_network_forward(self):
        """Test residual network forward pass."""
        from genik.model.network import ResidualIKNetwork
        
        network = ResidualIKNetwork(
            input_dim=7, output_dim=7, hidden_dim=256, n_blocks=4
        )
        
        x = torch.randn(16, 7)
        y = network(x)
        
        assert y.shape == (16, 7)
        assert not torch.isnan(y).any()
    
    def test_residual_network_with_dropout(self):
        """Test residual network with dropout."""
        from genik.model.network import ResidualIKNetwork
        
        network = ResidualIKNetwork(
            input_dim=7, output_dim=7, hidden_dim=256, n_blocks=4, dropout=0.1
        )
        
        # Training mode (dropout active)
        network.train()
        x = torch.randn(16, 7)
        y_train = network(x)
        
        # Eval mode (dropout inactive)
        network.eval()
        y_eval = network(x)
        
        # Outputs should differ in training mode due to dropout
        assert y_train.shape == y_eval.shape


class TestResidualBlock:
    """Tests for residual blocks."""
    
    def test_residual_block_forward(self):
        """Test residual block forward pass."""
        from genik.model.network import ResidualBlock
        
        block = ResidualBlock(hidden_dim=256, dropout=0.1)
        
        x = torch.randn(16, 256)
        y = block(x)
        
        assert y.shape == (16, 256)
    
    def test_residual_connection(self):
        """Test residual connection preserves information."""
        from genik.model.network import ResidualBlock
        
        block = ResidualBlock(hidden_dim=256, dropout=0.0)
        block.eval()
        
        # Set weights to near-zero to test residual
        with torch.no_grad():
            for name, param in block.named_parameters():
                if 'weight' in name:
                    param.fill_(0.001)
        
        x = torch.randn(16, 256)
        y = block(x)
        
        # With near-zero weights, output should be close to input
        # due to residual connection
        diff = (y - x).abs().mean()
        assert diff < 1.0  # Should be relatively small


class TestGeNIKLoss:
    """Tests for loss functions."""
    
    def test_loss_init(self, panda_urdf_path):
        """Test loss function initialization."""
        from genik.model.loss import GeNIKLoss
        from genik.model.diff_fk import DifferentiableFKLayer
        
        fk_layer = DifferentiableFKLayer(panda_urdf_path)
        loss_fn = GeNIKLoss(fk_layer, lambda_pose=1.0)
        
        assert loss_fn is not None
    
    def test_joint_loss(self, panda_urdf_path):
        """Test joint space loss computation."""
        from genik.model.loss import GeNIKLoss
        from genik.model.diff_fk import DifferentiableFKLayer
        
        fk_layer = DifferentiableFKLayer(panda_urdf_path)
        loss_fn = GeNIKLoss(fk_layer, lambda_pose=1.0)
        
        n_joints = fk_layer.n_joints
        q_pred = torch.randn(16, n_joints)
        q_target = torch.randn(16, n_joints)
        
        loss = loss_fn.joint_loss(q_pred, q_target)
        
        assert loss.shape == ()
        assert loss >= 0
    
    def test_pose_loss(self, panda_urdf_path):
        """Test pose space loss computation."""
        from genik.model.loss import GeNIKLoss
        from genik.model.diff_fk import DifferentiableFKLayer
        
        fk_layer = DifferentiableFKLayer(panda_urdf_path)
        loss_fn = GeNIKLoss(fk_layer, lambda_pose=1.0)
        
        n_joints = fk_layer.n_joints
        q_pred = torch.randn(16, n_joints)
        p_target = torch.randn(16, 7)
        
        loss = loss_fn.pose_loss(q_pred, p_target)
        
        assert loss.shape == ()
        assert loss >= 0
    
    def test_total_loss(self, panda_urdf_path):
        """Test total loss computation."""
        from genik.model.loss import GeNIKLoss
        from genik.model.diff_fk import DifferentiableFKLayer
        
        fk_layer = DifferentiableFKLayer(panda_urdf_path)
        loss_fn = GeNIKLoss(fk_layer, lambda_pose=1.0)
        
        n_joints = fk_layer.n_joints
        q_pred = torch.randn(16, n_joints)
        q_target = torch.randn(16, n_joints)
        p_target = torch.randn(16, 7)
        
        loss, components = loss_fn(q_pred, q_target, p_target)
        
        assert loss.shape == ()
        assert 'joint' in components
        assert 'pose' in components
    
    def test_adaptive_lambda(self, panda_urdf_path):
        """Test adaptive lambda scheduling."""
        from genik.model.loss import GeNIKLoss
        from genik.model.diff_fk import DifferentiableFKLayer
        
        fk_layer = DifferentiableFKLayer(panda_urdf_path)
        loss_fn = GeNIKLoss(
            fk_layer,
            lambda_pose=1.0,
            adaptive_lambda=True,
            lambda_warmup_epochs=10,
        )
        
        # Lambda should increase during warmup
        loss_fn.update_lambda(epoch=0)
        lambda_0 = loss_fn.lambda_pose
        
        loss_fn.update_lambda(epoch=5)
        lambda_5 = loss_fn.lambda_pose
        
        loss_fn.update_lambda(epoch=10)
        lambda_10 = loss_fn.lambda_pose
        
        assert lambda_0 <= lambda_5 <= lambda_10
    
    def test_loss_gradient(self, panda_urdf_path):
        """Test loss gradient flow."""
        from genik.model.loss import GeNIKLoss
        from genik.model.diff_fk import DifferentiableFKLayer
        
        fk_layer = DifferentiableFKLayer(panda_urdf_path)
        loss_fn = GeNIKLoss(fk_layer, lambda_pose=1.0)
        
        n_joints = fk_layer.n_joints
        q_pred = torch.randn(16, n_joints, requires_grad=True)
        q_target = torch.randn(16, n_joints)
        p_target = torch.randn(16, 7)
        
        loss, _ = loss_fn(q_pred, q_target, p_target)
        loss.backward()
        
        assert q_pred.grad is not None
        assert not torch.isnan(q_pred.grad).any()


class TestModelSaveLoad:
    """Tests for model saving and loading."""
    
    def test_save_load_network(self, tmp_path):
        """Test saving and loading network."""
        from genik.model.network import IKNetwork
        
        # Create and save network
        network = IKNetwork(input_dim=7, output_dim=7, hidden_dim=256, n_layers=4)
        
        checkpoint_path = tmp_path / "model.pt"
        network.save_checkpoint(checkpoint_path)
        
        # Load network
        loaded_network, checkpoint = IKNetwork.load_checkpoint(checkpoint_path)
        
        # Test same outputs
        x = torch.randn(8, 7)
        
        network.eval()
        loaded_network.eval()
        
        y_orig = network(x)
        y_loaded = loaded_network(x)
        
        assert torch.allclose(y_orig, y_loaded)
    
    def test_save_load_with_config(self, tmp_path):
        """Test saving and loading with configuration."""
        from genik.model.network import IKNetwork
        
        config = {
            'urdf_path': '/path/to/robot.urdf',
            'epochs': 100,
            'learning_rate': 1e-3,
        }
        
        network = IKNetwork(input_dim=7, output_dim=7, hidden_dim=256, n_layers=4)
        
        checkpoint_path = tmp_path / "model.pt"
        network.save_checkpoint(checkpoint_path, config=config)
        
        loaded_network, checkpoint = IKNetwork.load_checkpoint(checkpoint_path)
        
        assert checkpoint['config'] == config
