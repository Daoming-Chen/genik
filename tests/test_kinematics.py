"""Tests for forward kinematics and differentiable FK layer."""

import pytest
import torch
import numpy as np
from pathlib import Path


# Test fixtures
@pytest.fixture
def panda_urdf_path():
    """Path to Panda robot URDF."""
    urdf_path = Path(__file__).parent.parent / "robots" / "panda_arm.urdf"
    if not urdf_path.exists():
        pytest.skip(f"URDF not found: {urdf_path}")
    return str(urdf_path)


@pytest.fixture
def ur10_urdf_path():
    """Path to UR10 robot URDF."""
    urdf_path = Path(__file__).parent.parent / "robots" / "ur10.urdf"
    if not urdf_path.exists():
        pytest.skip(f"URDF not found: {urdf_path}")
    return str(urdf_path)


class TestURDFParser:
    """Tests for URDF parsing."""
    
    def test_load_panda_urdf(self, panda_urdf_path):
        """Test loading Panda URDF."""
        from genik.urdf import URDF
        
        robot = URDF.load(panda_urdf_path)
        assert robot is not None
        assert len(robot.actuated_joints) > 0
    
    def test_load_ur10_urdf(self, ur10_urdf_path):
        """Test loading UR10 URDF."""
        from genik.urdf import URDF
        
        robot = URDF.load(ur10_urdf_path)
        assert robot is not None
        assert len(robot.actuated_joints) > 0
    
    def test_joint_limits(self, panda_urdf_path):
        """Test joint limits extraction."""
        from genik.urdf import URDF
        
        robot = URDF.load(panda_urdf_path)
        
        for joint in robot.actuated_joints:
            assert joint.limit is not None
            assert joint.limit.lower < joint.limit.upper


class TestForwardKinematics:
    """Tests for forward kinematics computation."""
    
    def test_fk_single_config(self, panda_urdf_path):
        """Test FK for single configuration."""
        from genik.urdf import URDF
        
        robot = URDF.load(panda_urdf_path)
        n_joints = len(robot.actuated_joints)
        
        # Zero configuration
        cfg = {joint.name: 0.0 for joint in robot.actuated_joints}
        
        # Get end-effector link
        ee_link = robot.links[-1].name
        
        # Compute FK
        pose = robot.link_fk(cfg=cfg, link=ee_link)
        assert pose.shape == (4, 4)
        
        # Check valid rotation matrix
        R = pose[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)
    
    def test_fk_batch(self, panda_urdf_path):
        """Test batched FK computation."""
        from genik.urdf import URDF
        
        robot = URDF.load(panda_urdf_path)
        n_joints = len(robot.actuated_joints)
        
        # Random configurations
        batch_size = 10
        configs = np.random.uniform(-1, 1, (batch_size, n_joints))
        
        ee_link = robot.links[-1].name
        
        # Compute batch FK
        poses = robot.link_fk_batch(configs, link=ee_link)
        assert poses.shape == (batch_size, 4, 4)
        
        # Verify each pose is valid
        for i in range(batch_size):
            R = poses[i, :3, :3]
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    
    def test_fk_deterministic(self, panda_urdf_path):
        """Test FK is deterministic."""
        from genik.urdf import URDF
        
        robot = URDF.load(panda_urdf_path)
        n_joints = len(robot.actuated_joints)
        
        cfg = {joint.name: np.random.uniform(-1, 1) for joint in robot.actuated_joints}
        ee_link = robot.links[-1].name
        
        pose1 = robot.link_fk(cfg=cfg, link=ee_link)
        pose2 = robot.link_fk(cfg=cfg, link=ee_link)
        
        assert np.allclose(pose1, pose2)


class TestDifferentiableFK:
    """Tests for differentiable FK layer."""
    
    def test_diff_fk_forward(self, panda_urdf_path):
        """Test differentiable FK forward pass."""
        from genik.model.diff_fk import DifferentiableFKLayer
        
        fk_layer = DifferentiableFKLayer(panda_urdf_path)
        n_joints = fk_layer.n_joints
        
        # Random joint angles
        joints = torch.randn(8, n_joints)
        
        # Forward pass
        poses = fk_layer(joints)
        
        # Check output shape (position + quaternion)
        assert poses.shape == (8, 7)
        
        # Check quaternions are normalized
        quat = poses[:, 3:]
        quat_norms = torch.norm(quat, dim=1)
        assert torch.allclose(quat_norms, torch.ones(8), atol=1e-5)
    
    def test_diff_fk_gradient(self, panda_urdf_path):
        """Test differentiable FK gradients."""
        from genik.model.diff_fk import DifferentiableFKLayer
        
        fk_layer = DifferentiableFKLayer(panda_urdf_path)
        n_joints = fk_layer.n_joints
        
        # Input with gradients
        joints = torch.randn(4, n_joints, requires_grad=True)
        
        # Forward pass
        poses = fk_layer(joints)
        
        # Backward pass
        loss = poses.sum()
        loss.backward()
        
        # Check gradients exist
        assert joints.grad is not None
        assert joints.grad.shape == (4, n_joints)
        assert not torch.isnan(joints.grad).any()
    
    def test_diff_fk_consistency(self, panda_urdf_path):
        """Test differentiable FK matches standard FK."""
        from genik.urdf import URDF
        from genik.model.diff_fk import DifferentiableFKLayer
        
        robot = URDF.load(panda_urdf_path)
        fk_layer = DifferentiableFKLayer(panda_urdf_path)
        n_joints = fk_layer.n_joints
        
        # Random configuration
        joint_values = np.random.uniform(-0.5, 0.5, n_joints)
        
        # Standard FK
        cfg = {joint.name: joint_values[i] for i, joint in enumerate(robot.actuated_joints)}
        ee_link = robot.links[-1].name
        pose_std = robot.link_fk(cfg=cfg, link=ee_link)
        pos_std = pose_std[:3, 3]
        
        # Differentiable FK
        joints = torch.tensor(joint_values, dtype=torch.float32).unsqueeze(0)
        pose_diff = fk_layer(joints)
        pos_diff = pose_diff[0, :3].detach().numpy()
        
        # Positions should match (within tolerance due to float precision)
        assert np.allclose(pos_std, pos_diff, atol=1e-4)
    
    def test_diff_fk_joint_limits(self, panda_urdf_path):
        """Test differentiable FK returns joint limits."""
        from genik.model.diff_fk import DifferentiableFKLayer
        
        fk_layer = DifferentiableFKLayer(panda_urdf_path)
        
        lower, upper = fk_layer.get_joint_limits()
        
        assert lower.shape == (fk_layer.n_joints,)
        assert upper.shape == (fk_layer.n_joints,)
        assert (lower < upper).all()


class TestQuaternionOperations:
    """Tests for quaternion utility functions."""
    
    def test_matrix_to_quaternion(self):
        """Test rotation matrix to quaternion conversion."""
        from genik.model.diff_fk import matrix_to_quaternion
        
        # Identity rotation
        R = torch.eye(3).unsqueeze(0)
        q = matrix_to_quaternion(R)
        
        # Identity quaternion [w, x, y, z]
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(q, expected, atol=1e-6)
    
    def test_quaternion_normalization(self):
        """Test quaternions are normalized."""
        from genik.model.diff_fk import matrix_to_quaternion
        
        # Random rotation matrices
        from scipy.spatial.transform import Rotation
        rotations = Rotation.random(10)
        R = torch.tensor(rotations.as_matrix(), dtype=torch.float32)
        
        q = matrix_to_quaternion(R)
        norms = torch.norm(q, dim=1)
        
        assert torch.allclose(norms, torch.ones(10), atol=1e-5)
    
    def test_rodrigues_rotation(self):
        """Test Rodrigues rotation formula."""
        from genik.model.diff_fk import rodrigues_rotation
        
        # Rotation around z-axis by 90 degrees
        axis = torch.tensor([[0.0, 0.0, 1.0]])
        angle = torch.tensor([[np.pi / 2]])
        
        R = rodrigues_rotation(axis, angle)
        
        # Check rotation matrix properties
        assert R.shape == (1, 3, 3)
        assert torch.allclose(R[0] @ R[0].T, torch.eye(3), atol=1e-6)
        assert torch.allclose(torch.det(R[0]), torch.tensor(1.0), atol=1e-6)
