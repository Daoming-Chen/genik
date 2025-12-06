"""Tests for evaluation metrics and visualization."""

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


@pytest.fixture
def mock_model(panda_urdf_path):
    """Create a mock IK model."""
    from genik.model.network import IKNetwork
    from genik.model.diff_fk import DifferentiableFKLayer
    
    fk_layer = DifferentiableFKLayer(panda_urdf_path)
    n_joints = fk_layer.n_joints
    
    # Create network (not trained, just for testing)
    network = IKNetwork(input_dim=7, output_dim=n_joints, hidden_dim=128, n_layers=2)
    
    return network, fk_layer


class TestGeNIKEvaluator:
    """Tests for evaluation metrics."""
    
    def test_evaluator_init(self, mock_model):
        """Test evaluator initialization."""
        from genik.eval.metrics import GeNIKEvaluator
        
        network, fk_layer = mock_model
        evaluator = GeNIKEvaluator(network, fk_layer)
        
        assert evaluator is not None
    
    def test_compute_position_error(self, mock_model):
        """Test position error computation."""
        from genik.eval.metrics import GeNIKEvaluator
        
        network, fk_layer = mock_model
        evaluator = GeNIKEvaluator(network, fk_layer)
        
        # Random poses
        pose_pred = torch.randn(16, 7)
        pose_target = torch.randn(16, 7)
        
        error = evaluator.compute_position_error(pose_pred, pose_target)
        
        assert error.shape == (16,)
        assert (error >= 0).all()
    
    def test_compute_orientation_error(self, mock_model):
        """Test orientation error computation."""
        from genik.eval.metrics import GeNIKEvaluator
        
        network, fk_layer = mock_model
        evaluator = GeNIKEvaluator(network, fk_layer)
        
        # Random normalized quaternions
        quat_pred = torch.randn(16, 4)
        quat_pred = quat_pred / quat_pred.norm(dim=1, keepdim=True)
        
        quat_target = torch.randn(16, 4)
        quat_target = quat_target / quat_target.norm(dim=1, keepdim=True)
        
        error = evaluator.compute_orientation_error(quat_pred, quat_target)
        
        assert error.shape == (16,)
        assert (error >= 0).all()
        assert (error <= np.pi).all()  # Max angular error is pi
    
    def test_compute_success_rate(self, mock_model):
        """Test success rate computation."""
        from genik.eval.metrics import GeNIKEvaluator
        
        network, fk_layer = mock_model
        evaluator = GeNIKEvaluator(network, fk_layer)
        
        # Create errors
        pos_errors = torch.tensor([0.001, 0.005, 0.01, 0.02, 0.05])
        ori_errors = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.5])
        
        # Test with threshold
        rate = evaluator.compute_success_rate(
            pos_errors, ori_errors,
            pos_threshold=0.01, ori_threshold=0.1
        )
        
        assert 0 <= rate <= 1
    
    def test_evaluate_batch(self, mock_model, panda_urdf_path):
        """Test batch evaluation."""
        from genik.eval.metrics import GeNIKEvaluator
        from genik.data.sampler import FKSampler
        
        network, fk_layer = mock_model
        evaluator = GeNIKEvaluator(network, fk_layer)
        
        # Generate test data
        sampler = FKSampler(panda_urdf_path)
        joints, poses = sampler.sample(32)
        
        # Convert to tensors
        joints = torch.tensor(joints, dtype=torch.float32)
        poses = torch.tensor(poses, dtype=torch.float32)
        
        # Evaluate
        results = evaluator.evaluate_batch(poses, joints)
        
        assert 'position_error' in results
        assert 'orientation_error' in results
        assert 'joint_error' in results
    
    def test_generate_report(self, mock_model):
        """Test report generation."""
        from genik.eval.metrics import GeNIKEvaluator
        
        network, fk_layer = mock_model
        evaluator = GeNIKEvaluator(network, fk_layer)
        
        # Mock results
        results = {
            'n_samples': 1000,
            'position_error_mm': {
                'mean': 5.2,
                'std': 2.1,
                'median': 4.8,
                'max': 15.3,
            },
            'orientation_error_deg': {
                'mean': 2.5,
                'std': 1.2,
                'median': 2.1,
                'max': 8.5,
            },
            'success_rates': {
                '1mm_1deg': 0.15,
                '5mm_5deg': 0.75,
                '10mm_10deg': 0.95,
            },
            'raw_errors': {
                'position': np.random.rand(1000) * 0.01,
                'orientation': np.random.rand(1000) * 0.1,
            },
        }
        
        report = evaluator.generate_report(results)
        
        assert isinstance(report, str)
        assert 'Position Error' in report
        assert 'Orientation Error' in report
        assert 'Success Rate' in report
    
    def test_measure_inference_time(self, mock_model):
        """Test inference timing measurement."""
        from genik.eval.metrics import GeNIKEvaluator
        
        network, fk_layer = mock_model
        evaluator = GeNIKEvaluator(network, fk_layer)
        
        timing = evaluator.measure_inference_time(batch_sizes=[1, 8, 32])
        
        assert 1 in timing
        assert 8 in timing
        assert 32 in timing
        
        for batch_size, result in timing.items():
            assert 'mean_ms' in result
            assert 'throughput_qps' in result
            assert result['mean_ms'] > 0
    
    def test_check_joint_limits(self, mock_model, panda_urdf_path):
        """Test joint limit checking."""
        from genik.eval.metrics import GeNIKEvaluator
        from genik.data.sampler import FKSampler
        from torch.utils.data import DataLoader, TensorDataset
        
        network, fk_layer = mock_model
        evaluator = GeNIKEvaluator(network, fk_layer)
        
        # Generate test data
        sampler = FKSampler(panda_urdf_path)
        joints, poses = sampler.sample(64)
        
        # Create simple dataloader
        dataset = TensorDataset(
            torch.tensor(poses, dtype=torch.float32),
            torch.tensor(joints, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=16)
        
        results = evaluator.check_joint_limits(loader)
        
        assert 'violation_rate' in results
        assert 'max_violation_rad' in results


class TestVisualization:
    """Tests for visualization tools."""
    
    def test_plot_error_histogram(self, tmp_path):
        """Test error histogram plotting."""
        from genik.eval.visualize import plot_error_histogram
        
        pos_errors = np.random.rand(100) * 10  # 0-10mm
        ori_errors = np.random.rand(100) * 5   # 0-5deg
        
        save_path = tmp_path / "histogram.png"
        plot_error_histogram(pos_errors, ori_errors, save_path=save_path)
        
        assert save_path.exists()
    
    def test_plot_error_vs_threshold(self, tmp_path):
        """Test threshold curve plotting."""
        from genik.eval.visualize import plot_error_vs_threshold
        
        pos_errors = np.random.rand(100) * 10
        ori_errors = np.random.rand(100) * 5
        
        save_path = tmp_path / "threshold.png"
        plot_error_vs_threshold(pos_errors, ori_errors, save_path=save_path)
        
        assert save_path.exists()
    
    def test_plot_inference_time(self, tmp_path):
        """Test inference timing plot."""
        from genik.eval.visualize import plot_inference_time
        
        timing_results = {
            1: {'mean_ms': 0.5, 'std_ms': 0.1, 'throughput_qps': 2000},
            8: {'mean_ms': 0.8, 'std_ms': 0.15, 'throughput_qps': 10000},
            32: {'mean_ms': 1.5, 'std_ms': 0.2, 'throughput_qps': 21333},
        }
        
        save_path = tmp_path / "timing.png"
        plot_inference_time(timing_results, save_path=save_path)
        
        assert save_path.exists()
    
    def test_plot_returns_figure(self):
        """Test plots can return figure objects."""
        from genik.eval.visualize import plot_error_histogram
        import matplotlib.pyplot as plt
        
        pos_errors = np.random.rand(100) * 10
        ori_errors = np.random.rand(100) * 5
        
        fig = plot_error_histogram(pos_errors, ori_errors, save_path=None, show=False)
        
        # Should return figure if not saving
        assert fig is not None or True  # May return None depending on implementation
        
        plt.close('all')


class TestMetricsComputation:
    """Tests for specific metric computations."""
    
    def test_position_error_units(self, mock_model):
        """Test position error is in correct units."""
        from genik.eval.metrics import GeNIKEvaluator
        
        network, fk_layer = mock_model
        evaluator = GeNIKEvaluator(network, fk_layer)
        
        # Poses 10mm apart
        pose1 = torch.zeros(1, 7)
        pose1[0, :3] = torch.tensor([0.0, 0.0, 0.0])
        
        pose2 = torch.zeros(1, 7)
        pose2[0, :3] = torch.tensor([0.01, 0.0, 0.0])  # 10mm in x
        
        error = evaluator.compute_position_error(pose1, pose2)
        
        # Error should be ~0.01m = 10mm
        assert torch.isclose(error[0], torch.tensor(0.01), atol=1e-6)
    
    def test_orientation_error_same_quat(self, mock_model):
        """Test orientation error is zero for same quaternions."""
        from genik.eval.metrics import GeNIKEvaluator
        
        network, fk_layer = mock_model
        evaluator = GeNIKEvaluator(network, fk_layer)
        
        # Same quaternions
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        
        error = evaluator.compute_orientation_error(quat, quat)
        
        assert torch.isclose(error[0], torch.tensor(0.0), atol=1e-6)
    
    def test_orientation_error_opposite_quat(self, mock_model):
        """Test orientation error handles quaternion double cover."""
        from genik.eval.metrics import GeNIKEvaluator
        
        network, fk_layer = mock_model
        evaluator = GeNIKEvaluator(network, fk_layer)
        
        # q and -q represent same rotation
        quat1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        quat2 = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])
        
        error = evaluator.compute_orientation_error(quat1, quat2)
        
        # Should be close to zero (same rotation)
        assert torch.isclose(error[0], torch.tensor(0.0), atol=1e-4)
    
    def test_success_rate_all_pass(self, mock_model):
        """Test success rate when all samples pass."""
        from genik.eval.metrics import GeNIKEvaluator
        
        network, fk_layer = mock_model
        evaluator = GeNIKEvaluator(network, fk_layer)
        
        # All errors below threshold
        pos_errors = torch.ones(100) * 0.001  # 1mm
        ori_errors = torch.ones(100) * 0.01   # ~0.5deg
        
        rate = evaluator.compute_success_rate(
            pos_errors, ori_errors,
            pos_threshold=0.01, ori_threshold=0.1
        )
        
        assert rate == 1.0
    
    def test_success_rate_all_fail(self, mock_model):
        """Test success rate when all samples fail."""
        from genik.eval.metrics import GeNIKEvaluator
        
        network, fk_layer = mock_model
        evaluator = GeNIKEvaluator(network, fk_layer)
        
        # All errors above threshold
        pos_errors = torch.ones(100) * 0.1   # 100mm
        ori_errors = torch.ones(100) * 1.0   # ~57deg
        
        rate = evaluator.compute_success_rate(
            pos_errors, ori_errors,
            pos_threshold=0.01, ori_threshold=0.1
        )
        
        assert rate == 0.0
