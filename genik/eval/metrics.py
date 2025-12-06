"""Evaluation Metrics for GeNIK.

This module provides comprehensive evaluation metrics for assessing
the accuracy and performance of trained GeNIK models.
"""

from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model.diff_fk import DifferentiableFKLayer
from ..model.network import IKNetwork
from ..data.dataset import GeNIKDataset


def quaternion_distance_np(
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    """Compute geodesic distance between quaternions.
    
    Parameters
    ----------
    q1 : np.ndarray
        First quaternion(s) of shape (N, 4) or (4,).
    q2 : np.ndarray
        Second quaternion(s) of shape (N, 4) or (4,).
        
    Returns
    -------
    np.ndarray
        Angular distance(s) in radians.
    """
    if q1.ndim == 1:
        q1 = q1[np.newaxis, :]
    if q2.ndim == 1:
        q2 = q2[np.newaxis, :]
    
    dot = np.sum(q1 * q2, axis=1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    
    return 2.0 * np.arccos(dot)


def compute_success_rate(
    pos_errors: np.ndarray,
    ori_errors: np.ndarray,
    pos_threshold: float = 0.001,  # 1mm
    ori_threshold: float = 0.0175,  # ~1 degree
) -> float:
    """Compute success rate given error arrays.
    
    Parameters
    ----------
    pos_errors : np.ndarray
        Position errors in meters.
    ori_errors : np.ndarray
        Orientation errors in radians.
    pos_threshold : float
        Position threshold in meters.
    ori_threshold : float
        Orientation threshold in radians.
        
    Returns
    -------
    float
        Success rate as percentage (0-100).
    """
    success = (pos_errors < pos_threshold) & (ori_errors < ori_threshold)
    return 100.0 * np.mean(success)


class GeNIKEvaluator:
    """Evaluator for GeNIK models.
    
    Parameters
    ----------
    model : IKNetwork
        Trained IK network.
    fk_layer : DifferentiableFKLayer
        Differentiable FK layer (or standard FK).
    device : torch.device, optional
        Device for computation.
        
    Examples
    --------
    >>> model, _ = IKNetwork.load_checkpoint("best_model.pt")
    >>> fk_layer = DifferentiableFKLayer("robots/panda_arm.urdf")
    >>> evaluator = GeNIKEvaluator(model, fk_layer)
    >>> results = evaluator.evaluate(test_loader)
    """
    
    def __init__(
        self,
        model: IKNetwork,
        fk_layer: DifferentiableFKLayer,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.fk_layer = fk_layer
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.model.to(device)
        self.model.eval()
        self.fk_layer.to(device)
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate model on a dataset.
        
        Parameters
        ----------
        dataloader : DataLoader
            Test data loader.
        show_progress : bool
            Whether to show progress bar.
            
        Returns
        -------
        dict
            Evaluation results including errors and success rates.
        """
        all_pos_errors = []
        all_ori_errors = []
        all_joint_errors = []
        
        iterator = dataloader
        if show_progress:
            iterator = tqdm(iterator, desc="Evaluating")
        
        for q_ref, x_target, q_star in iterator:
            # Move to device
            q_ref = q_ref.to(self.device)
            x_target = x_target.to(self.device)
            q_star = q_star.to(self.device)
            
            # Predict
            q_pred = self.model(q_ref, x_target)
            
            # Compute FK for predicted joints
            x_pred = self.fk_layer(q_pred)
            
            # Position error
            pos_error = torch.norm(x_pred[:, :3] - x_target[:, :3], dim=1)
            all_pos_errors.append(pos_error.cpu().numpy())
            
            # Orientation error
            dot = torch.sum(x_pred[:, 3:] * x_target[:, 3:], dim=1)
            dot = torch.clamp(torch.abs(dot), -1.0, 1.0)
            ori_error = 2.0 * torch.acos(dot)
            all_ori_errors.append(ori_error.cpu().numpy())
            
            # Joint error
            joint_error = torch.norm(q_pred - q_star, dim=1)
            all_joint_errors.append(joint_error.cpu().numpy())
        
        # Concatenate all results
        pos_errors = np.concatenate(all_pos_errors)
        ori_errors = np.concatenate(all_ori_errors)
        joint_errors = np.concatenate(all_joint_errors)
        
        # Compute statistics
        return self._compute_statistics(pos_errors, ori_errors, joint_errors)
    
    def _compute_statistics(
        self,
        pos_errors: np.ndarray,
        ori_errors: np.ndarray,
        joint_errors: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute evaluation statistics."""
        
        # Position error statistics (convert to mm for readability)
        pos_mm = pos_errors * 1000
        pos_stats = {
            'mean': np.mean(pos_mm),
            'median': np.median(pos_mm),
            'std': np.std(pos_mm),
            'min': np.min(pos_mm),
            'max': np.max(pos_mm),
            'p50': np.percentile(pos_mm, 50),
            'p90': np.percentile(pos_mm, 90),
            'p95': np.percentile(pos_mm, 95),
            'p99': np.percentile(pos_mm, 99),
        }
        
        # Orientation error statistics (convert to degrees)
        ori_deg = np.degrees(ori_errors)
        ori_stats = {
            'mean': np.mean(ori_deg),
            'median': np.median(ori_deg),
            'std': np.std(ori_deg),
            'min': np.min(ori_deg),
            'max': np.max(ori_deg),
            'p50': np.percentile(ori_deg, 50),
            'p90': np.percentile(ori_deg, 90),
            'p95': np.percentile(ori_deg, 95),
            'p99': np.percentile(ori_deg, 99),
        }
        
        # Joint error statistics
        joint_stats = {
            'mean': np.mean(joint_errors),
            'median': np.median(joint_errors),
            'std': np.std(joint_errors),
            'max': np.max(joint_errors),
        }
        
        # Success rates at different thresholds
        success_rates = {}
        thresholds = [
            ('high_precision', 0.0001, 0.00175),  # 0.1mm, 0.1°
            ('standard', 0.001, 0.0175),           # 1mm, 1°
            ('relaxed', 0.005, 0.0873),            # 5mm, 5°
        ]
        
        for name, pos_thresh, ori_thresh in thresholds:
            success_rates[name] = compute_success_rate(
                pos_errors, ori_errors, pos_thresh, ori_thresh
            )
        
        return {
            'n_samples': len(pos_errors),
            'position_error_mm': pos_stats,
            'orientation_error_deg': ori_stats,
            'joint_error_rad': joint_stats,
            'success_rates': success_rates,
            'raw_errors': {
                'position': pos_errors,
                'orientation': ori_errors,
                'joint': joint_errors,
            },
        }
    
    @torch.no_grad()
    def measure_inference_time(
        self,
        batch_sizes: List[int] = [1, 16, 64, 256, 1024],
        num_warmup: int = 10,
        num_iterations: int = 100,
    ) -> Dict[str, Any]:
        """Measure inference time for different batch sizes.
        
        Parameters
        ----------
        batch_sizes : list
            Batch sizes to test.
        num_warmup : int
            Number of warmup iterations.
        num_iterations : int
            Number of timed iterations.
            
        Returns
        -------
        dict
            Timing results for each batch size.
        """
        n_joints = self.model.n_joints
        results = {}
        
        for batch_size in batch_sizes:
            # Create random inputs
            q_ref = torch.randn(batch_size, n_joints, device=self.device)
            x_target = torch.randn(batch_size, 7, device=self.device)
            
            # Warmup
            for _ in range(num_warmup):
                _ = self.model(q_ref, x_target)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Timed runs
            times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = self.model(q_ref, x_target)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            
            times = np.array(times) * 1000  # Convert to ms
            
            results[batch_size] = {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'median_ms': np.median(times),
                'per_sample_us': np.mean(times) / batch_size * 1000,
                'throughput_qps': batch_size / (np.mean(times) / 1000),
            }
        
        return results
    
    @torch.no_grad()
    def check_joint_limits(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, Any]:
        """Check for joint limit violations in predictions.
        
        Parameters
        ----------
        dataloader : DataLoader
            Test data loader.
            
        Returns
        -------
        dict
            Statistics about joint limit violations.
        """
        joint_lower, joint_upper = self.fk_layer.get_joint_limits()
        joint_lower = joint_lower.cpu().numpy()
        joint_upper = joint_upper.cpu().numpy()
        
        violations = []
        
        for q_ref, x_target, _ in dataloader:
            q_ref = q_ref.to(self.device)
            x_target = x_target.to(self.device)
            
            q_pred = self.model(q_ref, x_target).cpu().numpy()
            
            # Check violations
            lower_viol = np.maximum(0, joint_lower - q_pred)
            upper_viol = np.maximum(0, q_pred - joint_upper)
            
            max_viol = np.maximum(lower_viol, upper_viol).max(axis=1)
            violations.append(max_viol)
        
        violations = np.concatenate(violations)
        
        return {
            'violation_rate': 100 * np.mean(violations > 0),
            'max_violation_rad': np.max(violations),
            'mean_violation_rad': np.mean(violations[violations > 0]) if np.any(violations > 0) else 0,
        }
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate a text report from evaluation results.
        
        Parameters
        ----------
        results : dict
            Evaluation results from evaluate().
        output_path : str or Path, optional
            Path to save report.
            
        Returns
        -------
        str
            Report text.
        """
        lines = [
            "=" * 60,
            "GeNIK Evaluation Report",
            "=" * 60,
            "",
            f"Number of test samples: {results['n_samples']:,}",
            "",
            "Position Error (mm):",
            f"  Mean:   {results['position_error_mm']['mean']:.3f}",
            f"  Median: {results['position_error_mm']['median']:.3f}",
            f"  Std:    {results['position_error_mm']['std']:.3f}",
            f"  P95:    {results['position_error_mm']['p95']:.3f}",
            f"  P99:    {results['position_error_mm']['p99']:.3f}",
            f"  Max:    {results['position_error_mm']['max']:.3f}",
            "",
            "Orientation Error (degrees):",
            f"  Mean:   {results['orientation_error_deg']['mean']:.3f}",
            f"  Median: {results['orientation_error_deg']['median']:.3f}",
            f"  Std:    {results['orientation_error_deg']['std']:.3f}",
            f"  P95:    {results['orientation_error_deg']['p95']:.3f}",
            f"  P99:    {results['orientation_error_deg']['p99']:.3f}",
            f"  Max:    {results['orientation_error_deg']['max']:.3f}",
            "",
            "Success Rates:",
            f"  High Precision (0.1mm, 0.1°): {results['success_rates']['high_precision']:.1f}%",
            f"  Standard (1mm, 1°):           {results['success_rates']['standard']:.1f}%",
            f"  Relaxed (5mm, 5°):            {results['success_rates']['relaxed']:.1f}%",
            "",
            "=" * 60,
        ]
        
        report = "\n".join(lines)
        
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report
