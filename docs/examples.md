# GeNIK Examples

This document provides practical examples for common use cases with GeNIK.

## Table of Contents

- [Example 1: Train IK Solver for Panda Robot](#example-1-train-ik-solver-for-panda-robot)
- [Example 2: Custom Data Generation](#example-2-custom-data-generation)
- [Example 3: Fine-Tuning for Precision](#example-3-fine-tuning-for-precision)
- [Example 4: Real-Time Inference](#example-4-real-time-inference)
- [Example 5: Batch Processing](#example-5-batch-processing)
- [Example 6: Workspace Analysis](#example-6-workspace-analysis)

---

## Example 1: Train IK Solver for Panda Robot

Complete end-to-end example for the Franka Panda robot.

```python
"""
train_panda_ik.py

Complete training pipeline for Panda robot IK.
"""

import torch
import numpy as np
from pathlib import Path

from genik.data.sampler import FKSampler
from genik.data.clustering import PoseClusterer
from genik.data.consistency import ConsistencyMatcher
from genik.data.dataset import GeNIKDataset, create_data_splits
from genik.model.network import ResidualIKNetwork
from genik.model.diff_fk import DifferentiableFKLayer
from genik.model.loss import GeNIKLoss
from genik.train.config import TrainingConfig
from genik.train.trainer import GeNIKTrainer
from genik.eval.metrics import GeNIKEvaluator


def main():
    # Configuration
    URDF_PATH = "robots/panda_arm.urdf"
    OUTPUT_DIR = Path("outputs/panda_example")
    N_SAMPLES = 100000
    EPSILON = 0.05
    EPOCHS = 100
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================
    # Step 1: Generate Training Data
    # =========================================
    print("=" * 50)
    print("Step 1: Generating Training Data")
    print("=" * 50)
    
    sampler = FKSampler(URDF_PATH)
    print(f"Robot has {sampler.n_joints} joints")
    
    # Sample FK data
    joints, poses = sampler.sample(N_SAMPLES, batch_size=1024)
    print(f"Sampled {len(joints):,} configurations")
    
    # Cluster poses
    clusterer = PoseClusterer(epsilon=EPSILON)
    clusters = clusterer.cluster(poses)
    print(f"Created {len(clusters):,} clusters")
    
    # Generate triplets
    matcher = ConsistencyMatcher(strategy="mixed")
    triplets = matcher.generate_triplets(joints, poses, clusters)
    print(f"Generated {len(triplets['q_i']):,} training triplets")
    
    # Save dataset
    data_path = OUTPUT_DIR / "triplets.pt"
    torch.save(triplets, data_path)
    print(f"Saved data to {data_path}")
    
    # =========================================
    # Step 2: Setup Model and Training
    # =========================================
    print("\n" + "=" * 50)
    print("Step 2: Setting Up Training")
    print("=" * 50)
    
    config = TrainingConfig(
        urdf_path=URDF_PATH,
        data_path=str(data_path),
        output_dir=str(OUTPUT_DIR),
        hidden_dim=256,
        n_layers=4,
        dropout=0.1,
        use_residual=True,
        epochs=EPOCHS,
        batch_size=256,
    )
    config.save(OUTPUT_DIR / "config.yaml")
    
    trainer = GeNIKTrainer(config)
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    # =========================================
    # Step 3: Train
    # =========================================
    print("\n" + "=" * 50)
    print("Step 3: Training")
    print("=" * 50)
    
    results = trainer.train()
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    # =========================================
    # Step 4: Evaluate
    # =========================================
    print("\n" + "=" * 50)
    print("Step 4: Evaluation")
    print("=" * 50)
    
    # Create test loader
    dataset = GeNIKDataset(data_path)
    _, _, test_dataset = create_data_splits(dataset, 0.8, 0.1, 0.1)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )
    
    evaluator = GeNIKEvaluator(trainer.model, trainer.fk_layer)
    eval_results = evaluator.evaluate(test_loader)
    
    print(evaluator.generate_report(eval_results))
    
    # =========================================
    # Step 5: Test Inference
    # =========================================
    print("\n" + "=" * 50)
    print("Step 5: Test Inference")
    print("=" * 50)
    
    # Test on random poses
    trainer.model.eval()
    test_poses = torch.randn(10, 7)
    test_poses[:, 3:] = test_poses[:, 3:] / test_poses[:, 3:].norm(dim=1, keepdim=True)
    
    with torch.no_grad():
        pred_joints = trainer.model(test_poses)
        achieved_poses = trainer.fk_layer(pred_joints)
    
    pos_errors = (achieved_poses[:, :3] - test_poses[:, :3]).norm(dim=1) * 1000
    print(f"Position errors (mm): {pos_errors.numpy()}")
    print(f"Mean: {pos_errors.mean():.2f} mm")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
```

---

## Example 2: Custom Data Generation

Generate data with specific workspace constraints.

```python
"""
custom_data_generation.py

Generate training data for a specific workspace region.
"""

import numpy as np
import torch
from genik.data.sampler import FKSampler
from genik.data.clustering import PoseClusterer
from genik.data.consistency import ConsistencyMatcher


def sample_workspace_constrained(sampler, n_samples, workspace_bounds):
    """
    Sample configurations that reach a specific workspace region.
    
    Args:
        sampler: FKSampler instance
        n_samples: Number of valid samples to collect
        workspace_bounds: dict with 'x', 'y', 'z' min/max values
    
    Returns:
        joints, poses within workspace
    """
    joints_list = []
    poses_list = []
    
    # Oversample to account for rejection
    batch_size = 10000
    
    while len(joints_list) < n_samples:
        # Sample batch
        joints, poses = sampler.sample(batch_size)
        
        # Filter by workspace
        x, y, z = poses[:, 0], poses[:, 1], poses[:, 2]
        
        valid = (
            (x >= workspace_bounds['x'][0]) & (x <= workspace_bounds['x'][1]) &
            (y >= workspace_bounds['y'][0]) & (y <= workspace_bounds['y'][1]) &
            (z >= workspace_bounds['z'][0]) & (z <= workspace_bounds['z'][1])
        )
        
        joints_list.append(joints[valid])
        poses_list.append(poses[valid])
        
        print(f"Collected {sum(len(j) for j in joints_list):,}/{n_samples:,} samples")
    
    joints = np.concatenate(joints_list)[:n_samples]
    poses = np.concatenate(poses_list)[:n_samples]
    
    return joints, poses


def main():
    # Define workspace region (e.g., table in front of robot)
    workspace_bounds = {
        'x': (0.3, 0.7),   # 30-70cm in front
        'y': (-0.3, 0.3),  # 30cm left/right
        'z': (0.0, 0.5),   # 0-50cm height
    }
    
    # Sample
    sampler = FKSampler("robots/panda_arm.urdf")
    joints, poses = sample_workspace_constrained(
        sampler, n_samples=50000, workspace_bounds=workspace_bounds
    )
    
    print(f"\nWorkspace statistics:")
    print(f"  X range: [{poses[:, 0].min():.3f}, {poses[:, 0].max():.3f}]")
    print(f"  Y range: [{poses[:, 1].min():.3f}, {poses[:, 1].max():.3f}]")
    print(f"  Z range: [{poses[:, 2].min():.3f}, {poses[:, 2].max():.3f}]")
    
    # Cluster and generate triplets
    clusterer = PoseClusterer(epsilon=0.03)  # Tighter clusters for precision
    clusters = clusterer.cluster(poses)
    
    matcher = ConsistencyMatcher()
    triplets = matcher.generate_triplets(joints, poses, clusters)
    
    # Save
    torch.save(triplets, "data/workspace_constrained_triplets.pt")
    print(f"\nSaved {len(triplets['q_i']):,} triplets")


if __name__ == "__main__":
    main()
```

---

## Example 3: Fine-Tuning for Precision

Fine-tune a model for higher precision in specific regions.

```python
"""
fine_tune_precision.py

Fine-tune trained model for sub-millimeter precision.
"""

import torch
from genik.model.network import IKNetwork
from genik.model.diff_fk import DifferentiableFKLayer
from genik.model.loss import GeNIKLoss
from genik.train.config import TrainingConfig, OptimizerConfig, LossConfig
from genik.train.trainer import GeNIKTrainer


def main():
    # Load pretrained model
    model, checkpoint = IKNetwork.load_checkpoint(
        "outputs/panda/best_model.pt",
        device="cuda"
    )
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Original val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    # Fine-tuning configuration
    # - Lower learning rate
    # - Higher lambda_pose (emphasize pose accuracy)
    # - Smaller epsilon in data (tighter clusters)
    
    config = TrainingConfig(
        urdf_path="robots/panda_arm.urdf",
        data_path="data/precision_triplets.pt",  # Tighter epsilon data
        output_dir="outputs/panda_precision",
        
        # Keep same architecture
        hidden_dim=256,
        n_layers=4,
        use_residual=True,
        
        # Fine-tuning settings
        epochs=50,
        batch_size=128,  # Smaller batches for stability
        
        optimizer=OptimizerConfig(
            lr=1e-5,  # Much lower LR
            weight_decay=1e-5,
            scheduler="cosine",
        ),
        
        loss=LossConfig(
            lambda_pose=2.0,  # Higher pose weight
            adaptive_lambda=False,  # Fixed lambda
        ),
    )
    
    # Create trainer with pretrained model
    trainer = GeNIKTrainer(config)
    trainer.model = model  # Use pretrained weights
    
    # Fine-tune
    results = trainer.train()
    
    print(f"\nFine-tuning complete!")
    print(f"Final val loss: {trainer.best_val_loss:.6f}")


if __name__ == "__main__":
    main()
```

---

## Example 4: Real-Time Inference

Optimized inference for real-time applications.

```python
"""
realtime_inference.py

Real-time IK inference with performance optimization.
"""

import time
import torch
import numpy as np
from genik.model.network import IKNetwork
from genik.model.diff_fk import DifferentiableFKLayer


class RealTimeIKSolver:
    """
    Optimized IK solver for real-time applications.
    
    Features:
    - GPU acceleration
    - JIT compilation
    - Pre-allocated buffers
    """
    
    def __init__(self, model_path, urdf_path, device="cuda"):
        self.device = torch.device(device)
        
        # Load model
        self.model, _ = IKNetwork.load_checkpoint(model_path, device=self.device)
        self.model.eval()
        
        # JIT compile for faster inference
        example_input = torch.randn(1, 7, device=self.device)
        self.model = torch.jit.trace(self.model, example_input)
        
        # FK for verification (optional)
        self.fk = DifferentiableFKLayer(urdf_path)
        self.fk.to(self.device)
        
        # Pre-allocate buffers
        self._pose_buffer = torch.zeros(1, 7, device=self.device)
        
        # Warm up
        self._warmup()
    
    def _warmup(self):
        """Warm up GPU and JIT cache."""
        for _ in range(10):
            self.solve([0.5, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0])
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
    
    def solve(self, position, orientation):
        """
        Solve IK for target pose.
        
        Args:
            position: [x, y, z]
            orientation: [qw, qx, qy, qz]
        
        Returns:
            joints: numpy array of joint angles
        """
        # Fill buffer (avoid allocation)
        self._pose_buffer[0, :3] = torch.tensor(position)
        self._pose_buffer[0, 3:] = torch.tensor(orientation)
        
        # Inference
        with torch.no_grad():
            joints = self.model(self._pose_buffer)
        
        return joints[0].cpu().numpy()
    
    def solve_batch(self, positions, orientations):
        """
        Batch IK solving.
        
        Args:
            positions: (N, 3) array
            orientations: (N, 4) array
        
        Returns:
            joints: (N, n_joints) array
        """
        poses = torch.cat([
            torch.tensor(positions, dtype=torch.float32),
            torch.tensor(orientations, dtype=torch.float32)
        ], dim=1).to(self.device)
        
        with torch.no_grad():
            joints = self.model(poses)
        
        return joints.cpu().numpy()
    
    def benchmark(self, n_iterations=1000):
        """Benchmark inference speed."""
        times = []
        
        for _ in range(n_iterations):
            start = time.perf_counter()
            self.solve([0.5, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0])
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "hz": 1000 / np.mean(times),
        }


def main():
    # Create solver
    solver = RealTimeIKSolver(
        model_path="outputs/panda/best_model.pt",
        urdf_path="robots/panda_arm.urdf",
        device="cuda"
    )
    
    # Benchmark
    results = solver.benchmark(n_iterations=10000)
    
    print("Benchmark Results:")
    print(f"  Mean:  {results['mean_ms']:.3f} ms")
    print(f"  Std:   {results['std_ms']:.3f} ms")
    print(f"  Min:   {results['min_ms']:.3f} ms")
    print(f"  Max:   {results['max_ms']:.3f} ms")
    print(f"  Rate:  {results['hz']:.0f} Hz")
    
    # Example usage
    position = [0.5, 0.0, 0.3]
    orientation = [1.0, 0.0, 0.0, 0.0]
    
    joints = solver.solve(position, orientation)
    print(f"\nSolved IK: {joints}")


if __name__ == "__main__":
    main()
```

---

## Example 5: Batch Processing

Process multiple IK queries efficiently.

```python
"""
batch_processing.py

Efficient batch IK processing for trajectory planning.
"""

import torch
import numpy as np
from genik.model.network import IKNetwork
from genik.model.diff_fk import DifferentiableFKLayer


def generate_trajectory(start_pos, end_pos, n_points):
    """Generate linear trajectory in Cartesian space."""
    positions = np.linspace(start_pos, end_pos, n_points)
    
    # Keep orientation constant (identity quaternion)
    orientations = np.tile([1.0, 0.0, 0.0, 0.0], (n_points, 1))
    
    return positions, orientations


def solve_trajectory(model, fk_layer, positions, orientations, device="cuda"):
    """
    Solve IK for entire trajectory.
    
    Returns joint trajectory and errors.
    """
    # Prepare poses
    poses = torch.cat([
        torch.tensor(positions, dtype=torch.float32),
        torch.tensor(orientations, dtype=torch.float32)
    ], dim=1).to(device)
    
    model.to(device)
    model.eval()
    
    # Batch inference
    with torch.no_grad():
        joints = model(poses)
        achieved_poses = fk_layer(joints)
    
    # Compute errors
    pos_errors = (achieved_poses[:, :3] - poses[:, :3]).norm(dim=1) * 1000  # mm
    
    return joints.cpu().numpy(), pos_errors.cpu().numpy()


def smooth_trajectory(joints, window_size=5):
    """Apply moving average smoothing to joint trajectory."""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(joints, size=window_size, axis=0)


def main():
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = IKNetwork.load_checkpoint("outputs/panda/best_model.pt", device=device)
    fk_layer = DifferentiableFKLayer("robots/panda_arm.urdf")
    
    # Generate trajectory
    start = np.array([0.4, -0.2, 0.3])
    end = np.array([0.6, 0.2, 0.5])
    n_points = 100
    
    positions, orientations = generate_trajectory(start, end, n_points)
    
    # Solve IK
    joints, errors = solve_trajectory(
        model, fk_layer, positions, orientations, device
    )
    
    print(f"Trajectory Statistics:")
    print(f"  Points: {n_points}")
    print(f"  Mean error: {errors.mean():.2f} mm")
    print(f"  Max error:  {errors.max():.2f} mm")
    
    # Smooth trajectory
    joints_smooth = smooth_trajectory(joints)
    
    # Verify smoothed trajectory
    with torch.no_grad():
        joints_t = torch.tensor(joints_smooth, dtype=torch.float32, device=device)
        achieved = fk_layer(joints_t)
        poses_t = torch.tensor(
            np.concatenate([positions, orientations], axis=1),
            dtype=torch.float32, device=device
        )
        smooth_errors = (achieved[:, :3] - poses_t[:, :3]).norm(dim=1) * 1000
    
    print(f"\nAfter smoothing:")
    print(f"  Mean error: {smooth_errors.mean():.2f} mm")
    print(f"  Max error:  {smooth_errors.max():.2f} mm")
    
    # Save trajectory
    np.save("trajectory_joints.npy", joints_smooth)
    print(f"\nSaved trajectory to trajectory_joints.npy")


if __name__ == "__main__":
    main()
```

---

## Example 6: Workspace Analysis

Analyze IK accuracy across the robot workspace.

```python
"""
workspace_analysis.py

Analyze IK performance across the robot workspace.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from genik.model.network import IKNetwork
from genik.model.diff_fk import DifferentiableFKLayer
from genik.data.sampler import FKSampler


def analyze_workspace(model, fk_layer, sampler, n_samples=10000, device="cuda"):
    """
    Analyze IK accuracy across workspace.
    
    Returns position errors mapped to workspace coordinates.
    """
    model.to(device)
    model.eval()
    
    # Sample workspace
    joints_gt, poses_gt = sampler.sample(n_samples)
    
    # Convert to tensors
    poses = torch.tensor(poses_gt, dtype=torch.float32, device=device)
    joints_true = torch.tensor(joints_gt, dtype=torch.float32, device=device)
    
    # Predict
    with torch.no_grad():
        joints_pred = model(poses)
        achieved = fk_layer(joints_pred)
    
    # Compute errors
    pos_errors = (achieved[:, :3] - poses[:, :3]).norm(dim=1) * 1000  # mm
    
    return {
        'positions': poses_gt[:, :3],
        'errors_mm': pos_errors.cpu().numpy(),
    }


def plot_workspace_analysis(results, save_path=None):
    """Create visualization of workspace analysis."""
    positions = results['positions']
    errors = results['errors_mm']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY plane
    ax = axes[0]
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=errors, cmap='RdYlGn_r', s=1, alpha=0.5
    )
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('XY Plane')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Error (mm)')
    
    # XZ plane
    ax = axes[1]
    scatter = ax.scatter(
        positions[:, 0], positions[:, 2],
        c=errors, cmap='RdYlGn_r', s=1, alpha=0.5
    )
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('XZ Plane')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Error (mm)')
    
    # YZ plane
    ax = axes[2]
    scatter = ax.scatter(
        positions[:, 1], positions[:, 2],
        c=errors, cmap='RdYlGn_r', s=1, alpha=0.5
    )
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('YZ Plane')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Error (mm)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    
    plt.show()


def identify_problem_regions(results, error_threshold=10.0):
    """Identify regions with high error."""
    positions = results['positions']
    errors = results['errors_mm']
    
    high_error = errors > error_threshold
    
    if high_error.sum() == 0:
        print(f"No regions with error > {error_threshold} mm")
        return None
    
    problem_positions = positions[high_error]
    
    print(f"\nProblem regions (error > {error_threshold} mm):")
    print(f"  Count: {high_error.sum()} ({100 * high_error.mean():.1f}%)")
    print(f"  X range: [{problem_positions[:, 0].min():.3f}, {problem_positions[:, 0].max():.3f}]")
    print(f"  Y range: [{problem_positions[:, 1].min():.3f}, {problem_positions[:, 1].max():.3f}]")
    print(f"  Z range: [{problem_positions[:, 2].min():.3f}, {problem_positions[:, 2].max():.3f}]")
    
    return problem_positions


def main():
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = IKNetwork.load_checkpoint("outputs/panda/best_model.pt", device=device)
    fk_layer = DifferentiableFKLayer("robots/panda_arm.urdf")
    sampler = FKSampler("robots/panda_arm.urdf")
    
    # Analyze
    print("Analyzing workspace...")
    results = analyze_workspace(model, fk_layer, sampler, n_samples=50000, device=device)
    
    # Statistics
    errors = results['errors_mm']
    print(f"\nWorkspace Statistics:")
    print(f"  Mean error:   {errors.mean():.2f} mm")
    print(f"  Median error: {np.median(errors):.2f} mm")
    print(f"  Std error:    {errors.std():.2f} mm")
    print(f"  Max error:    {errors.max():.2f} mm")
    
    # Success rates at thresholds
    for threshold in [1, 5, 10]:
        rate = 100 * (errors <= threshold).mean()
        print(f"  < {threshold}mm: {rate:.1f}%")
    
    # Identify problem regions
    identify_problem_regions(results, error_threshold=10.0)
    
    # Visualize
    plot_workspace_analysis(results, save_path="workspace_analysis.png")


if __name__ == "__main__":
    main()
```

---

## Running the Examples

1. Make sure you have trained a model first:
   ```bash
   python scripts/train_genik.py --urdf robots/panda_arm.urdf --data data/triplets.pt
   ```

2. Run any example:
   ```bash
   python examples/realtime_inference.py
   ```

3. Adjust paths in examples to match your setup.

## Tips

- Start with Example 1 for a complete workflow
- Use Example 4 for robotics applications
- Use Example 6 to identify accuracy issues
- Combine techniques from multiple examples as needed
