# GeNIK User Guide

This guide provides step-by-step instructions for using GeNIK (Generative Network for Inverse Kinematics) to train neural network-based IK solvers for your robot.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Data Generation](#data-generation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Using Trained Models](#using-trained-models)
8. [Advanced Topics](#advanced-topics)
9. [Troubleshooting](#troubleshooting)

---

## Overview

GeNIK implements a consistency-based training approach for learning inverse kinematics without requiring ground-truth IK solutions. The key insight is that while IK is a one-to-many problem (multiple joint configurations can reach the same pose), we can train a network by ensuring it produces consistent outputs for similar poses.

### Key Components

1. **Data Generation**: Sample FK data and cluster poses
2. **Consistency Matching**: Generate training triplets
3. **Neural Network**: MLP with differentiable FK layer
4. **Mixed Loss**: Joint-space + pose-space supervision

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy
- (Optional) CUDA for GPU acceleration

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Installation

```python
from genik.urdf import URDF
from genik.data.sampler import FKSampler

# Load a robot
robot = URDF.load("robots/panda_arm.urdf")
print(f"Robot has {len(robot.actuated_joints)} joints")
```

---

## Quick Start

Here's a minimal example to train an IK solver:

```bash
# 1. Generate dataset
python scripts/generate_dataset.py --urdf robots/panda_arm.urdf --output data/panda --num-triplets 5000000 --samples-per-pose 5

# 2. Train model
python scripts/train_genik.py --urdf robots/panda_arm.urdf --data data/panda/train_triplets.pt --output outputs/panda --epochs 100

# 3. Evaluate
python scripts/evaluate_genik.py --checkpoint outputs/panda/best_model.pt --data data/panda/train_triplets.pt --output results/panda
```

---

## Data Generation

### Step 1: FK Sampling

Sample random joint configurations and compute their forward kinematics:

```python
from genik.data.sampler import FKSampler

# Initialize sampler with URDF
sampler = FKSampler("robots/panda_arm.urdf")

# Sample 100,000 configurations
joints, poses = sampler.sample(100000, batch_size=1024)

print(f"Joints shape: {joints.shape}")  # (100000, 7)
print(f"Poses shape: {poses.shape}")    # (100000, 7)
```

The poses are 7D vectors: `[x, y, z, qw, qx, qy, qz]` where `(qw, qx, qy, qz)` is a unit quaternion.

### Step 2: Pose Clustering

Group similar poses to identify multi-solution regions:

```python
from genik.data.clustering import PoseClusterer

# Create clusterer with epsilon radius
clusterer = PoseClusterer(epsilon=0.05)

# Cluster poses
clusters = clusterer.cluster(poses)

print(f"Found {len(clusters)} clusters")
print(f"Average cluster size: {np.mean([len(c) for c in clusters]):.1f}")
```

#### Choosing Epsilon

- **Small epsilon (0.01-0.03)**: Tight clusters, more clusters, slower training
- **Medium epsilon (0.03-0.10)**: Balanced approach
- **Large epsilon (0.10-0.20)**: Loose clusters, faster but less precise

### Step 3: Consistency Matching

Generate training triplets:

```python
from genik.data.consistency import ConsistencyMatcher

matcher = ConsistencyMatcher(strategy="mixed")

triplets = matcher.generate_triplets(joints, poses, clusters)

print(f"Generated {len(triplets['q_i'])} triplets")
```

#### Strategies

- `"perturbation"`: Add noise to queries (better for smooth regions)
- `"random"`: Random samples from clusters (better diversity)
- `"mixed"`: 50/50 combination (recommended)

### Step 4: Save Dataset

```python
import torch

# Save as PyTorch file
torch.save(triplets, "data/panda/train_triplets.pt")
```

### Using the CLI Script

```bash
python scripts/generate_dataset.py \
    --urdf robots/panda_arm.urdf \
    --output data/panda \
    --samples 100000 \
    --epsilon 0.05 \
    --strategy mixed
```

---

## Training

### Configuration

Create a training configuration:

```python
from genik.train.config import TrainingConfig

config = TrainingConfig(
    # Data paths
    urdf_path="robots/panda_arm.urdf",
    data_path="data/panda/train_triplets.pt",
    output_dir="outputs/panda",
    
    # Model architecture
    hidden_dim=256,
    n_layers=4,
    dropout=0.1,
    use_residual=True,
    
    # Training
    epochs=100,
    batch_size=256,
    
    # Learning rate schedule
    optimizer=OptimizerConfig(
        lr=1e-3,
        scheduler="cosine",
    ),
    
    # Loss function
    loss=LossConfig(
        lambda_pose=1.0,
        adaptive_lambda=True,
        lambda_warmup_epochs=10,
    ),
)

# Save for reproducibility
config.save("config.yaml")
```

### Running Training

```python
from genik.train.trainer import GeNIKTrainer

trainer = GeNIKTrainer(config)
results = trainer.train()

print(f"Best validation loss: {trainer.best_val_loss:.4f}")
```

### Monitoring with TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/panda/logs

# Open http://localhost:6006 in browser
```

### Using the CLI Script

```bash
python scripts/train_genik.py \
    --urdf robots/panda_arm.urdf \
    --data data/panda/train_triplets.pt \
    --output outputs/panda \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-3 \
    --hidden-dim 256 \
    --n-layers 4
```

### Resuming Training

```bash
python scripts/train_genik.py \
    --resume outputs/panda/checkpoints/epoch_50.pt \
    --epochs 100
```

---

## Evaluation

### Loading a Trained Model

```python
from genik.model.network import IKNetwork

model, checkpoint = IKNetwork.load_checkpoint(
    "outputs/panda/best_model.pt",
    device="cuda"
)
model.eval()
```

### Running Evaluation

```python
from genik.eval.metrics import GeNIKEvaluator
from genik.model.diff_fk import DifferentiableFKLayer

fk_layer = DifferentiableFKLayer("robots/panda_arm.urdf")
evaluator = GeNIKEvaluator(model, fk_layer, device="cuda")

# Evaluate on test data
results = evaluator.evaluate(test_loader, show_progress=True)

# Print report
print(evaluator.generate_report(results))
```

### Understanding Metrics

- **Position Error**: Distance between predicted and target end-effector position (mm)
- **Orientation Error**: Angular difference between quaternions (degrees)
- **Success Rate**: Percentage of samples within threshold
  - `1mm/1°`: Precision applications
  - `5mm/5°`: General manipulation
  - `10mm/10°`: Coarse positioning

### Visualization

```python
from genik.eval.visualize import plot_error_histogram, plot_error_vs_threshold

# Error distributions
plot_error_histogram(
    results['raw_errors']['position'] * 1000,  # Convert to mm
    np.degrees(results['raw_errors']['orientation']),
    save_path="error_histogram.png"
)

# Success rate curves
plot_error_vs_threshold(
    results['raw_errors']['position'] * 1000,
    np.degrees(results['raw_errors']['orientation']),
    save_path="threshold_curves.png"
)
```

### Using the CLI Script

```bash
python scripts/evaluate_genik.py \
    --checkpoint outputs/panda/best_model.pt \
    --data data/panda/train_triplets.pt \
    --output results/panda \
    --measure-timing
```

---

## Using Trained Models

### Basic Inference

```python
import torch
from genik.model.network import IKNetwork

# Load model
model, _ = IKNetwork.load_checkpoint("outputs/panda/best_model.pt")
model.eval()

# Prepare target pose: [x, y, z, qw, qx, qy, qz]
target_pose = torch.tensor([[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]])

# Predict joint angles
with torch.no_grad():
    joints = model(target_pose)

print(f"Predicted joints: {joints.numpy()}")
```

### Batch Inference

```python
# Multiple poses at once
target_poses = torch.randn(100, 7)

# Normalize quaternions
target_poses[:, 3:] = target_poses[:, 3:] / target_poses[:, 3:].norm(dim=1, keepdim=True)

with torch.no_grad():
    joints_batch = model(target_poses)
```

### Verifying Solutions

```python
from genik.model.diff_fk import DifferentiableFKLayer

fk_layer = DifferentiableFKLayer("robots/panda_arm.urdf")

# Check if solution reaches target
with torch.no_grad():
    achieved_pose = fk_layer(joints)

# Compute error
pos_error = (achieved_pose[:, :3] - target_pose[:, :3]).norm(dim=1)
print(f"Position error: {pos_error.item() * 1000:.2f} mm")
```

### Integration Example

```python
class IKSolver:
    """Simple IK solver wrapper."""
    
    def __init__(self, model_path, urdf_path, device="cpu"):
        self.model, _ = IKNetwork.load_checkpoint(model_path, device=device)
        self.model.eval()
        self.fk = DifferentiableFKLayer(urdf_path)
        self.device = device
    
    def solve(self, position, orientation):
        """
        Solve IK for target pose.
        
        Args:
            position: [x, y, z] target position
            orientation: [qw, qx, qy, qz] target orientation
        
        Returns:
            joints: Joint angles
            error: Position error in meters
        """
        pose = torch.tensor(
            [*position, *orientation],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            joints = self.model(pose)
            achieved = self.fk(joints)
        
        error = (achieved[0, :3] - pose[0, :3]).norm().item()
        return joints.squeeze().cpu().numpy(), error

# Usage
solver = IKSolver("outputs/panda/best_model.pt", "robots/panda_arm.urdf")
joints, error = solver.solve([0.5, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0])
```

---

## Advanced Topics

### GPU Training

```python
config = TrainingConfig(
    # ...
    device="cuda",  # Use GPU
)
```

For multi-GPU:
```python
import torch.nn as nn

model = IKNetwork(...)
model = nn.DataParallel(model)
```

### Custom Network Architectures

```python
from genik.model.network import IKNetwork

class MyIKNetwork(IKNetwork):
    def __init__(self, ...):
        super().__init__(...)
        # Add custom layers
        self.extra_layer = nn.Linear(256, 256)
    
    def forward(self, x):
        # Custom forward pass
        x = super().forward(x)
        return x
```

### Custom Loss Functions

```python
from genik.model.loss import GeNIKLoss

class MyLoss(GeNIKLoss):
    def forward(self, q_pred, q_target, p_target):
        loss, components = super().forward(q_pred, q_target, p_target)
        
        # Add regularization
        reg = 0.01 * (q_pred ** 2).mean()
        components['regularization'] = reg
        
        return loss + reg, components
```

### Working with Different Robots

GeNIK supports any robot with a URDF file:

```python
# UR10
sampler = FKSampler("robots/ur10.urdf")

# Custom robot
sampler = FKSampler("my_robot.urdf", ee_link="end_effector")
```

### Transfer Learning

```python
# Load pretrained model
model, _ = IKNetwork.load_checkpoint("panda_model.pt")

# Modify for different robot
model.output = nn.Linear(256, 6)  # UR10 has 6 joints

# Fine-tune
trainer = GeNIKTrainer(config)
trainer.model = model
trainer.train()
```

---

## Troubleshooting

### Common Issues

#### "URDF file not found"
- Check the path is correct
- Ensure file has `.urdf` extension
- Try absolute path

#### "CUDA out of memory"
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision: `config.use_amp = True`

#### "Loss is NaN"
- Reduce learning rate
- Check for invalid data (NaN in poses)
- Increase gradient clipping threshold

#### "Poor accuracy"
- Increase training data
- Increase network capacity
- Tune epsilon for clustering
- Increase training epochs

#### "Slow training"
- Use GPU: `device="cuda"`
- Increase batch size
- Enable data loader workers
- Use lazy loading for large datasets

### Getting Help

1. Check the [API Reference](api.md)
2. Review example scripts in `scripts/`
3. Open an issue on GitHub

---

## Best Practices

1. **Data Generation**
   - Use at least 100K samples for good coverage
   - Validate samples before training
   - Use appropriate epsilon for your precision needs

2. **Training**
   - Monitor TensorBoard for convergence
   - Use early stopping to avoid overfitting
   - Save checkpoints regularly

3. **Evaluation**
   - Test on held-out data
   - Verify joint limits are respected
   - Test across the full workspace

4. **Deployment**
   - Use `model.eval()` for inference
   - Benchmark inference time
   - Validate solutions with FK
