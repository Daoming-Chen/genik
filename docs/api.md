# GeNIK API Reference

This document provides a comprehensive reference for the GeNIK (Generative Network for Inverse Kinematics) package API.

## Table of Contents

- [Data Module](#data-module)
  - [FKSampler](#fksampler)
  - [PoseClusterer](#poseclusterer)
  - [ConsistencyMatcher](#consistencymatcher)
  - [GeNIKDataset](#genikdataset)
- [Model Module](#model-module)
  - [DifferentiableFKLayer](#differentiablefklayer)
  - [IKNetwork](#iknetwork)
  - [ResidualIKNetwork](#residualiknetwork)
  - [GeNIKLoss](#genikloss)
- [Training Module](#training-module)
  - [TrainingConfig](#trainingconfig)
  - [GeNIKTrainer](#geniktrainer)
- [Evaluation Module](#evaluation-module)
  - [GeNIKEvaluator](#genikevaluator)
  - [Visualization Functions](#visualization-functions)

---

## Data Module

### FKSampler

Samples joint configurations and computes forward kinematics.

```python
from genik.data.sampler import FKSampler

sampler = FKSampler(
    urdf_path: str,          # Path to URDF file
    ee_link: str = None,     # End-effector link name (default: last link)
)
```

#### Methods

##### `sample(n_samples, batch_size=1024)`
Sample random joint configurations and compute their FK.

**Arguments:**
- `n_samples` (int): Number of samples to generate
- `batch_size` (int): Batch size for FK computation

**Returns:**
- `joints` (np.ndarray): Joint configurations, shape `(n_samples, n_joints)`
- `poses` (np.ndarray): End-effector poses, shape `(n_samples, 7)` [x, y, z, qw, qx, qy, qz]

##### `validate_samples(joints, poses)`
Validate that poses match FK for given joints.

**Arguments:**
- `joints` (np.ndarray): Joint configurations
- `poses` (np.ndarray): Corresponding poses

**Returns:**
- `bool`: True if all poses are valid

#### Properties

- `n_joints` (int): Number of joints
- `joint_lower` (np.ndarray): Lower joint limits
- `joint_upper` (np.ndarray): Upper joint limits

---

### PoseClusterer

Clusters poses in workspace to identify multi-solution regions.

```python
from genik.data.clustering import PoseClusterer

clusterer = PoseClusterer(
    epsilon: float = 0.05,   # Clustering radius in pose space
    pos_weight: float = 1.0, # Weight for position in distance metric
    ori_weight: float = 1.0, # Weight for orientation in distance metric
)
```

#### Methods

##### `cluster(poses)`
Cluster poses into epsilon-balls.

**Arguments:**
- `poses` (np.ndarray): Poses to cluster, shape `(n_samples, 7)`

**Returns:**
- `clusters` (List[List[int]]): List of clusters, each containing sample indices

##### `build_index(poses)`
Build KD-tree index for efficient neighbor queries.

**Arguments:**
- `poses` (np.ndarray): Poses to index

##### `find_neighbors(pose, epsilon=None)`
Find all poses within epsilon of given pose.

**Arguments:**
- `pose` (np.ndarray): Query pose, shape `(7,)`
- `epsilon` (float): Search radius (default: self.epsilon)

**Returns:**
- `neighbors` (List[int]): Indices of neighboring poses

##### `pose_distance(p1, p2)`
Compute distance between two poses.

**Arguments:**
- `p1` (np.ndarray): First pose
- `p2` (np.ndarray): Second pose

**Returns:**
- `distance` (float): Weighted pose distance

---

### ConsistencyMatcher

Generates training triplets with consistency matching.

```python
from genik.data.consistency import ConsistencyMatcher

matcher = ConsistencyMatcher(
    strategy: str = "mixed",      # "perturbation", "random", or "mixed"
    perturbation_std: float = 0.1, # Standard deviation for perturbation
    n_candidates: int = 10,       # Candidates per query
)
```

#### Methods

##### `generate_triplets(joints, poses, clusters, n_triplets=None)`
Generate training triplets from clustered data.

**Arguments:**
- `joints` (np.ndarray): Joint configurations, shape `(n_samples, n_joints)`
- `poses` (np.ndarray): Corresponding poses, shape `(n_samples, 7)`
- `clusters` (List[List[int]]): Cluster assignments from PoseClusterer
- `n_triplets` (int): Number of triplets to generate (default: len(clusters))

**Returns:**
- `triplets` (dict): Dictionary with keys:
  - `q_i`: Query joint configurations
  - `p_i`: Query poses
  - `q_c`: Candidate joint configurations
  - `p_c`: Candidate poses

---

### GeNIKDataset

PyTorch Dataset for GeNIK training.

```python
from genik.data.dataset import GeNIKDataset

dataset = GeNIKDataset(
    data_path: str,           # Path to .pt file with triplets
    lazy_load: bool = False,  # Lazy loading for large datasets
)
```

#### Methods

##### `__len__()`
Returns number of samples.

##### `__getitem__(idx)`
Get sample at index.

**Returns:**
- `sample` (dict): Dictionary with q_i, p_i, q_c, p_c tensors

#### Utility Functions

##### `create_data_splits(dataset, train_ratio, val_ratio, test_ratio, seed=42)`
Split dataset into train/val/test.

**Arguments:**
- `dataset`: GeNIKDataset instance
- `train_ratio` (float): Training set fraction
- `val_ratio` (float): Validation set fraction
- `test_ratio` (float): Test set fraction
- `seed` (int): Random seed

**Returns:**
- `train_dataset`, `val_dataset`, `test_dataset`: Subset datasets

---

## Model Module

### DifferentiableFKLayer

PyTorch module for differentiable forward kinematics.

```python
from genik.model.diff_fk import DifferentiableFKLayer

fk_layer = DifferentiableFKLayer(
    urdf_path: str,          # Path to URDF file
    ee_link: str = None,     # End-effector link (default: last link)
    device: str = "cpu",     # Device for computation
)
```

#### Methods

##### `forward(joints)`
Compute FK for batch of joint configurations.

**Arguments:**
- `joints` (torch.Tensor): Joint angles, shape `(batch, n_joints)`

**Returns:**
- `poses` (torch.Tensor): End-effector poses, shape `(batch, 7)`

##### `get_joint_limits()`
Get joint limits as tensors.

**Returns:**
- `lower` (torch.Tensor): Lower limits
- `upper` (torch.Tensor): Upper limits

#### Properties

- `n_joints` (int): Number of joints

---

### IKNetwork

Standard MLP architecture for IK.

```python
from genik.model.network import IKNetwork

network = IKNetwork(
    input_dim: int = 7,       # Pose dimension (position + quaternion)
    output_dim: int = 7,      # Number of joints
    hidden_dim: int = 256,    # Hidden layer size
    n_layers: int = 4,        # Number of hidden layers
    dropout: float = 0.1,     # Dropout rate
    activation: str = "relu", # Activation function
)
```

#### Methods

##### `forward(pose)`
Predict joint angles from pose.

**Arguments:**
- `pose` (torch.Tensor): End-effector pose, shape `(batch, 7)`

**Returns:**
- `joints` (torch.Tensor): Predicted joints, shape `(batch, n_joints)`

##### `save_checkpoint(path, config=None, optimizer=None, epoch=None)`
Save model checkpoint.

##### `load_checkpoint(path, device="cpu")` (classmethod)
Load model from checkpoint.

**Returns:**
- `model`: Loaded IKNetwork instance
- `checkpoint` (dict): Full checkpoint dictionary

---

### ResidualIKNetwork

IK network with residual connections.

```python
from genik.model.network import ResidualIKNetwork

network = ResidualIKNetwork(
    input_dim: int = 7,
    output_dim: int = 7,
    hidden_dim: int = 256,
    n_blocks: int = 4,        # Number of residual blocks
    dropout: float = 0.1,
)
```

Same interface as IKNetwork.

---

### GeNIKLoss

Mixed physics loss with adaptive weighting.

```python
from genik.model.loss import GeNIKLoss

loss_fn = GeNIKLoss(
    fk_layer: DifferentiableFKLayer,
    lambda_pose: float = 1.0,        # Pose loss weight
    adaptive_lambda: bool = True,    # Use adaptive scheduling
    lambda_warmup_epochs: int = 10,  # Warmup period
    pos_weight: float = 1.0,         # Position weight in pose loss
    ori_weight: float = 1.0,         # Orientation weight in pose loss
)
```

#### Methods

##### `forward(q_pred, q_target, p_target)`
Compute total loss.

**Arguments:**
- `q_pred` (torch.Tensor): Predicted joints
- `q_target` (torch.Tensor): Target joints (from consistency matching)
- `p_target` (torch.Tensor): Target pose

**Returns:**
- `loss` (torch.Tensor): Total loss value
- `components` (dict): Individual loss components {"joint": ..., "pose": ...}

##### `joint_loss(q_pred, q_target)`
Compute joint-space loss (MSE).

##### `pose_loss(q_pred, p_target)`
Compute pose-space loss using differentiable FK.

##### `update_lambda(epoch)`
Update lambda based on training progress.

---

## Training Module

### TrainingConfig

Configuration dataclass for training.

```python
from genik.train.config import TrainingConfig, OptimizerConfig, LossConfig

config = TrainingConfig(
    # Data
    urdf_path: str,
    data_path: str,
    output_dir: str = "outputs/genik",
    
    # Model
    hidden_dim: int = 256,
    n_layers: int = 4,
    dropout: float = 0.1,
    use_residual: bool = True,
    
    # Training
    epochs: int = 100,
    batch_size: int = 256,
    
    # Optimizer
    optimizer: OptimizerConfig = OptimizerConfig(),
    
    # Loss
    loss: LossConfig = LossConfig(),
    
    # Misc
    seed: int = 42,
    device: str = "auto",
)
```

#### Methods

##### `save(path)`
Save config to YAML file.

##### `load(path)` (classmethod)
Load config from YAML file.

---

### GeNIKTrainer

Main training loop manager.

```python
from genik.train.trainer import GeNIKTrainer

trainer = GeNIKTrainer(
    config: TrainingConfig,
    model: IKNetwork = None,      # Optional pre-created model
    fk_layer: DifferentiableFKLayer = None,
)
```

#### Methods

##### `train()`
Run full training loop.

**Returns:**
- `results` (dict): Training results with metrics history

##### `validate()`
Run validation epoch.

**Returns:**
- `metrics` (dict): Validation metrics

##### `save_checkpoint(path, is_best=False)`
Save training checkpoint.

##### `load_checkpoint(path)`
Resume training from checkpoint.

#### Properties

- `model`: The IK network
- `optimizer`: The optimizer
- `best_val_loss`: Best validation loss seen

---

## Evaluation Module

### GeNIKEvaluator

Comprehensive model evaluation.

```python
from genik.eval.metrics import GeNIKEvaluator

evaluator = GeNIKEvaluator(
    model: IKNetwork,
    fk_layer: DifferentiableFKLayer,
    device: str = "cpu",
)
```

#### Methods

##### `evaluate(dataloader, show_progress=True)`
Evaluate model on dataset.

**Arguments:**
- `dataloader`: PyTorch DataLoader
- `show_progress` (bool): Show progress bar

**Returns:**
- `results` (dict): Comprehensive evaluation results:
  - `n_samples`: Number of samples evaluated
  - `position_error_mm`: Position error statistics (mean, std, median, max)
  - `orientation_error_deg`: Orientation error statistics
  - `success_rates`: Success rates at various thresholds
  - `raw_errors`: Raw error arrays for plotting

##### `evaluate_batch(poses, joints_target)`
Evaluate single batch.

**Returns:**
- `results` (dict): Batch metrics

##### `compute_position_error(pose_pred, pose_target)`
Compute position errors in meters.

##### `compute_orientation_error(quat_pred, quat_target)`
Compute orientation errors in radians.

##### `compute_success_rate(pos_errors, ori_errors, pos_threshold, ori_threshold)`
Compute success rate at given thresholds.

##### `measure_inference_time(batch_sizes=[1, 8, 32, 64, 128, 256])`
Measure inference timing.

**Returns:**
- `timing` (dict): Timing results per batch size with mean_ms, std_ms, throughput_qps

##### `check_joint_limits(dataloader)`
Check for joint limit violations.

**Returns:**
- `results` (dict): violation_rate, max_violation_rad, violations_per_joint

##### `generate_report(results)`
Generate human-readable evaluation report.

**Returns:**
- `report` (str): Formatted report string

---

### Visualization Functions

```python
from genik.eval.visualize import (
    plot_error_histogram,
    plot_error_vs_threshold,
    plot_workspace_heatmap,
    plot_inference_time,
)
```

##### `plot_error_histogram(pos_errors_mm, ori_errors_deg, save_path=None, show=True)`
Plot error distribution histograms.

##### `plot_error_vs_threshold(pos_errors_mm, ori_errors_deg, save_path=None, show=True)`
Plot success rate vs threshold curves.

##### `plot_workspace_heatmap(poses, errors, save_path=None, show=True)`
Plot spatial distribution of errors.

##### `plot_inference_time(timing_results, save_path=None, show=True)`
Plot inference timing by batch size.

---

## Quick Start Example

```python
import torch
from genik.data.sampler import FKSampler
from genik.data.clustering import PoseClusterer
from genik.data.consistency import ConsistencyMatcher
from genik.model.network import ResidualIKNetwork
from genik.model.diff_fk import DifferentiableFKLayer
from genik.model.loss import GeNIKLoss
from genik.train.config import TrainingConfig
from genik.train.trainer import GeNIKTrainer

# 1. Generate data
sampler = FKSampler("robots/panda_arm.urdf")
joints, poses = sampler.sample(100000)

clusterer = PoseClusterer(epsilon=0.05)
clusters = clusterer.cluster(poses)

matcher = ConsistencyMatcher()
triplets = matcher.generate_triplets(joints, poses, clusters)
torch.save(triplets, "data/triplets.pt")

# 2. Train model
config = TrainingConfig(
    urdf_path="robots/panda_arm.urdf",
    data_path="data/triplets.pt",
    epochs=100,
)
trainer = GeNIKTrainer(config)
results = trainer.train()

# 3. Evaluate
from genik.eval.metrics import GeNIKEvaluator
evaluator = GeNIKEvaluator(trainer.model, trainer.fk_layer)
eval_results = evaluator.evaluate(test_loader)
print(evaluator.generate_report(eval_results))
```
