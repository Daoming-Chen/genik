# Specification: IK Training

**Capability ID:** `ik-training`  
**Version:** 1.0.0  
**Status:** Draft

## Overview

The IK Training capability defines the training pipeline for GeNIK inverse kinematics networks. It includes the mixed physics loss function, training loop, hyperparameter management, and monitoring utilities.

## ADDED Requirements

### Requirement: Mixed Physics Loss Function

The system SHALL implement a loss function that combines joint-space guidance with task-space physical constraints.

#### Scenario: Joint Guidance Loss
```
GIVEN predicted joint angles q_pred
AND ground truth joint angles q_star
WHEN computing joint guidance loss
THEN the system SHALL:
  - Compute L_joint = ||q_pred - q_star||₂²
  - Support L1 loss as alternative: ||q_pred - q_star||₁
  - Apply per-joint weighting (optional)
  - Return scalar loss value
```

#### Scenario: Task-Space Pose Loss
```
GIVEN predicted joint angles q_pred
AND target end-effector pose x_target
WHEN computing pose loss
THEN the system SHALL:
  - Compute x_pred = DiffFK(q_pred)
  - Calculate position error: e_pos = ||pos_pred - pos_target||₂²
  - Calculate orientation error: e_ori = quat_distance(quat_pred, quat_target)²
  - Combine: L_pose = w_pos × e_pos + w_ori × e_ori
  - Support configurable position/orientation weights
```

#### Scenario: Adaptive Loss Weighting
```
GIVEN current training epoch t
AND warmup period T_warmup
AND weight bounds λ_min, λ_max
WHEN computing total loss
THEN the system SHALL:
  - Calculate λ(t) = λ_min + (λ_max - λ_min) × min(t / T_warmup, 1.0)
  - Compute L_total = L_joint + λ(t) × L_pose
  - Start with small λ_min (e.g., 0.1) for initial joint space guidance
  - End with large λ_max (e.g., 1.0) for final pose accuracy
  - Log current λ value to monitoring system
```

#### Scenario: Collision Penalty (Optional)
```
GIVEN predicted joint configuration q_pred
AND collision detection model
WHEN configuration is in collision
THEN the system MAY:
  - Add collision penalty: L_collision = k × penetration_depth
  - Weight collision loss: L_total += w_collision × L_collision
  - Skip collision check during early training (performance)
  - Enable collision checking after epoch N_start
```

### Requirement: Training Loop

The system SHALL implement a robust training loop with validation, checkpointing, and early stopping.

#### Scenario: Standard Training Epoch
```
GIVEN a training dataset with DataLoader
WHEN running one training epoch
THEN the system SHALL:
  - Iterate over all batches
  - For each batch:
    1. Load (q_ref, x_target, q_star)
    2. Forward pass: q_pred = model(q_ref, x_target)
    3. Compute loss: L_total
    4. Backward pass: L_total.backward()
    5. Clip gradients to prevent explosion
    6. Update weights: optimizer.step()
    7. Zero gradients: optimizer.zero_grad()
  - Log average epoch loss
  - Update learning rate scheduler
```

#### Scenario: Validation Loop
```
GIVEN a validation dataset
WHEN performing validation (every N epochs)
THEN the system SHALL:
  - Set model to eval mode: model.eval()
  - Disable gradient computation: with torch.no_grad()
  - Compute validation metrics:
    - Average joint loss
    - Average pose loss (position and orientation separate)
    - Success rate (pose error < threshold)
  - Log all metrics to monitoring system
  - Restore training mode: model.train()
```

#### Scenario: Best Model Tracking
```
GIVEN validation metrics after each validation run
WHEN comparing to previous best validation loss
THEN the system SHALL:
  - Track best_val_loss (initialized to infinity)
  - If current_val_loss < best_val_loss:
    - Update best_val_loss = current_val_loss
    - Save checkpoint as "best_model.pt"
    - Reset early stopping counter
  - Else:
    - Increment early stopping counter
  - Log whether improvement occurred
```

#### Scenario: Early Stopping
```
GIVEN early_stopping_patience = P epochs
AND current_no_improvement_count = C
WHEN validation shows no improvement
THEN the system SHALL:
  - Increment C after each validation without improvement
  - If C ≥ P:
    - Log "Early stopping triggered"
    - Stop training
    - Load best model checkpoint
  - Allow disabling early stopping via config
```

### Requirement: Gradient Management

The system SHALL manage gradients to ensure stable and efficient training.

#### Scenario: Gradient Clipping
```
GIVEN computed gradients ∂L/∂θ after loss.backward()
AND clip threshold τ
WHEN gradients may explode
THEN the system SHALL:
  - Compute gradient norm: ||∇θ||₂
  - If ||∇θ||₂ > τ:
    - Scale gradients: ∇θ = τ × ∇θ / ||∇θ||₂
  - Use default τ = 1.0 for stability
  - Log gradient norms for monitoring
```

#### Scenario: Gradient Accumulation
```
GIVEN effective batch size B_eff > GPU memory limit
AND physical batch size B_phys that fits in memory
WHEN training with gradient accumulation
THEN the system SHALL:
  - Set accumulation steps: K = B_eff / B_phys
  - For K iterations:
    - Compute loss on batch of size B_phys
    - Scale loss by 1/K
    - Call loss.backward() (accumulates gradients)
  - After K steps:
    - Clip and apply gradients
    - Zero gradients
  - Achieve same effective batch size with limited memory
```

### Requirement: Optimizer Configuration

The system SHALL support various optimization algorithms with hyperparameter scheduling.

#### Scenario: AdamW Optimizer
```
GIVEN model parameters θ
WHEN configuring optimizer
THEN the system SHALL:
  - Use AdamW as default optimizer
  - Set learning rate: lr = 1e-3 (adjustable)
  - Set weight decay: wd = 1e-4 (L2 regularization)
  - Set beta parameters: β₁ = 0.9, β₂ = 0.999
  - Support other optimizers: Adam, SGD, RMSprop
```

#### Scenario: Learning Rate Warmup
```
GIVEN warmup period T_warmup epochs
AND base learning rate lr_base
WHEN epoch t < T_warmup
THEN the system SHALL:
  - Set lr(t) = lr_base × (t / T_warmup)
  - Linearly increase learning rate
  - Stabilize training in early epochs
  - Start warmup from lr_min = lr_base / 10
```

#### Scenario: Cosine Annealing Schedule
```
GIVEN total training epochs T_total
AND current epoch t
AND base learning rate lr_base
WHEN t ≥ T_warmup
THEN the system SHALL:
  - Compute: lr(t) = lr_min + 0.5 × (lr_base - lr_min) × (1 + cos(π × t / T_total))
  - Smoothly decay learning rate
  - Set lr_min = 1e-6 (minimum learning rate)
  - Support warm restarts (optional)
```

### Requirement: Hyperparameter Configuration

The system SHALL provide a flexible configuration system for all training hyperparameters.

#### Scenario: YAML Configuration Loading
```
GIVEN a YAML configuration file path
WHEN loading training config
THEN the system SHALL:
  - Parse YAML into structured config object
  - Validate all required fields present
  - Apply default values for optional fields
  - Raise descriptive error if validation fails
  - Log loaded configuration for reproducibility
```

#### Scenario: Configuration Schema
```
GIVEN the training configuration
THEN it SHALL include these sections:
  - data: dataset paths, batch size, num workers
  - model: architecture hyperparameters
  - optimizer: type, learning rate, weight decay
  - scheduler: warmup, total epochs, schedule type
  - loss: weights (λ_min, λ_max, w_pos, w_ori)
  - training: epochs, validation frequency, checkpoint dir
  - early_stopping: patience, min_delta
  - logging: tensorboard dir, log frequency
```

#### Scenario: Command-Line Override
```
GIVEN a base configuration file
AND command-line arguments
WHEN starting training
THEN the system SHALL:
  - Load base config from YAML
  - Override specified fields from CLI args
  - Priority: CLI args > YAML > defaults
  - Example: --learning_rate 0.001 overrides config
```

### Requirement: Training Monitoring

The system SHALL provide comprehensive monitoring and logging of training progress.

#### Scenario: TensorBoard Logging
```
GIVEN training is in progress
WHEN logging metrics
THEN the system SHALL log to TensorBoard:
  - Scalars: train_loss, val_loss, learning_rate, gradient_norm
  - Per-epoch: joint_loss, pose_loss, success_rate
  - Pose errors: position_error_mm, orientation_error_deg
  - Distributions: weight histograms, gradient histograms
  - Images: example predictions vs ground truth (optional)
```

#### Scenario: Console Progress Display
```
GIVEN training loop execution
WHEN processing batches
THEN the system SHALL display:
  - Progress bar with batch iteration
  - Current loss value
  - Estimated time remaining (ETA)
  - Throughput (samples/second)
  - Use tqdm for progress visualization
```

#### Scenario: Periodic Checkpoint Saving
```
GIVEN checkpoint frequency F (e.g., every 10 epochs)
WHEN epoch % F == 0
THEN the system SHALL:
  - Save checkpoint: "checkpoint_epoch{epoch}.pt"
  - Include: model, optimizer, epoch, metrics
  - Optionally delete old checkpoints (keep last K)
  - Always keep best model checkpoint separate
```

### Requirement: Resumable Training

The system SHALL support resuming interrupted training from checkpoints.

#### Scenario: Resume from Checkpoint
```
GIVEN a checkpoint file path
WHEN resuming training
THEN the system SHALL:
  - Load model state: model.load_state_dict(checkpoint["model"])
  - Load optimizer state: optimizer.load_state_dict(checkpoint["optimizer"])
  - Resume from epoch: start_epoch = checkpoint["epoch"] + 1
  - Restore learning rate scheduler state
  - Continue training seamlessly
  - Log "Resuming from epoch X"
```

#### Scenario: Training State Verification
```
GIVEN a loaded checkpoint
WHEN verifying training state
THEN the system SHALL:
  - Check model architecture matches (DOF, layers)
  - Verify optimizer type matches configuration
  - Validate dataset compatibility
  - Warn if hyperparameters differ from current config
  - Provide option to override config with checkpoint config
```

## Non-Functional Requirements

### Performance
- Train on 1M samples in <4 hours on single GPU (V100/A100)
- Support distributed training across multiple GPUs
- GPU utilization >80% during training
- Efficient data loading with prefetching

### Reliability
- Handle CUDA out-of-memory errors gracefully
- Automatic checkpoint saving on interruption (SIGINT)
- Validate model outputs (no NaN/Inf)
- Detect and report gradient issues

### Usability
- Single command to start training from config
- Clear error messages with resolution hints
- Real-time training visualization
- Comprehensive logging for debugging

### Reproducibility
- Fixed random seeds for deterministic training
- Log all hyperparameters and random seeds
- Track git commit hash in checkpoints
- Version control for datasets

## Dependencies

- `torch`: Neural network training
- `torch.utils.tensorboard`: Training visualization
- `pyyaml`: Configuration file parsing
- `tqdm`: Progress bar display
- `genik.data.dataset`: Training dataset
- `genik.model.network`: IK network
- `genik.model.diff_fk`: Differentiable FK layer

## Configuration

```yaml
# config/train_panda.yaml
data:
  train_dataset: "data/panda/train_dataset.pt"
  val_dataset: "data/panda/val_dataset.pt"
  batch_size: 256
  num_workers: 4
  pin_memory: true

model:
  dof: 7
  hidden_size: 256
  num_layers: 4
  dropout: 0.1

optimizer:
  type: "adamw"
  learning_rate: 0.001
  weight_decay: 0.0001
  betas: [0.9, 0.999]

scheduler:
  warmup_epochs: 10
  total_epochs: 200
  type: "cosine"
  min_lr: 1e-6

loss:
  lambda_min: 0.1
  lambda_max: 1.0
  warmup_epochs: 50
  w_position: 1.0
  w_orientation: 0.1

training:
  num_epochs: 200
  validation_frequency: 5
  checkpoint_dir: "checkpoints/panda"
  checkpoint_frequency: 10
  gradient_clip: 1.0

early_stopping:
  enabled: true
  patience: 20
  min_delta: 1e-4

logging:
  tensorboard_dir: "runs/panda"
  log_frequency: 10
  verbose: true

reproducibility:
  random_seed: 42
  cudnn_deterministic: true
  cudnn_benchmark: false
```

## API

### Training Script
```python
from genik.train.trainer import Trainer
from genik.train.config import TrainingConfig

# Load config
config = TrainingConfig.from_yaml("config/train_panda.yaml")

# Create trainer
trainer = Trainer(config)

# Start training
trainer.train()

# Or resume from checkpoint
trainer.train(resume_from="checkpoints/panda/checkpoint_epoch100.pt")
```

### Custom Training Loop
```python
import torch
from genik.model.network import GeNIKNet
from genik.model.loss import MixedPhysicsLoss
from genik.data.dataset import GeNIKDataset

# Setup
model = GeNIKNet(dof=7).cuda()
loss_fn = MixedPhysicsLoss(lambda_min=0.1, lambda_max=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

dataset = GeNIKDataset.load("data/train.pt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256)

# Training loop
for epoch in range(num_epochs):
    for q_ref, x_target, q_star in dataloader:
        q_pred = model(q_ref.cuda(), x_target.cuda())
        loss = loss_fn(q_pred, q_star.cuda(), x_target.cuda(), epoch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

## Testing

### Unit Tests
- Test loss computation produces correct shapes
- Verify gradient clipping limits gradient norms
- Validate learning rate schedule values
- Check config loading and validation

### Integration Tests
- Train on small dataset (1000 samples) for few epochs
- Verify loss decreases monotonically
- Test checkpoint save/load preserves training state
- Validate resume training continues correctly

### Regression Tests
- Ensure training converges to expected accuracy
- Compare training curves against baseline
- Verify performance metrics match expectations
- Test with multiple random seeds for stability
