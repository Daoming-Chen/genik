# Design Document: GeNIK Algorithm

**Change ID:** `implement-genik-algorithm`

## Architecture Overview

GeNIK follows a three-stage pipeline architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: Data Preprocessing                  │
│  ┌────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │ FK Sampling├───►│Pose Clustering├───►│Consistency Matching│  │
│  └────────────┘    └──────────────┘    └─────────┬──────────┘  │
└────────────────────────────────────────────────────┼─────────────┘
                                                     │
                                         ┌───────────▼───────────┐
                                         │  Training Dataset     │
                                         │  (q_in, x_target, q*) │
                                         └───────────┬───────────┘
                                                     │
┌────────────────────────────────────────────────────┼─────────────┐
│                      STAGE 2: Training                           │
│  ┌──────────────┐    ┌──────────────┐    ┌───────▼──────────┐  │
│  │ Data Loader  ├───►│ IK Network Φ ├───►│  Mixed Physics   │  │
│  └──────────────┘    └───────┬──────┘    │  Loss Function   │  │
│                              │            │  (joint + pose)  │  │
│                      ┌───────▼────────┐   └──────────────────┘  │
│                      │ Diff-FK Layer  │                          │
│                      └────────────────┘                          │
└──────────────────────────────────────────────────────────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │  Trained Model Φ*      │
                       └───────────┬────────────┘
                                   │
┌──────────────────────────────────┼───────────────────────────────┐
│                    STAGE 3: Evaluation                           │
│  ┌──────────────┐    ┌──────────▼────┐    ┌──────────────────┐  │
│  │ Test Queries ├───►│ IK Inference  ├───►│ FK Validation    │  │
│  └──────────────┘    └───────────────┘    └────────┬─────────┘  │
│                                                     │             │
│                                         ┌───────────▼──────────┐ │
│                                         │ Error Metrics & Viz  │ │
│                                         └──────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
genik/
├── __init__.py                 # Package initialization
├── urdf.py                     # Existing URDF parser (modified)
├── kinematics.py               # FK/Diff-FK implementation
├── data/
│   ├── __init__.py
│   ├── sampler.py              # FK sampling
│   ├── clustering.py           # Pose space clustering
│   ├── consistency.py          # Consistency matching
│   └── dataset.py              # PyTorch Dataset class
├── model/
│   ├── __init__.py
│   ├── network.py              # IK network architecture
│   ├── diff_fk.py              # Differentiable FK layer
│   └── loss.py                 # Mixed physics loss
├── train/
│   ├── __init__.py
│   ├── trainer.py              # Training loop
│   └── config.py               # Training configuration
└── eval/
    ├── __init__.py
    ├── metrics.py              # Evaluation metrics
    └── visualize.py            # Visualization tools

scripts/
├── generate_dataset.py         # Data generation script
├── train_genik.py              # Training script
└── evaluate_genik.py           # Evaluation script

tests/
├── test_kinematics.py
├── test_data_generation.py
├── test_network.py
└── test_evaluation.py

data/                           # Generated datasets
└── panda/
    ├── fk_samples.pt
    ├── pose_clusters.pt
    └── train_dataset.pt
```

## Key Design Decisions

### 1. Data Preprocessing Strategy

**Decision**: Offline preprocessing with serialized datasets

**Rationale**:
- Pose clustering is computationally expensive (O(N²) in worst case)
- Training should not be blocked by dynamic data generation
- Pre-computed datasets enable reproducible experiments
- Enables easy dataset sharing and versioning

**Implementation**:
- Use KD-Tree for efficient nearest neighbor search during clustering
- Store pose-to-candidate-solutions mapping as dictionary
- Serialize datasets using PyTorch's `.pt` format for compatibility

### 2. Consistency Matching Strategy

**Decision**: Minimum distance in joint space

**Rationale**:
- Euclidean distance in joint space is simple and effective
- Ensures smooth trajectories when used in sequential control
- Avoids need for weighted distance metrics (configuration-dependent)

**Alternative Considered**: Weighted distance with joint limits
- Pro: Could account for joint ranges
- Con: Adds complexity without clear benefit in preliminary tests

### 3. Network Architecture

**Decision**: MLP with residual connections

**Architecture**:
```python
Input: [q_in (dof), x_target (7)]  # 7 = position(3) + quaternion(4)
↓
Embedding: Linear(dof+7, 256)
↓
Residual Blocks: [Linear(256, 256) + ReLU + Dropout] × 4
↓
Output: Linear(256, dof) + q_in  # Residual connection
```

**Rationale**:
- Residual connection from `q_in` encourages smooth updates
- Dropout prevents overfitting on large datasets
- 256 hidden units provide sufficient capacity for 7-DOF robots

**Alternative Considered**: Transformer architecture
- Pro: Better handling of sequential dependencies
- Con: Overkill for single-step IK; adds training complexity

### 4. Differentiable FK Implementation

**Decision**: PyTorch autograd-compatible FK computation

**Implementation**:
```python
def diff_fk(joint_angles, urdf_chain):
    """Compute FK with gradient support"""
    T = torch.eye(4)
    for joint, angle in zip(urdf_chain, joint_angles):
        T = T @ joint_transform(joint, angle)
    return T  # Returns 4×4 transformation matrix
```

**Rationale**:
- Pure PyTorch operations enable automatic differentiation
- No need for manual Jacobian computation
- Supports batched operations for efficient training

### 5. Loss Function Design

**Decision**: Adaptive weight schedule for pose loss

**Implementation**:
```python
λ(epoch) = λ_min + (λ_max - λ_min) × min(epoch / warmup_epochs, 1.0)
```

**Rationale**:
- Early training: prioritize joint guidance (λ_min ≈ 0.1)
- Late training: emphasize pose accuracy (λ_max ≈ 1.0)
- Smooth transition prevents training instability

### 6. Pose Error Metric

**Decision**: Separate position and orientation errors

**Implementation**:
```python
E_pos = ||p_pred - p_target||_2                    # mm
E_ori = arccos(|q_pred · q_target|)                # radians
success = (E_pos < 1mm) AND (E_ori < 0.0175 rad)   # 1° threshold
```

**Rationale**:
- Decoupled metrics easier to interpret and debug
- Industry-standard thresholds for robotics applications
- Avoids arbitrary weighting between position and orientation

## Data Flow

### Training Data Generation

1. **FK Sampling** (1M samples)
   ```python
   Input: Robot URDF
   Output: {(q_i, x_i)} where x_i = FK(q_i)
   Storage: fk_samples.pt (~80MB for 1M samples)
   ```

2. **Pose Clustering** (ε = 1mm position, 1° orientation)
   ```python
   Input: {(q_i, x_i)}
   Output: Map<pose, List<joint_configs>>
   Storage: pose_clusters.pt (~200MB for 1M samples)
   ```

3. **Consistency Matching** (3M training triplets)
   ```python
   Input: pose_clusters
   Output: {(q_in, x_target, q*)}
   Storage: train_dataset.pt (~240MB for 3M triplets)
   ```

### Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        q_in, x_target, q_star = batch
        
        # Forward pass
        q_pred = model(q_in, x_target)
        
        # Compute losses
        L_joint = ||q_pred - q_star||²
        x_pred = diff_fk(q_pred)
        L_pose = ||x_pred ⊖ x_target||²
        L_total = L_joint + λ(epoch) × L_pose
        
        # Backward pass
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
```

### Inference

```python
Input: (q_prev, x_desired)
Output: q_next = Φ(q_prev, x_desired)
Latency: ~5ms on GPU, ~20ms on CPU (7-DOF robot)
```

## Performance Considerations

### Memory Requirements
- **Training**: ~4GB GPU memory (batch size 256, 7-DOF robot)
- **Dataset**: ~500MB for 3M training samples
- **Model**: ~1MB (MLP with 256 hidden units)

### Computational Complexity
- **FK Sampling**: O(N × DOF) - parallelizable
- **Pose Clustering**: O(N log N) with KD-Tree
- **Training**: O(epochs × N / batch_size × DOF)
- **Inference**: O(DOF × hidden_units)

### Scalability
- Linear scaling with DOF (tested up to 7-DOF)
- Parallelizable data generation across multiple processes
- GPU acceleration for training and inference

## Testing Strategy

### Unit Tests
- FK correctness against analytical solutions
- Diff-FK gradient correctness via finite differences
- Pose distance metrics symmetry and triangle inequality

### Integration Tests
- End-to-end data generation pipeline
- Training convergence on small dataset
- Evaluation metrics computation

### Validation Tests
- Compare against analytical IK (if available)
- Test on workspace boundaries and singularities
- Stress test with random initializations

## Monitoring and Debugging

### Training Metrics
- Joint loss, pose loss, total loss per epoch
- Validation metrics every N epochs
- Gradient norms to detect vanishing/exploding gradients

### Debugging Tools
- Visualize predicted joint configurations
- Overlay FK results on target poses
- Heatmap of errors across workspace

## Future Extensibility

### Planned Extensions
1. **Multi-robot support**: Generalize URDF loading and FK computation
2. **Collision avoidance**: Add collision loss term during training
3. **Trajectory optimization**: Extend to sequential IK with temporal smoothness
4. **Real-time integration**: ROS wrapper for live control

### API Stability
- Core interfaces (IK query, FK computation) will remain stable
- Internal implementation can be optimized without breaking changes
- Dataset format versioned for backward compatibility
