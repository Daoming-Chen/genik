# Specification: IK Neural Network

**Capability ID:** `ik-neural-network`  
**Version:** 1.0.0  
**Status:** Draft

## Overview

The IK Neural Network capability defines the architecture and components for the GeNIK inverse kinematics solver. It includes the conditional IK network, differentiable forward kinematics layer, and model utilities for training and inference.

## ADDED Requirements

### Requirement: Conditional IK Network Architecture

The system SHALL implement a neural network that maps reference joint configurations and target poses to predicted joint configurations.

#### Scenario: Multi-Layer Perceptron with Residual Connections
```
GIVEN a robot with N degrees of freedom
WHEN constructing the IK network
THEN the system SHALL:
  - Accept input: [q_ref (N dims), x_target (7 dims)]  # pose as [pos(3), quat(4)]
  - Apply embedding: Linear(N+7, hidden_size)
  - Stack L residual blocks: Linear(h, h) + ReLU + Dropout
  - Output: Linear(h, N) + q_ref  # residual connection
  - Use hidden_size ≥ 256 for robots with N ≥ 6 DOF
```

#### Scenario: Residual Block Design
```
GIVEN a residual block in the network
WHEN processing a hidden state h
THEN the block SHALL:
  - Compute: h' = Dropout(ReLU(Linear(h)))
  - Return: h + h'  # skip connection
  - Support configurable dropout rate (default 0.1)
  - Enable gradient flow through skip connections
```

#### Scenario: Batch Processing
```
GIVEN a batch of B input pairs (q_ref, x_target)
WHEN performing forward pass
THEN the network SHALL:
  - Process all B samples in parallel
  - Output B predicted joint configurations
  - Support batch sizes up to 1024 on modern GPUs
  - Utilize batch normalization for stable training (optional)
```

### Requirement: Differentiable Forward Kinematics Layer

The system SHALL implement a PyTorch-compatible forward kinematics layer that supports gradient backpropagation.

#### Scenario: Chain-of-Transformations FK
```
GIVEN a kinematic chain with N joints
AND joint angles θ = [θ₁, θ₂, ..., θₙ]
WHEN computing forward kinematics
THEN the system SHALL:
  - Initialize T = I₄ (4×4 identity matrix)
  - For each joint i: T = T @ Joint_Transform(θᵢ)
  - Extract pose: position = T[:3, 3], rotation = T[:3, :3]
  - Convert rotation to quaternion
  - Return 7D pose vector [x, y, z, qw, qx, qy, qz]
```

#### Scenario: Gradient Computation via Autograd
```
GIVEN predicted joint angles θ_pred (PyTorch tensor)
WHEN computing pose = DiffFK(θ_pred)
AND backpropagating loss gradient ∂L/∂pose
THEN the system SHALL:
  - Automatically compute ∂L/∂θ_pred using PyTorch autograd
  - NOT require manual Jacobian implementation
  - Maintain numerical stability for small angle changes
  - Support double precision for high-accuracy applications
```

#### Scenario: Batched FK Computation
```
GIVEN a batch of B joint configurations [B × N]
WHEN computing forward kinematics
THEN the system SHALL:
  - Process batch in parallel: [B × N] → [B × 7]
  - Utilize matrix operations for efficiency
  - Achieve >10× speedup vs. loop-based implementation
  - Support GPU acceleration with CUDA
```

### Requirement: Joint Type Support

The system SHALL support different joint types commonly found in robot manipulators.

#### Scenario: Revolute Joint Transform
```
GIVEN a revolute joint with:
  - Axis of rotation: a = [ax, ay, az]
  - Joint angle: θ
  - Joint origin: [x, y, z, roll, pitch, yaw]
WHEN computing joint transformation
THEN the system SHALL:
  - Compute rotation matrix using Rodrigues' formula
  - Apply translation offset
  - Return 4×4 homogeneous transformation matrix
  - Ensure result is PyTorch differentiable
```

#### Scenario: Prismatic Joint Transform
```
GIVEN a prismatic joint with:
  - Translation axis: a = [ax, ay, az]
  - Joint displacement: d
  - Joint origin transform
WHEN computing joint transformation
THEN the system SHALL:
  - Apply translation along axis: t = d × a
  - Combine with origin transform
  - Return 4×4 homogeneous transformation matrix
  - Support gradient flow through displacement d
```

#### Scenario: Fixed Joint Handling
```
GIVEN a fixed joint in the kinematic chain
WHEN computing forward kinematics
THEN the system SHALL:
  - Apply constant transformation (no learnable parameters)
  - Not include in joint angle input vector
  - Properly chain transformations across fixed joints
```

### Requirement: Pose Representation and Distance Metrics

The system SHALL provide utilities for pose manipulation and distance computation.

#### Scenario: Quaternion Distance
```
GIVEN two unit quaternions q₁ and q₂
WHEN computing orientation distance
THEN the system SHALL:
  - Compute dot product: d = |q₁ · q₂|
  - Return angle: θ = arccos(clip(d, -1, 1))
  - Handle antipodal ambiguity (q and -q represent same rotation)
  - Return distance in radians [0, π]
```

#### Scenario: SE(3) Pose Distance
```
GIVEN two poses p₁ = [pos₁, quat₁] and p₂ = [pos₂, quat₂]
WHEN computing pose distance
THEN the system SHALL:
  - Compute position distance: d_pos = ||pos₁ - pos₂||₂
  - Compute orientation distance: d_ori = quat_distance(quat₁, quat₂)
  - Support weighted combination: d = √(w_pos × d_pos² + w_ori × d_ori²)
  - Provide separate metrics for position and orientation
```

#### Scenario: Pose Normalization
```
GIVEN a pose with quaternion component
WHEN preparing input for neural network
THEN the system SHALL:
  - Ensure quaternion is unit-norm: q = q / ||q||
  - Normalize position to workspace bounds (optional)
  - Handle numerical precision issues (quaternions near zero)
```

### Requirement: Model Utilities

The system SHALL provide utilities for model management, including saving, loading, and inspection.

#### Scenario: Model Checkpointing
```
GIVEN a trained IK network model
WHEN saving a checkpoint
THEN the system SHALL save:
  - Model state dictionary (weights and biases)
  - Optimizer state (for resume training)
  - Training hyperparameters
  - Epoch number and validation metrics
  - Robot URDF path and configuration
  - Timestamp and version information
```

#### Scenario: Model Loading with Validation
```
GIVEN a checkpoint file path
WHEN loading a model
THEN the system SHALL:
  - Verify checkpoint compatibility (architecture, DOF)
  - Load weights with strict=True (fail on mismatch)
  - Optionally load optimizer state for resume training
  - Log loaded configuration for reproducibility
  - Raise clear error if checkpoint corrupted or incompatible
```

#### Scenario: Model Inspection
```
GIVEN a constructed IK network
WHEN inspecting the model
THEN the system SHALL provide:
  - Total parameter count
  - Trainable parameter count
  - Layer-wise parameter breakdown
  - Input/output shapes for each layer
  - Estimated memory footprint
```

### Requirement: Model Export for Deployment

The system SHALL support exporting trained models for efficient inference in production environments.

#### Scenario: TorchScript Export
```
GIVEN a trained IK network
WHEN exporting to TorchScript
THEN the system SHALL:
  - Trace or script the model for static graph
  - Include DiffFK layer in exported graph
  - Validate exported model produces identical outputs
  - Reduce model size via quantization (optional)
  - Support loading in C++ applications
```

#### Scenario: ONNX Export
```
GIVEN a trained IK network
WHEN exporting to ONNX format
THEN the system SHALL:
  - Convert all operations to ONNX operators
  - Validate conversion with example inputs
  - Support opset version ≥ 11
  - Enable deployment on non-PyTorch runtimes
```

## Non-Functional Requirements

### Performance
- Forward pass <5ms for single sample on GPU (7-DOF robot)
- Forward pass <20ms for single sample on CPU
- Batched inference: >1000 queries/second on GPU
- DiffFK computation: <1ms per sample

### Memory Efficiency
- Model size: <10MB for standard architecture (256 hidden units)
- Training memory: <4GB GPU for batch size 256
- Support models up to 10M parameters

### Numerical Stability
- Handle joint angles near ±π (wraparound)
- Stable gradient flow through FK layer
- No NaN or Inf in forward/backward pass
- Support both float32 and float64 precision

### Extensibility
- Easy to swap network architecture (MLP, ResNet, Transformer)
- Modular DiffFK layer (can be used independently)
- Support custom joint types via plugin interface
- Enable multi-robot models with shared weights

## Dependencies

- `torch`: Neural network implementation and autograd
- `numpy`: Numerical operations for data preprocessing
- `genik.urdf.URDF`: Robot model definition

## Configuration

```yaml
model:
  architecture:
    type: "mlp"  # or "resnet", "transformer"
    hidden_size: 256
    num_layers: 4
    dropout: 0.1
    use_batch_norm: false
  
  diff_fk:
    precision: "float32"  # or "float64"
    use_gpu: true
  
  pose_representation:
    position_normalization: false
    quaternion_convention: "wxyz"  # or "xyzw"
  
  export:
    format: "torchscript"  # or "onnx"
    quantization: false
```

## API

### Network Construction
```python
from genik.model.network import GeNIKNet

model = GeNIKNet(
    dof=7,
    hidden_size=256,
    num_layers=4,
    dropout=0.1
)

# Forward pass
q_pred = model(q_ref, x_target)
```

### Differentiable FK
```python
from genik.model.diff_fk import DifferentiablFK

diff_fk = DifferentiablFK(urdf_path="robots/panda_arm.urdf")
pose = diff_fk(joint_angles)  # Returns [B × 7] tensor

# Gradient computation
loss = pose_distance(pose, target_pose)
loss.backward()  # Gradients flow through FK
```

### Model Utilities
```python
from genik.model.utils import save_checkpoint, load_checkpoint

# Save
save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=100,
    path="checkpoints/model_epoch100.pt"
)

# Load
checkpoint = load_checkpoint("checkpoints/model_epoch100.pt")
model.load_state_dict(checkpoint["model"])
```

### Model Export
```python
from genik.model.export import export_torchscript, export_onnx

# TorchScript
export_torchscript(model, "model.pt")

# ONNX
export_onnx(model, "model.onnx", opset_version=13)
```

## Testing

### Unit Tests
- Test network forward pass shape correctness
- Verify residual connections add identity shortcut
- Validate DiffFK matches analytical FK (within tolerance)
- Check gradient computation via finite differences

### Gradient Tests
```python
# Verify gradients using finite differences
def test_diff_fk_gradients():
    joint_angles = torch.randn(7, requires_grad=True)
    pose = diff_fk(joint_angles)
    loss = pose.sum()
    loss.backward()
    
    # Numerical gradient
    eps = 1e-5
    numerical_grad = []
    for i in range(7):
        theta_plus = joint_angles.clone().detach()
        theta_plus[i] += eps
        theta_minus = joint_angles.clone().detach()
        theta_minus[i] -= eps
        
        grad = (diff_fk(theta_plus).sum() - diff_fk(theta_minus).sum()) / (2 * eps)
        numerical_grad.append(grad)
    
    assert torch.allclose(joint_angles.grad, torch.tensor(numerical_grad), atol=1e-3)
```

### Integration Tests
- Test full forward pass with real robot URDF
- Verify model can overfit small dataset (sanity check)
- Test checkpoint save/load preserves weights exactly
- Validate exported models produce identical outputs

### Performance Tests
- Benchmark inference time on CPU and GPU
- Measure memory usage for different batch sizes
- Profile gradient computation overhead
- Test scalability with different DOF (3, 6, 7, 10)
