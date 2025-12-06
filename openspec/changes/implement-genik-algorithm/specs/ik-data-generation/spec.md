# Specification: IK Data Generation

**Capability ID:** `ik-data-generation`  
**Version:** 1.0.0  
**Status:** Draft

## Overview

The IK Data Generation capability provides tools for generating large-scale training datasets for inverse kinematics neural networks. It implements forward kinematics sampling, pose space clustering, and consistency-based sample matching to create high-quality training data.

## ADDED Requirements

### Requirement: Forward Kinematics Sampling

The system SHALL generate large-scale datasets by sampling joint configurations and computing their corresponding end-effector poses.

#### Scenario: Uniform Joint Space Sampling
```
GIVEN a robot URDF file with N degrees of freedom
AND joint limits defined for each joint
WHEN the user requests M samples
THEN the system SHALL:
  - Generate M random joint configurations uniformly within joint limits
  - Compute forward kinematics for each configuration
  - Return a dataset of (joint_config, pose) pairs
  - Complete sampling in O(M × N) time complexity
```

#### Scenario: Batched FK Computation
```
GIVEN a batch of K joint configurations
WHEN computing forward kinematics
THEN the system SHALL:
  - Process all K configurations in a single batch operation
  - Utilize GPU acceleration if available
  - Return K corresponding end-effector poses
  - Achieve at least 10× speedup compared to sequential processing
```

#### Scenario: Deterministic Sampling
```
GIVEN a random seed value
WHEN generating joint samples
THEN the system SHALL:
  - Produce identical samples across multiple runs with same seed
  - Enable reproducible dataset generation
  - Support dataset versioning and comparison
```

### Requirement: Pose Space Clustering

The system SHALL cluster joint configurations by their end-effector poses to identify multi-solution regions in the workspace.

#### Scenario: Epsilon-Ball Clustering
```
GIVEN a dataset of (joint_config, pose) pairs
AND a distance threshold ε (epsilon)
WHEN performing pose clustering
THEN the system SHALL:
  - Group all joint configs whose poses are within ε distance
  - Use separate thresholds for position (mm) and orientation (degrees)
  - Build an efficient index structure (KD-Tree) for O(log N) queries
  - Return a mapping: pose → list[joint_configs]
```

#### Scenario: Multi-Solution Detection
```
GIVEN a clustered pose-to-configs mapping
WHEN analyzing a specific pose region
THEN the system SHALL:
  - Identify poses with multiple distinct joint configurations
  - Classify solutions by configuration type (e.g., elbow-up/down)
  - Report statistics: percentage of workspace with multi-solutions
  - Enable targeted sampling in ambiguous regions
```

#### Scenario: Efficient Neighbor Search
```
GIVEN a query pose x_query
WHEN searching for similar poses in the dataset
THEN the system SHALL:
  - Return all poses within ε distance in O(log N) time
  - Support batch queries for multiple poses simultaneously
  - Use Euclidean distance for position and geodesic distance for orientation
```

### Requirement: Consistency-Based Sample Matching

The system SHALL generate training samples with consistency labels to avoid mode averaging during neural network training.

#### Scenario: Reference-Guided Matching
```
GIVEN a target pose x_target with candidate solutions Q = {q₁, q₂, ..., qₙ}
AND a reference joint configuration q_ref
WHEN generating a training sample
THEN the system SHALL:
  - Compute distances: d_i = ||qᵢ - q_ref||₂ for all i
  - Select q* = argmin(d_i) as the ground truth label
  - Return training triplet (q_ref, x_target, q*)
  - Ensure q* is the geometrically closest solution to q_ref
```

#### Scenario: Perturbation Strategy
```
GIVEN a valid joint configuration q₀ that reaches pose x
WHEN generating a reference input q_ref
THEN the system SHALL:
  - Add Gaussian noise: q_ref = q₀ + N(0, σ²I)
  - Clip q_ref to joint limits
  - Ensure q_ref != q₀ (avoid trivial learning)
  - Support configurable noise level σ
```

#### Scenario: Mixed Sampling Strategy
```
GIVEN a target pose with multiple candidate solutions
WHEN generating training samples
THEN the system SHALL support three strategies:
  - Perturbation: q_ref = candidate + noise (60% probability)
  - Random: q_ref sampled uniformly from joint space (30% probability)
  - Near-boundary: q_ref near workspace boundaries (10% probability)
AND generate multiple samples per pose for data augmentation
```

### Requirement: Dataset Management

The system SHALL provide tools for serializing, loading, and managing training datasets efficiently.

#### Scenario: PyTorch Dataset Serialization
```
GIVEN a generated training dataset with M samples
WHEN saving to disk
THEN the system SHALL:
  - Serialize as PyTorch-compatible .pt file
  - Include metadata: robot name, sample count, epsilon, timestamp
  - Compress data to reduce storage (target: <1KB per sample)
  - Validate data integrity with checksums
```

#### Scenario: Train/Val/Test Split
```
GIVEN a complete dataset of M samples
WHEN creating data splits
THEN the system SHALL:
  - Split into train/val/test (e.g., 80%/10%/10%)
  - Ensure no pose overlap between splits
  - Maintain stratification if multi-solution regions identified
  - Support custom split ratios via configuration
```

#### Scenario: Lazy Loading for Large Datasets
```
GIVEN a dataset file larger than available RAM
WHEN loading for training
THEN the system SHALL:
  - Support memory-mapped file access
  - Load samples on-demand during iteration
  - Implement efficient caching for frequently accessed samples
  - Avoid loading entire dataset into memory
```

## Non-Functional Requirements

### Performance
- Generate 1M FK samples in <10 minutes on CPU
- Cluster 1M samples with ε=1mm in <30 minutes
- Generate 3M training triplets in <1 hour
- Support datasets up to 100M samples (memory-mapped)

### Scalability
- Linear scaling with sample count for FK sampling
- O(N log N) complexity for clustering with KD-Tree
- Parallel processing support for multi-core CPUs
- GPU acceleration for batched FK computation

### Reliability
- Validate all generated samples (FK within tolerance)
- Detect and report degenerate configurations
- Handle edge cases: singularities, joint limits, collision
- Provide progress tracking and ETA for long operations

### Usability
- CLI interface for dataset generation
- Progress bars with time estimates
- Configurable via YAML files
- Comprehensive error messages with recovery suggestions

## Dependencies

- `genik.urdf.URDF`: Robot model loading and FK computation
- `numpy`: Numerical operations and array manipulation
- `scipy.spatial.KDTree`: Efficient nearest neighbor search
- `torch`: Dataset serialization and tensor operations

## Configuration

```yaml
data_generation:
  fk_sampling:
    num_samples: 1000000
    random_seed: 42
    batch_size: 10000
    use_gpu: true
  
  clustering:
    epsilon_position: 0.001  # meters (1mm)
    epsilon_orientation: 0.017453  # radians (1 degree)
    index_type: "kdtree"
  
  consistency:
    samples_per_pose: 3
    perturbation_sigma: 0.1
    strategy_weights:
      perturbation: 0.6
      random: 0.3
      boundary: 0.1
  
  dataset:
    output_dir: "data/panda"
    split_ratios: [0.8, 0.1, 0.1]
    compression: true
```

## API

### FK Sampling
```python
from genik.data.sampler import UniformSampler

sampler = UniformSampler(urdf_path="robots/panda_arm.urdf")
samples = sampler.generate(num_samples=1000000, seed=42)
# Returns: List[Tuple[np.ndarray, np.ndarray]]  # (joint_config, pose)
```

### Pose Clustering
```python
from genik.data.clustering import PoseClusterer

clusterer = PoseClusterer(epsilon_pos=0.001, epsilon_ori=0.017453)
clusters = clusterer.fit(samples)
# Returns: Dict[int, List[int]]  # pose_id → [sample_indices]
```

### Consistency Matching
```python
from genik.data.consistency import ConsistencyMatcher

matcher = ConsistencyMatcher(strategy="mixed")
triplets = matcher.generate(clusters, num_triplets=3000000)
# Returns: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]  # (q_ref, x_target, q_star)
```

### Dataset Creation
```python
from genik.data.dataset import GeNIKDataset

dataset = GeNIKDataset.from_triplets(triplets)
dataset.save("data/panda/train_dataset.pt")

# Loading
dataset = GeNIKDataset.load("data/panda/train_dataset.pt")
train, val, test = dataset.split([0.8, 0.1, 0.1])
```

## Testing

### Unit Tests
- Test FK sampling produces valid configurations
- Verify clustering correctness on known multi-solution cases
- Validate consistency matching selects nearest solution
- Check dataset serialization/deserialization integrity

### Integration Tests
- Generate complete dataset for Panda robot (7-DOF)
- Verify end-to-end pipeline completes without errors
- Check dataset statistics match expected distributions
- Validate performance meets benchmarks

### Edge Cases
- Handle robots with <3 DOF and >7 DOF
- Test with joint limits at [0, 2π] and non-standard ranges
- Verify behavior when ε is very small or very large
- Test with empty clusters (isolated poses)
