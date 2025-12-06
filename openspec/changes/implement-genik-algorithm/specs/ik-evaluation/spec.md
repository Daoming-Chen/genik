# Specification: IK Evaluation

**Capability ID:** `ik-evaluation`  
**Version:** 1.0.0  
**Status:** Draft

## Overview

The IK Evaluation capability provides tools for assessing the accuracy, performance, and robustness of trained GeNIK inverse kinematics models. It includes standard FK-based validation, comprehensive error metrics, and visualization utilities.

## ADDED Requirements

### Requirement: Forward Kinematics Validation

The system SHALL validate predicted joint configurations by computing their forward kinematics and comparing against target poses.

#### Scenario: FK-Based Pose Error Computation
```
GIVEN a predicted joint configuration q_pred
AND a target end-effector pose x_target
WHEN validating the prediction
THEN the system SHALL:
  - Compute actual pose: x_actual = FK(q_pred)
  - Calculate position error: e_pos = ||pos_actual - pos_target||₂
  - Calculate orientation error: e_ori = quat_distance(quat_actual, quat_target)
  - Return both errors as separate values
  - Use standard FK (not differentiable) for validation
```

#### Scenario: Batch Validation
```
GIVEN a batch of B predictions and targets
WHEN validating all predictions
THEN the system SHALL:
  - Process all B samples in parallel
  - Return arrays of shape [B] for position and orientation errors
  - Support GPU acceleration for FK computation
  - Complete validation in <100ms for B=1000 on GPU
```

#### Scenario: Joint Limit Violation Detection
```
GIVEN predicted joint angles q_pred
AND joint limits [q_min, q_max] from URDF
WHEN checking configuration validity
THEN the system SHALL:
  - For each joint i: check q_min[i] ≤ q_pred[i] ≤ q_max[i]
  - Report number and indices of violated joints
  - Calculate violation magnitude: max(0, q_pred - q_max, q_min - q_pred)
  - Flag predictions with any violations as invalid
```

### Requirement: Accuracy Metrics

The system SHALL compute comprehensive accuracy metrics following robotics industry standards.

#### Scenario: Position Error Metric
```
GIVEN position error in meters
WHEN reporting position accuracy
THEN the system SHALL:
  - Report in millimeters (mm) for readability
  - Compute statistics: mean, median, std, min, max
  - Calculate percentiles: 50th, 90th, 95th, 99th
  - Report as both absolute values and per-axis breakdown
```

#### Scenario: Orientation Error Metric
```
GIVEN orientation error in radians
WHEN reporting orientation accuracy
THEN the system SHALL:
  - Report in degrees for readability
  - Compute statistics: mean, median, std, min, max
  - Calculate percentiles: 50th, 90th, 95th, 99th
  - Use geodesic distance on SO(3) manifold
```

#### Scenario: Success Rate Calculation
```
GIVEN a set of N test samples
AND accuracy thresholds: τ_pos (mm), τ_ori (degrees)
WHEN computing success rate
THEN the system SHALL:
  - Count samples where (e_pos < τ_pos) AND (e_ori < τ_ori)
  - Compute success_rate = count / N × 100%
  - Use default thresholds: τ_pos = 1mm, τ_ori = 1°
  - Support custom thresholds via configuration
  - Report separate success rates for different threshold levels
```

#### Scenario: Multi-Threshold Analysis
```
GIVEN test results
WHEN analyzing performance across accuracy levels
THEN the system SHALL report success rates for:
  - High precision: 0.1mm, 0.1°
  - Standard: 1mm, 1°
  - Relaxed: 5mm, 5°
  - Allow user-defined threshold sets
  - Generate threshold vs. success rate curve
```

### Requirement: Performance Metrics

The system SHALL measure inference speed and computational efficiency.

#### Scenario: Inference Time Measurement
```
GIVEN a trained model and test dataset
WHEN measuring inference performance
THEN the system SHALL:
  - Warm up GPU with 10 forward passes
  - Measure time for 1000 predictions
  - Compute mean, median, std of per-sample time
  - Report separately for CPU and GPU (if available)
  - Exclude data loading time from measurement
```

#### Scenario: Throughput Measurement
```
GIVEN batched inference capability
WHEN measuring throughput
THEN the system SHALL:
  - Test batch sizes: 1, 16, 64, 256, 1024
  - Measure queries per second (QPS) for each batch size
  - Report maximum achievable throughput
  - Identify optimal batch size for efficiency
  - Measure GPU memory usage per batch size
```

#### Scenario: Model Size and Memory Footprint
```
GIVEN a trained model
WHEN analyzing resource requirements
THEN the system SHALL report:
  - Model file size (MB)
  - Parameter count (millions)
  - Training memory (GB GPU RAM)
  - Inference memory per batch (MB)
  - Estimated memory for deployment
```

### Requirement: Robustness Analysis

The system SHALL evaluate model robustness across different test scenarios and edge cases.

#### Scenario: Workspace Coverage Analysis
```
GIVEN test samples distributed across workspace
WHEN analyzing spatial performance
THEN the system SHALL:
  - Partition workspace into 3D grid cells
  - Compute average error per cell
  - Identify high-error regions
  - Report percentage of workspace with success_rate >95%
  - Generate 3D heatmap of error distribution
```

#### Scenario: Singularity Handling Test
```
GIVEN test poses near kinematic singularities
WHEN evaluating near-singular configurations
THEN the system SHALL:
  - Identify singular regions using manipulability index
  - Test samples within ε distance of singularities
  - Report success rate near singularities vs. general workspace
  - Flag if model fails gracefully or produces invalid outputs
```

#### Scenario: Multi-Solution Consistency Test
```
GIVEN a pose with multiple valid IK solutions
AND different reference configurations q_ref_1, q_ref_2, ...
WHEN querying model with same target pose
THEN the system SHALL:
  - Verify model produces different valid solutions
  - Check each solution reaches target pose (FK validation)
  - Verify solutions are consistent with reference (nearest neighbor)
  - Measure configuration distance: ||q_pred - q_ref||
```

#### Scenario: Boundary Condition Testing
```
GIVEN test poses at workspace boundaries
WHEN evaluating edge cases
THEN the system SHALL test:
  - Maximum reach positions
  - Joint limit boundaries (near q_min, q_max)
  - Extreme orientations (gimbal lock regions)
  - Report success rate: boundary vs. interior workspace
```

### Requirement: Comparative Evaluation

The system SHALL support comparison against baseline methods and previous model versions.

#### Scenario: Baseline Comparison
```
GIVEN a trained GeNIK model
AND a baseline IK solver (e.g., numerical IK, analytical IK)
WHEN comparing methods
THEN the system SHALL measure:
  - Accuracy: position error, orientation error
  - Speed: inference time, throughput
  - Robustness: success rate, failure modes
  - Generate comparison table with metrics side-by-side
```

#### Scenario: Model Version Comparison
```
GIVEN multiple model checkpoints (v1, v2, v3, ...)
WHEN comparing across versions
THEN the system SHALL:
  - Evaluate all versions on same test set
  - Plot learning curves showing improvement over versions
  - Highlight regression in any metrics
  - Support statistical significance testing (t-test, bootstrap)
```

#### Scenario: Ablation Study Support
```
GIVEN model variants with different components
WHEN analyzing component contributions
THEN the system SHALL:
  - Compare: with/without DiffFK loss, with/without consistency matching
  - Report metric deltas: Δ accuracy, Δ speed
  - Identify critical components for performance
  - Generate ablation results table
```

### Requirement: Visualization Tools

The system SHALL provide intuitive visualizations for error analysis and result presentation.

#### Scenario: Error Distribution Histograms
```
GIVEN position and orientation errors for test set
WHEN generating histograms
THEN the system SHALL:
  - Create separate histograms for position (mm) and orientation (deg)
  - Use logarithmic x-axis for wide error range
  - Mark mean, median, and threshold lines
  - Use 50 bins for smooth distribution
  - Export as high-resolution PNG/PDF
```

#### Scenario: Workspace Heatmap Visualization
```
GIVEN spatial error data
WHEN generating workspace heatmap
THEN the system SHALL:
  - Project 3D workspace onto 2D planes (XY, XZ, YZ)
  - Color-code cells by average error (green=low, red=high)
  - Overlay robot base and reachability boundary
  - Add colorbar with error scale (mm)
  - Support interactive 3D visualization (optional)
```

#### Scenario: Error vs. Metric Plots
```
GIVEN test results with metadata
WHEN analyzing error correlations
THEN the system SHALL generate plots:
  - Error vs. distance from base
  - Error vs. manipulability index (singularity proximity)
  - Error vs. reference distance ||q_pred - q_ref||
  - Use scatter plots with regression line
  - Identify outliers for investigation
```

#### Scenario: Prediction Visualization
```
GIVEN a test sample with prediction
WHEN visualizing IK solution
THEN the system SHALL:
  - Render robot in predicted configuration (3D)
  - Overlay target pose as coordinate frame
  - Show actual end-effector pose
  - Display error vectors (position and orientation)
  - Support animation for trajectories (optional)
```

### Requirement: Evaluation Reporting

The system SHALL generate comprehensive evaluation reports for documentation and publication.

#### Scenario: Summary Report Generation
```
GIVEN completed evaluation
WHEN generating summary report
THEN the system SHALL include:
  - Executive summary with key metrics
  - Accuracy statistics table
  - Performance benchmarks table
  - Robustness analysis results
  - Failure case analysis
  - Recommendations for improvement
  - Export as Markdown, PDF, or HTML
```

#### Scenario: LaTeX Table Export
```
GIVEN evaluation metrics
WHEN exporting for publication
THEN the system SHALL:
  - Generate formatted LaTeX table code
  - Include mean ± std for each metric
  - Bold best results in comparison tables
  - Format numbers with appropriate precision
  - Include caption and label for referencing
```

#### Scenario: JSON/CSV Export
```
GIVEN detailed test results
WHEN exporting raw data
THEN the system SHALL:
  - Export per-sample results as CSV
  - Export summary metrics as JSON
  - Include metadata: model version, test date, config
  - Enable post-processing with external tools
  - Compress large result files
```

## Non-Functional Requirements

### Performance
- Evaluate 10K test samples in <1 minute on GPU
- Generate all visualizations in <30 seconds
- Support streaming evaluation for large test sets
- Minimal memory overhead (<2GB for evaluation)

### Accuracy
- Position error measurement precision: 0.01mm
- Orientation error measurement precision: 0.01°
- Consistent results across multiple evaluation runs
- Numerical stability for near-zero errors

### Usability
- Single command evaluation from config file
- Clear progress indication for long evaluations
- Automatic result saving with timestamps
- Informative error messages for failures

### Extensibility
- Plugin architecture for custom metrics
- Support for new visualization types
- Configurable report templates
- Easy integration with experiment tracking (MLflow, W&B)

## Dependencies

- `torch`: Model inference
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `pandas`: Data manipulation and export
- `scipy`: Statistical analysis
- `genik.model.network`: IK network
- `genik.urdf.URDF`: FK computation

## Configuration

```yaml
# config/eval_panda.yaml
model:
  checkpoint_path: "checkpoints/panda/best_model.pt"
  device: "cuda"

data:
  test_dataset: "data/panda/test_dataset.pt"
  batch_size: 256

metrics:
  position_thresholds: [0.1, 0.5, 1.0, 5.0]  # mm
  orientation_thresholds: [0.1, 0.5, 1.0, 5.0]  # degrees
  compute_percentiles: [50, 90, 95, 99]

performance:
  measure_inference_time: true
  num_warmup: 10
  num_timing_samples: 1000
  batch_sizes: [1, 16, 64, 256, 1024]

robustness:
  workspace_grid_size: 20
  singularity_threshold: 0.01  # manipulability
  boundary_margin: 0.05  # meters

visualization:
  generate_histograms: true
  generate_heatmaps: true
  generate_comparison_plots: true
  output_dir: "results/panda"
  figure_format: "png"
  figure_dpi: 300

reporting:
  generate_summary: true
  export_latex: true
  export_csv: true
  export_json: true
```

## API

### Evaluation Script
```python
from genik.eval.evaluator import Evaluator
from genik.eval.config import EvalConfig

# Load config
config = EvalConfig.from_yaml("config/eval_panda.yaml")

# Create evaluator
evaluator = Evaluator(config)

# Run evaluation
results = evaluator.evaluate()

# Generate report
evaluator.generate_report(results, output_dir="results/panda")
```

### Custom Evaluation
```python
from genik.eval.metrics import compute_pose_error, success_rate
from genik.model.network import GeNIKNet
from genik.data.dataset import GeNIKDataset

# Load model and data
model = GeNIKNet.load("checkpoints/best_model.pt").cuda()
test_data = GeNIKDataset.load("data/test.pt")

# Evaluate
pos_errors = []
ori_errors = []

for q_ref, x_target, _ in test_data:
    q_pred = model(q_ref.cuda(), x_target.cuda())
    e_pos, e_ori = compute_pose_error(q_pred, x_target)
    pos_errors.append(e_pos)
    ori_errors.append(e_ori)

# Compute metrics
mean_pos_error = np.mean(pos_errors)  # mm
mean_ori_error = np.mean(ori_errors)  # degrees
success = success_rate(pos_errors, ori_errors, pos_thresh=1.0, ori_thresh=1.0)

print(f"Position Error: {mean_pos_error:.3f} mm")
print(f"Orientation Error: {mean_ori_error:.3f}°")
print(f"Success Rate: {success:.1f}%")
```

### Visualization
```python
from genik.eval.visualize import plot_error_histogram, plot_workspace_heatmap

# Error histograms
plot_error_histogram(pos_errors, ori_errors, save_path="results/histogram.png")

# Workspace heatmap
plot_workspace_heatmap(test_poses, pos_errors, save_path="results/heatmap.png")
```

## Testing

### Unit Tests
- Test pose error computation matches manual calculation
- Verify success rate counting is correct
- Validate percentile computation
- Check histogram generation produces valid plots

### Integration Tests
- Evaluate on small test set (100 samples)
- Generate all visualizations without errors
- Export all report formats successfully
- Compare against known baseline results

### Validation Tests
- Verify evaluation results consistent across runs
- Test with different batch sizes (same results)
- Validate FK computation against analytical solutions
- Check statistical measures (mean, median) correctness

### Performance Tests
- Benchmark evaluation time on 10K samples
- Measure memory usage during evaluation
- Test scalability with different test set sizes
- Profile bottlenecks in evaluation pipeline
