# Project Context

## Purpose
MFIK (Multiple Forward Inverse Kinematics) is a research project implementing GeNIK (Generative Network for Inverse Kinematics), a data-driven approach to solve the inverse kinematics problem for robotic manipulators. The goal is to achieve high-precision, single-step IK solutions while handling multi-solution scenarios through consistency-based supervision and differentiable physics constraints.

## Tech Stack
- **Python 3.8+**: Primary development language
- **PyTorch 2.0+**: Neural network implementation and automatic differentiation
- **NumPy**: Numerical operations and array manipulation
- **URDF**: Robot model representation (using custom parser)
- **SciPy**: KD-Tree for efficient spatial indexing
- **Matplotlib**: Visualization and plotting
- **TensorBoard**: Training monitoring and metrics visualization

## Project Conventions

### Code Style
- Follow PEP 8 style guide for Python code
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use descriptive variable names (e.g., `joint_angles` not `ja`)
- Docstrings: Google style format with Args, Returns, Raises sections
- Format code with `black` before committing

### Architecture Patterns
- Modular design: separate concerns into data, model, training, evaluation
- Configuration-driven: use YAML files for hyperparameters
- Factory pattern for model and dataset creation
- Plugin architecture for extensibility (custom joint types, metrics)
- Dependency injection for testing

### Testing Strategy
- Unit tests for all core algorithms (FK, clustering, loss computation)
- Integration tests for end-to-end pipeline
- Gradient validation tests using finite differences
- Performance benchmarks for critical paths
- Test coverage target: >80% for core modules
- Use `pytest` as test framework

### Git Workflow
- Main branch: `main` (protected)
- Feature branches: `feature/<description>`
- Commit messages: conventional commits format
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation
  - `test:` for tests
  - `refactor:` for refactoring

## Domain Context

### Robotics and Kinematics
- **Forward Kinematics (FK)**: Compute end-effector pose from joint angles
- **Inverse Kinematics (IK)**: Compute joint angles from target end-effector pose
- **Multi-solution problem**: Same pose can be reached with different joint configurations (e.g., elbow-up vs. elbow-down)
- **Singularities**: Configurations where robot loses degrees of freedom
- **Joint limits**: Physical constraints on joint ranges

### Neural Network Approach
- **Mode averaging**: Problem where direct regression outputs invalid average of multiple solutions
- **Consistency matching**: Strategy to select ground truth based on reference configuration
- **Differentiable FK**: Physics-informed loss to refine predictions beyond data discretization
- **Residual learning**: Network predicts correction to reference configuration

## Important Constraints
- **Computational efficiency**: Target <10ms inference time for real-time control
- **Memory constraints**: Training must fit in 8GB GPU memory
- **Accuracy requirements**: Position error <1mm, orientation error <1° for 95% of test cases
- **Reproducibility**: All experiments must be reproducible with fixed random seeds
- **URDF compatibility**: Support standard URDF format for robot models

## External Dependencies
- **Robot models**: URDF files for Panda, UR10, and other manipulators
- **trimesh** (optional): For mesh loading and visualization
- **PyTorch ecosystem**: torchvision, tensorboard, onnx for model export
- **Scientific Python**: scipy, pandas for data analysis
- **Plotting**: matplotlib, seaborn for visualization

## Current Status
- Existing URDF parser implementation: `genik/urdf.py`
- Test robot models available: `robots/panda_arm.urdf`, `robots/ur10.urdf`
- Active change proposal: `implement-genik-algorithm` (validated)
- Next steps: Implementation of data preprocessing pipeline
