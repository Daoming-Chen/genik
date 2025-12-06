# GeNIK Evaluation Module
"""
This module provides evaluation utilities for the GeNIK inverse kinematics solver.

Modules:
    - metrics: Evaluation metrics computation
    - visualize: Visualization utilities
"""

from .metrics import GeNIKEvaluator, compute_success_rate
from .visualize import plot_error_histogram, plot_workspace_heatmap

__all__ = [
    'GeNIKEvaluator',
    'compute_success_rate',
    'plot_error_histogram',
    'plot_workspace_heatmap',
]
