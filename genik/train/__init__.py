# GeNIK Training Module
"""
This module provides training utilities for the GeNIK inverse kinematics solver.

Modules:
    - config: Training configuration and hyperparameters
    - trainer: Training loop and utilities
"""

from .config import TrainingConfig
from .trainer import Trainer

__all__ = [
    'TrainingConfig',
    'Trainer',
]
