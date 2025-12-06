# GeNIK Model Module
"""
This module provides neural network architectures and utilities for the GeNIK
inverse kinematics solver.

Modules:
    - diff_fk: Differentiable forward kinematics layer
    - network: IK network architectures
    - loss: Mixed physics loss functions
"""

from .diff_fk import DifferentiableFKLayer
from .network import IKNetwork
from .loss import GeNIKLoss

__all__ = [
    'DifferentiableFKLayer',
    'IKNetwork',
    'GeNIKLoss',
]
