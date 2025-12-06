# GeNIK Data Processing Module
"""
This module provides data preprocessing utilities for the GeNIK inverse kinematics algorithm.

Modules:
    - sampler: FK sampling for generating joint-pose pairs
    - clustering: Pose space clustering for multi-solution identification
    - consistency: Consistency-based training sample generation
    - dataset: PyTorch Dataset classes for training
"""

from .sampler import FKSampler
from .clustering import PoseClusterer
from .consistency import ConsistencyMatcher
from .dataset import GeNIKDataset

__all__ = [
    'FKSampler',
    'PoseClusterer',
    'ConsistencyMatcher',
    'GeNIKDataset',
]
