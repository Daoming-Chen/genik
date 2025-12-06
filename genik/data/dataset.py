"""PyTorch Dataset for GeNIK Training.

This module provides PyTorch Dataset classes for loading and managing
training data for the GeNIK inverse kinematics network.
"""

from typing import Optional, Tuple, Dict, Any, Union, List
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class GeNIKDataset(Dataset):
    """PyTorch Dataset for GeNIK training triplets.
    
    This dataset loads and manages (q_ref, x_target, q*) training triplets
    with support for train/val/test splitting and data augmentation.
    
    Parameters
    ----------
    data_path : str or Path, optional
        Path to the .pt file containing training data.
    q_ref : np.ndarray, optional
        Reference configurations of shape (N, n_joints).
    x_target : np.ndarray, optional
        Target poses of shape (N, 7).
    q_star : np.ndarray, optional
        Ground truth solutions of shape (N, n_joints).
    transform : callable, optional
        Optional transform to apply to samples.
    normalize_pose : bool
        Whether to normalize pose positions.
    pose_mean : np.ndarray, optional
        Mean for pose normalization.
    pose_std : np.ndarray, optional
        Std for pose normalization.
        
    Notes
    -----
    Either provide data_path to load from file, or provide q_ref, x_target,
    and q_star arrays directly.
    
    Examples
    --------
    >>> dataset = GeNIKDataset("data/panda/train_triplets.pt")
    >>> q_ref, x_target, q_star = dataset[0]
    >>> print(q_ref.shape, x_target.shape, q_star.shape)
    
    >>> train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    """
    
    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        q_ref: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        q_star: Optional[np.ndarray] = None,
        transform: Optional[callable] = None,
        normalize_pose: bool = False,
        pose_mean: Optional[np.ndarray] = None,
        pose_std: Optional[np.ndarray] = None,
    ):
        self.transform = transform
        self.normalize_pose = normalize_pose
        
        if data_path is not None:
            self._load_from_file(data_path)
        elif q_ref is not None and x_target is not None and q_star is not None:
            self.q_ref = torch.from_numpy(q_ref.astype(np.float32))
            self.x_target = torch.from_numpy(x_target.astype(np.float32))
            self.q_star = torch.from_numpy(q_star.astype(np.float32))
            self.metadata = {}
        else:
            raise ValueError(
                "Either provide data_path or all of q_ref, x_target, q_star"
            )
        
        # Setup normalization
        if normalize_pose:
            if pose_mean is not None and pose_std is not None:
                self.pose_mean = torch.from_numpy(pose_mean.astype(np.float32))
                self.pose_std = torch.from_numpy(pose_std.astype(np.float32))
            else:
                # Compute from data (only for position, not quaternion)
                self.pose_mean = torch.zeros(7)
                self.pose_std = torch.ones(7)
                self.pose_mean[:3] = self.x_target[:, :3].mean(dim=0)
                self.pose_std[:3] = self.x_target[:, :3].std(dim=0) + 1e-8
        else:
            self.pose_mean = None
            self.pose_std = None
    
    def _load_from_file(self, data_path: Union[str, Path]):
        """Load data from a .pt file."""
        data = torch.load(data_path, weights_only=False)
        
        self.q_ref = data['q_ref']
        self.x_target = data['x_target']
        self.q_star = data['q_star']
        self.metadata = data.get('metadata', {})
        
        # Ensure float32
        if self.q_ref.dtype != torch.float32:
            self.q_ref = self.q_ref.float()
        if self.x_target.dtype != torch.float32:
            self.x_target = self.x_target.float()
        if self.q_star.dtype != torch.float32:
            self.q_star = self.q_star.float()
    
    def __len__(self) -> int:
        return len(self.q_ref)
    
    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a training sample.
        
        Parameters
        ----------
        idx : int
            Sample index.
            
        Returns
        -------
        tuple
            (q_ref, x_target, q_star) tensors.
        """
        q_ref = self.q_ref[idx]
        x_target = self.x_target[idx].clone()
        q_star = self.q_star[idx]
        
        # Apply pose normalization
        if self.normalize_pose and self.pose_mean is not None:
            x_target = (x_target - self.pose_mean) / self.pose_std
        
        # Apply transform
        if self.transform is not None:
            q_ref, x_target, q_star = self.transform(q_ref, x_target, q_star)
        
        return q_ref, x_target, q_star
    
    def get_normalization_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get pose normalization parameters.
        
        Returns
        -------
        tuple
            (pose_mean, pose_std) tensors.
        """
        if self.pose_mean is None:
            # Compute from data
            pose_mean = torch.zeros(7)
            pose_std = torch.ones(7)
            pose_mean[:3] = self.x_target[:, :3].mean(dim=0)
            pose_std[:3] = self.x_target[:, :3].std(dim=0) + 1e-8
            return pose_mean, pose_std
        return self.pose_mean, self.pose_std
    
    @property
    def n_joints(self) -> int:
        """Number of robot joints."""
        return self.q_ref.shape[1]
    
    @property
    def pose_dim(self) -> int:
        """Dimension of pose vector."""
        return self.x_target.shape[1]


def create_data_splits(
    dataset: GeNIKDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[Subset, Subset, Subset]:
    """Create train/val/test splits from a dataset.
    
    Parameters
    ----------
    dataset : GeNIKDataset
        The full dataset.
    train_ratio : float
        Fraction of data for training.
    val_ratio : float
        Fraction of data for validation.
    test_ratio : float
        Fraction of data for testing.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    tuple
        (train_dataset, val_dataset, test_dataset) Subset objects.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    n_samples = len(dataset)
    
    # Generate shuffled indices
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    
    # Calculate split points
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)
    
    # Create subsets
    train_indices = indices[:train_end].tolist()
    val_indices = indices[train_end:val_end].tolist()
    test_indices = indices[val_end:].tolist()
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for training, validation, and testing.
    
    Parameters
    ----------
    train_dataset : Dataset
        Training dataset.
    val_dataset : Dataset
        Validation dataset.
    test_dataset : Dataset, optional
        Test dataset.
    batch_size : int
        Batch size for all loaders.
    num_workers : int
        Number of workers for data loading.
    pin_memory : bool
        Whether to pin memory for GPU transfer.
        
    Returns
    -------
    dict
        Dictionary with 'train', 'val', and optionally 'test' DataLoaders.
    """
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    
    if test_dataset is not None:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    return loaders


class MemoryMappedDataset(Dataset):
    """Memory-mapped dataset for large-scale data.
    
    This dataset uses memory-mapping to load samples on-demand,
    suitable for datasets larger than available RAM.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing the memory-mapped data files.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
    ):
        self.data_dir = Path(data_dir)
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.n_samples = self.metadata['n_samples']
        self.n_joints = self.metadata['n_joints']
        
        # Memory-map the data arrays
        self.q_ref = np.memmap(
            self.data_dir / "q_ref.npy",
            dtype=np.float32,
            mode='r',
            shape=(self.n_samples, self.n_joints),
        )
        self.x_target = np.memmap(
            self.data_dir / "x_target.npy",
            dtype=np.float32,
            mode='r',
            shape=(self.n_samples, 7),
        )
        self.q_star = np.memmap(
            self.data_dir / "q_star.npy",
            dtype=np.float32,
            mode='r',
            shape=(self.n_samples, self.n_joints),
        )
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a training sample."""
        return (
            torch.from_numpy(self.q_ref[idx].copy()),
            torch.from_numpy(self.x_target[idx].copy()),
            torch.from_numpy(self.q_star[idx].copy()),
        )
    
    @staticmethod
    def save_memory_mapped(
        q_ref: np.ndarray,
        x_target: np.ndarray,
        q_star: np.ndarray,
        output_dir: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Save data as memory-mapped files.
        
        Parameters
        ----------
        q_ref : np.ndarray
            Reference configurations.
        x_target : np.ndarray
            Target poses.
        q_star : np.ndarray
            Ground truth solutions.
        output_dir : str or Path
            Output directory.
        metadata : dict, optional
            Additional metadata.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        n_samples = len(q_ref)
        n_joints = q_ref.shape[1]
        
        # Save as memory-mapped files
        q_ref_mm = np.memmap(
            output_dir / "q_ref.npy",
            dtype=np.float32,
            mode='w+',
            shape=(n_samples, n_joints),
        )
        q_ref_mm[:] = q_ref
        q_ref_mm.flush()
        
        x_target_mm = np.memmap(
            output_dir / "x_target.npy",
            dtype=np.float32,
            mode='w+',
            shape=(n_samples, 7),
        )
        x_target_mm[:] = x_target
        x_target_mm.flush()
        
        q_star_mm = np.memmap(
            output_dir / "q_star.npy",
            dtype=np.float32,
            mode='w+',
            shape=(n_samples, n_joints),
        )
        q_star_mm[:] = q_star
        q_star_mm.flush()
        
        # Save metadata
        meta = {
            'n_samples': n_samples,
            'n_joints': n_joints,
        }
        if metadata is not None:
            meta.update(metadata)
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)
