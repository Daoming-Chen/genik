"""Tests for data generation pipeline."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def panda_urdf_path():
    """Path to Panda robot URDF."""
    urdf_path = Path(__file__).parent.parent / "robots" / "panda_arm.urdf"
    if not urdf_path.exists():
        pytest.skip(f"URDF not found: {urdf_path}")
    return str(urdf_path)


@pytest.fixture
def sample_triplets(panda_urdf_path):
    """Generate sample triplets for testing."""
    from genik.data.sampler import FKSampler
    from genik.data.clustering import PoseClusterer
    from genik.data.consistency import ConsistencyMatcher
    
    # Sample FK data
    sampler = FKSampler(panda_urdf_path)
    joints, poses = sampler.sample(1000)
    
    # Cluster
    clusterer = PoseClusterer(epsilon=0.05)
    clusters = clusterer.cluster(poses)
    
    # Generate triplets
    matcher = ConsistencyMatcher()
    triplets = matcher.generate_triplets(joints, poses, clusters)
    
    return triplets


class TestFKSampler:
    """Tests for FK sampling."""
    
    def test_sampler_init(self, panda_urdf_path):
        """Test sampler initialization."""
        from genik.data.sampler import FKSampler
        
        sampler = FKSampler(panda_urdf_path)
        
        assert sampler.n_joints > 0
        assert sampler.joint_lower.shape == (sampler.n_joints,)
        assert sampler.joint_upper.shape == (sampler.n_joints,)
    
    def test_sample_shapes(self, panda_urdf_path):
        """Test sample output shapes."""
        from genik.data.sampler import FKSampler
        
        sampler = FKSampler(panda_urdf_path)
        joints, poses = sampler.sample(100)
        
        assert joints.shape == (100, sampler.n_joints)
        assert poses.shape == (100, 7)
    
    def test_samples_within_limits(self, panda_urdf_path):
        """Test samples are within joint limits."""
        from genik.data.sampler import FKSampler
        
        sampler = FKSampler(panda_urdf_path)
        joints, poses = sampler.sample(500)
        
        # Check joint limits
        assert (joints >= sampler.joint_lower - 1e-6).all()
        assert (joints <= sampler.joint_upper + 1e-6).all()
    
    def test_sample_batch(self, panda_urdf_path):
        """Test batch sampling."""
        from genik.data.sampler import FKSampler
        
        sampler = FKSampler(panda_urdf_path)
        joints, poses = sampler.sample(1000, batch_size=256)
        
        assert joints.shape[0] == 1000
        assert poses.shape[0] == 1000
    
    def test_poses_normalized_quaternions(self, panda_urdf_path):
        """Test pose quaternions are normalized."""
        from genik.data.sampler import FKSampler
        
        sampler = FKSampler(panda_urdf_path)
        joints, poses = sampler.sample(100)
        
        # Quaternion is last 4 elements
        quaternions = poses[:, 3:]
        norms = np.linalg.norm(quaternions, axis=1)
        
        assert np.allclose(norms, 1.0, atol=1e-5)
    
    def test_validate_samples(self, panda_urdf_path):
        """Test sample validation."""
        from genik.data.sampler import FKSampler
        
        sampler = FKSampler(panda_urdf_path)
        joints, poses = sampler.sample(100)
        
        # Should not raise
        sampler.validate_samples(joints, poses)


class TestPoseClusterer:
    """Tests for pose clustering."""
    
    def test_clusterer_init(self):
        """Test clusterer initialization."""
        from genik.data.clustering import PoseClusterer
        
        clusterer = PoseClusterer(epsilon=0.1)
        assert clusterer.epsilon == 0.1
    
    def test_cluster_output(self, panda_urdf_path):
        """Test clustering output format."""
        from genik.data.sampler import FKSampler
        from genik.data.clustering import PoseClusterer
        
        sampler = FKSampler(panda_urdf_path)
        joints, poses = sampler.sample(500)
        
        clusterer = PoseClusterer(epsilon=0.1)
        clusters = clusterer.cluster(poses)
        
        # Each cluster should be a list of indices
        assert isinstance(clusters, list)
        
        # All indices should be valid
        all_indices = set()
        for cluster in clusters:
            for idx in cluster:
                assert 0 <= idx < 500
                all_indices.add(idx)
    
    def test_find_neighbors(self, panda_urdf_path):
        """Test neighbor finding."""
        from genik.data.sampler import FKSampler
        from genik.data.clustering import PoseClusterer
        
        sampler = FKSampler(panda_urdf_path)
        joints, poses = sampler.sample(500)
        
        clusterer = PoseClusterer(epsilon=0.1)
        clusterer.build_index(poses)
        
        # Find neighbors for first pose
        neighbors = clusterer.find_neighbors(poses[0])
        
        # Should at least find itself
        assert len(neighbors) >= 1
        assert 0 in neighbors
    
    def test_epsilon_affects_clustering(self, panda_urdf_path):
        """Test that epsilon affects cluster sizes."""
        from genik.data.sampler import FKSampler
        from genik.data.clustering import PoseClusterer
        
        sampler = FKSampler(panda_urdf_path)
        joints, poses = sampler.sample(500)
        
        # Small epsilon = smaller clusters
        clusterer_small = PoseClusterer(epsilon=0.01)
        clusters_small = clusterer_small.cluster(poses)
        
        # Large epsilon = larger clusters
        clusterer_large = PoseClusterer(epsilon=0.5)
        clusters_large = clusterer_large.cluster(poses)
        
        # More clusters with smaller epsilon
        assert len(clusters_small) >= len(clusters_large)


class TestConsistencyMatcher:
    """Tests for consistency matching."""
    
    def test_matcher_init(self):
        """Test matcher initialization."""
        from genik.data.consistency import ConsistencyMatcher
        
        matcher = ConsistencyMatcher()
        assert matcher is not None
    
    def test_triplet_structure(self, panda_urdf_path):
        """Test triplet output structure."""
        from genik.data.sampler import FKSampler
        from genik.data.clustering import PoseClusterer
        from genik.data.consistency import ConsistencyMatcher
        
        sampler = FKSampler(panda_urdf_path)
        joints, poses = sampler.sample(500)
        
        clusterer = PoseClusterer(epsilon=0.1)
        clusters = clusterer.cluster(poses)
        
        matcher = ConsistencyMatcher()
        triplets = matcher.generate_triplets(joints, poses, clusters)
        
        # Check triplet structure
        assert 'q_i' in triplets  # Query joints
        assert 'p_i' in triplets  # Query poses
        assert 'q_c' in triplets  # Candidate joints
        assert 'p_c' in triplets  # Candidate poses
    
    def test_triplet_shapes_match(self, panda_urdf_path):
        """Test triplet arrays have matching shapes."""
        from genik.data.sampler import FKSampler
        from genik.data.clustering import PoseClusterer
        from genik.data.consistency import ConsistencyMatcher
        
        sampler = FKSampler(panda_urdf_path)
        joints, poses = sampler.sample(500)
        
        clusterer = PoseClusterer(epsilon=0.1)
        clusters = clusterer.cluster(poses)
        
        matcher = ConsistencyMatcher()
        triplets = matcher.generate_triplets(joints, poses, clusters)
        
        n_triplets = len(triplets['q_i'])
        assert len(triplets['p_i']) == n_triplets
        assert len(triplets['q_c']) == n_triplets
        assert len(triplets['p_c']) == n_triplets


class TestGeNIKDataset:
    """Tests for PyTorch dataset."""
    
    def test_dataset_init(self, sample_triplets, tmp_path):
        """Test dataset initialization."""
        from genik.data.dataset import GeNIKDataset
        
        # Save triplets to file
        torch.save(sample_triplets, tmp_path / "triplets.pt")
        
        dataset = GeNIKDataset(tmp_path / "triplets.pt")
        assert len(dataset) > 0
    
    def test_dataset_getitem(self, sample_triplets, tmp_path):
        """Test dataset __getitem__."""
        from genik.data.dataset import GeNIKDataset
        
        torch.save(sample_triplets, tmp_path / "triplets.pt")
        
        dataset = GeNIKDataset(tmp_path / "triplets.pt")
        item = dataset[0]
        
        # Check item structure
        assert 'q_i' in item
        assert 'p_i' in item
        assert 'q_c' in item
        assert 'p_c' in item
    
    def test_data_splits(self, sample_triplets, tmp_path):
        """Test data split creation."""
        from genik.data.dataset import GeNIKDataset, create_data_splits
        
        torch.save(sample_triplets, tmp_path / "triplets.pt")
        
        dataset = GeNIKDataset(tmp_path / "triplets.pt")
        train, val, test = create_data_splits(
            dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )
        
        total = len(train) + len(val) + len(test)
        assert total == len(dataset)
    
    def test_dataloader_compatible(self, sample_triplets, tmp_path):
        """Test dataset works with DataLoader."""
        from genik.data.dataset import GeNIKDataset
        from torch.utils.data import DataLoader
        
        torch.save(sample_triplets, tmp_path / "triplets.pt")
        
        dataset = GeNIKDataset(tmp_path / "triplets.pt")
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Should be able to iterate
        batch = next(iter(loader))
        
        assert batch['q_i'].shape[0] <= 32
        assert batch['p_i'].shape[0] <= 32
