#!/usr/bin/env python
"""Data Generation Script for GeNIK.

This script generates training datasets for the GeNIK inverse kinematics network.
It performs FK sampling, pose clustering, and consistency matching to create
training triplets.

Usage:
    python scripts/generate_dataset.py --urdf robots/panda_arm.urdf --output data/panda
    python scripts/generate_dataset.py --config config/data_generation.yaml
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genik.data.sampler import FKSampler
from genik.data.clustering import PoseClusterer
from genik.data.consistency import ConsistencyMatcher


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate GeNIK training dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input/Output
    parser.add_argument(
        "--urdf", type=str, required=True,
        help="Path to robot URDF file",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--end-link", type=str, default=None,
        help="Name of end-effector link (default: last link)",
    )
    
    # FK Sampling
    parser.add_argument(
        "--num-fk-samples", type=int, default=1_000_000,
        help="Number of FK samples to generate",
    )
    parser.add_argument(
        "--fk-batch-size", type=int, default=10000,
        help="Batch size for FK computation",
    )
    
    # Clustering
    parser.add_argument(
        "--epsilon-pos", type=float, default=0.001,
        help="Position epsilon for clustering (meters)",
    )
    parser.add_argument(
        "--epsilon-ori", type=float, default=0.0175,
        help="Orientation epsilon for clustering (radians)",
    )
    
    # Consistency Matching
    parser.add_argument(
        "--num-triplets", type=int, default=3_000_000,
        help="Number of training triplets to generate",
    )
    parser.add_argument(
        "--samples-per-pose", type=int, default=3,
        help="Samples per unique pose",
    )
    parser.add_argument(
        "--strategy", type=str, default="mixed",
        choices=["perturbation", "random", "mixed"],
        help="Reference sampling strategy",
    )
    parser.add_argument(
        "--perturbation-std", type=float, default=0.3,
        help="Standard deviation for perturbation noise",
    )
    
    # General
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use-cpu", action="store_true",
        help="Use CPU for FK computation (default: use GPU if available)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides command line args)",
    )
    
    return parser.parse_args()


def load_config(config_path: str, args: argparse.Namespace) -> dict:
    """Load config from YAML file and merge with command line args."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Merge with command line args (CLI takes precedence for explicitly set values)
    merged = vars(args).copy()
    for key, value in config.items():
        if key in merged and merged[key] is None:
            merged[key] = value
    
    return merged


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config if provided
    if args.config is not None:
        config = load_config(args.config, args)
    else:
        config = vars(args)
    
    # Set random seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Setup device
    device = None
    if config['use_cpu'] or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Using CPU")
    else:
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    # Create output directory
    output_dir = Path(config['output'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths for intermediate files
    fk_samples_path = output_dir / "fk_samples.pt"
    clusters_path = output_dir / "pose_clusters.pt"
    triplets_path = output_dir / "train_triplets.pt"
    
    # =========================================================================
    # Step 1: FK Sampling
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Forward Kinematics Sampling")
    print("=" * 60)
    
    if config['resume'] and fk_samples_path.exists():
        print(f"Loading existing FK samples from {fk_samples_path}")
        fk_data = FKSampler.load(fk_samples_path)
    else:
        start_time = time.time()
        
        sampler = FKSampler(
            urdf_path=config['urdf'],
            end_link=config['end_link'],
        )
        
        print(f"Robot: {sampler.robot.name}")
        print(f"End-effector: {sampler.end_link.name}")
        print(f"DOF: {sampler.n_joints}")
        print(f"Generating {config['num_fk_samples']:,} FK samples...")
        
        fk_data = sampler.sample(
            num_samples=config['num_fk_samples'],
            seed=config['seed'],
            batch_size=config['fk_batch_size'],
            use_gpu=(device.type == 'cuda'),
            device=device,
            show_progress=True,
        )
        
        # Save FK samples
        FKSampler.save(fk_data, fk_samples_path)
        
        elapsed = time.time() - start_time
        print(f"FK sampling completed in {elapsed:.1f}s")
        print(f"Saved to {fk_samples_path}")
    
    print(f"FK samples shape: {fk_data['joint_configs'].shape}")
    print(f"Poses shape: {fk_data['poses'].shape}")
    
    # =========================================================================
    # Step 2: Pose Clustering
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Pose Space Clustering")
    print("=" * 60)
    
    if config['resume'] and clusters_path.exists():
        print(f"Loading existing clusters from {clusters_path}")
        clusterer = PoseClusterer.load(clusters_path)
    else:
        start_time = time.time()
        
        print(f"Clustering with epsilon_pos={config['epsilon_pos']}m, "
              f"epsilon_ori={config['epsilon_ori']}rad")
        
        clusterer = PoseClusterer(
            epsilon_pos=config['epsilon_pos'],
            epsilon_ori=config['epsilon_ori'],
        )
        
        clusterer.fit(
            poses=fk_data['poses'],
            joint_configs=fk_data['joint_configs'],
            show_progress=True,
        )
        
        # Save clusters
        clusterer.save(clusters_path)
        
        elapsed = time.time() - start_time
        print(f"Clustering completed in {elapsed:.1f}s")
        print(f"Saved to {clusters_path}")
    
    # Print clustering statistics
    stats = clusterer.get_multi_solution_stats()
    print(f"\nClustering Statistics:")
    print(f"  Number of clusters: {stats['n_clusters']:,}")
    print(f"  Avg solutions per cluster: {stats['avg_solutions_per_cluster']:.2f}")
    print(f"  Max solutions: {stats['max_solutions']}")
    print(f"  Multi-solution percentage: {stats['multi_solution_percentage']:.1f}%")
    
    # =========================================================================
    # Step 3: Consistency Matching
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Consistency-Based Sample Generation")
    print("=" * 60)
    
    if config['resume'] and triplets_path.exists():
        print(f"Loading existing triplets from {triplets_path}")
        triplets = ConsistencyMatcher.load(triplets_path)
    else:
        start_time = time.time()
        
        # Get joint limits from FK data metadata
        joint_limits = np.array(fk_data['metadata']['joint_limits'])
        
        matcher = ConsistencyMatcher(
            clusterer=clusterer,
            joint_limits=joint_limits,
            strategy=config['strategy'],
            perturbation_std=config['perturbation_std'],
        )
        
        print(f"Generating {config['num_triplets']:,} training triplets...")
        print(f"Strategy: {config['strategy']}")
        print(f"Samples per pose: {config['samples_per_pose']}")
        
        triplets = matcher.generate(
            num_samples=config['num_triplets'],
            samples_per_pose=config['samples_per_pose'],
            seed=config['seed'],
            show_progress=True,
        )
        
        # Save triplets
        ConsistencyMatcher.save(triplets, triplets_path)
        
        elapsed = time.time() - start_time
        print(f"Triplet generation completed in {elapsed:.1f}s")
        print(f"Saved to {triplets_path}")
    
    print(f"\nTriplets generated: {len(triplets['q_ref']):,}")
    print(f"q_ref shape: {triplets['q_ref'].shape}")
    print(f"x_target shape: {triplets['x_target'].shape}")
    print(f"q_star shape: {triplets['q_star'].shape}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  FK samples: {fk_samples_path}")
    print(f"  Pose clusters: {clusters_path}")
    print(f"  Training triplets: {triplets_path}")
    
    # Calculate approximate file sizes
    if fk_samples_path.exists():
        print(f"\nFile sizes:")
        print(f"  FK samples: {fk_samples_path.stat().st_size / 1e6:.1f} MB")
        print(f"  Pose clusters: {clusters_path.stat().st_size / 1e6:.1f} MB")
        print(f"  Training triplets: {triplets_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
