#!/usr/bin/env python
"""Training Script for GeNIK.

This script trains the GeNIK inverse kinematics network.

Usage:
    python scripts/train_genik.py --urdf robots/panda_arm.urdf --data data/panda/train_triplets.pt
    python scripts/train_genik.py --config config/training.yaml
    python scripts/train_genik.py --config config/training.yaml --resume outputs/genik/checkpoint.pt
"""

import argparse
import sys
from pathlib import Path
import logging

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genik.train.config import TrainingConfig
from genik.train.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GeNIK inverse kinematics network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Configuration
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file",
    )
    
    # Basic options (override config)
    parser.add_argument(
        "--urdf", type=str, default=None,
        help="Path to robot URDF file",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to training data (.pt file)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Experiment name",
    )
    
    # Training options
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate",
    )
    
    # Model options
    parser.add_argument(
        "--hidden-size", type=int, default=None,
        help="Hidden layer size",
    )
    parser.add_argument(
        "--num-blocks", type=int, default=None,
        help="Number of residual blocks",
    )
    
    # System options
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda', 'cpu', or 'auto'",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> TrainingConfig:
    """Create training configuration from arguments."""
    
    # Start with default or loaded config
    if args.config is not None:
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig()
    
    # Override with command line arguments
    if args.urdf is not None:
        config.urdf_path = args.urdf
    if args.data is not None:
        config.data.train_path = args.data
    if args.output is not None:
        config.output_dir = args.output
    if args.name is not None:
        config.experiment_name = args.name
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.optimizer.learning_rate = args.lr
    if args.hidden_size is not None:
        config.model.hidden_size = args.hidden_size
    if args.num_blocks is not None:
        config.model.num_blocks = args.num_blocks
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    
    # Validate required fields
    if not config.urdf_path:
        raise ValueError("URDF path is required. Use --urdf or config file.")
    if not config.data.train_path:
        raise ValueError("Training data path is required. Use --data or config file.")
    
    return config


def setup_logging():
    """Setup console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create config
    try:
        config = create_config(args)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Print configuration summary
    logger.info("=" * 60)
    logger.info("GeNIK Training")
    logger.info("=" * 60)
    logger.info(f"URDF: {config.urdf_path}")
    logger.info(f"Data: {config.data.train_path}")
    logger.info(f"Output: {config.get_experiment_dir()}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Batch size: {config.data.batch_size}")
    logger.info(f"Learning rate: {config.optimizer.learning_rate}")
    logger.info(f"Model: {config.model.hidden_size}x{config.model.num_blocks} blocks")
    logger.info("=" * 60)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint('interrupted_checkpoint')
    except Exception as e:
        logger.exception("Training failed with error")
        raise


if __name__ == "__main__":
    main()
