#!/usr/bin/env python
"""Evaluation Script for GeNIK.

This script evaluates trained GeNIK models on test data.

Usage:
    python scripts/evaluate_genik.py --checkpoint outputs/genik/best_model.pt --data data/panda/train_triplets.pt
    python scripts/evaluate_genik.py --checkpoint outputs/genik/best_model.pt --data data/panda/train_triplets.pt --output results/
"""

import argparse
import sys
from pathlib import Path
import logging
import json

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genik.model.network import IKNetwork
from genik.model.diff_fk import DifferentiableFKLayer
from genik.data.dataset import GeNIKDataset, create_data_splits
from genik.eval.metrics import GeNIKEvaluator
from genik.eval.visualize import (
    plot_error_histogram,
    plot_error_vs_threshold,
    plot_workspace_heatmap,
    plot_inference_time,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate GeNIK model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to test data (.pt file)",
    )
    parser.add_argument(
        "--urdf", type=str, default=None,
        help="Path to URDF (defaults to checkpoint config)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--measure-timing", action="store_true",
        help="Measure inference timing",
    )
    
    return parser.parse_args()


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
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, checkpoint = IKNetwork.load_checkpoint(args.checkpoint, device=device)
    
    # Get URDF path
    urdf_path = args.urdf
    if urdf_path is None and 'config' in checkpoint:
        urdf_path = checkpoint['config'].get('urdf_path')
    
    if urdf_path is None:
        logger.error("URDF path not found. Please specify with --urdf")
        sys.exit(1)
    
    logger.info(f"Loading URDF: {urdf_path}")
    
    # Create FK layer
    fk_layer = DifferentiableFKLayer(urdf_path)
    
    # Load test data
    logger.info(f"Loading test data: {args.data}")
    dataset = GeNIKDataset(args.data)
    
    # Create test split (use last 10%)
    _, _, test_dataset = create_data_splits(
        dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    logger.info(f"Test samples: {len(test_dataset):,}")
    
    # Create evaluator
    evaluator = GeNIKEvaluator(model, fk_layer, device=device)
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate(test_loader, show_progress=True)
    
    # Print report
    report = evaluator.generate_report(results)
    print(report)
    
    # Measure timing if requested
    if args.measure_timing:
        logger.info("Measuring inference timing...")
        timing_results = evaluator.measure_inference_time()
        
        print("\nInference Timing:")
        print("-" * 40)
        for batch_size, timing in timing_results.items():
            print(f"Batch {batch_size:4d}: {timing['mean_ms']:.2f}ms "
                  f"({timing['throughput_qps']:.0f} QPS)")
    
    # Check joint limits
    logger.info("Checking joint limits...")
    limit_results = evaluator.check_joint_limits(test_loader)
    print(f"\nJoint Limit Violations: {limit_results['violation_rate']:.1f}%")
    if limit_results['max_violation_rad'] > 0:
        print(f"Max violation: {np.degrees(limit_results['max_violation_rad']):.2f}°")
    
    # Save results
    if args.output is not None:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save report
        with open(output_dir / "report.txt", "w") as f:
            f.write(report)
        
        # Save metrics as JSON
        metrics = {
            'n_samples': results['n_samples'],
            'position_error_mm': results['position_error_mm'],
            'orientation_error_deg': results['orientation_error_deg'],
            'success_rates': results['success_rates'],
        }
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Generate plots
        if not args.no_plots:
            logger.info("Generating plots...")
            
            pos_errors_mm = results['raw_errors']['position'] * 1000
            ori_errors_deg = np.degrees(results['raw_errors']['orientation'])
            
            # Error histograms
            plot_error_histogram(
                pos_errors_mm, ori_errors_deg,
                save_path=output_dir / "error_histogram.png"
            )
            
            # Threshold curves
            plot_error_vs_threshold(
                pos_errors_mm, ori_errors_deg,
                save_path=output_dir / "threshold_curves.png"
            )
            
            # Timing plot
            if args.measure_timing:
                plot_inference_time(
                    timing_results,
                    save_path=output_dir / "inference_timing.png"
                )
        
        logger.info(f"Results saved to: {output_dir}")
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
