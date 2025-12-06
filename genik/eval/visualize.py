"""Visualization Tools for GeNIK Evaluation.

This module provides visualization utilities for analyzing
GeNIK model performance.
"""

from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_error_histogram(
    pos_errors: np.ndarray,
    ori_errors: np.ndarray,
    pos_threshold: float = 1.0,  # mm
    ori_threshold: float = 1.0,  # degrees
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot histograms of position and orientation errors.
    
    Parameters
    ----------
    pos_errors : np.ndarray
        Position errors in mm.
    ori_errors : np.ndarray
        Orientation errors in degrees.
    pos_threshold : float
        Position threshold line (mm).
    ori_threshold : float
        Orientation threshold line (degrees).
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Position error histogram
    ax = axes[0]
    ax.hist(pos_errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(pos_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold: {pos_threshold}mm')
    ax.set_xlabel('Position Error (mm)')
    ax.set_ylabel('Count')
    ax.set_title('Position Error Distribution')
    ax.legend()
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(pos_errors):.3f}mm\n'
    stats_text += f'Median: {np.median(pos_errors):.3f}mm\n'
    stats_text += f'P95: {np.percentile(pos_errors, 95):.3f}mm'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Orientation error histogram
    ax = axes[1]
    ax.hist(ori_errors, bins=50, edgecolor='black', alpha=0.7, color='darkorange')
    ax.axvline(ori_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold: {ori_threshold}°')
    ax.set_xlabel('Orientation Error (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Orientation Error Distribution')
    ax.legend()
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(ori_errors):.3f}°\n'
    stats_text += f'Median: {np.median(ori_errors):.3f}°\n'
    stats_text += f'P95: {np.percentile(ori_errors, 95):.3f}°'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_error_vs_threshold(
    pos_errors: np.ndarray,
    ori_errors: np.ndarray,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot cumulative success rate vs error threshold.
    
    Parameters
    ----------
    pos_errors : np.ndarray
        Position errors in mm.
    ori_errors : np.ndarray
        Orientation errors in degrees.
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Position threshold curve
    ax = axes[0]
    thresholds = np.linspace(0, np.percentile(pos_errors, 99), 100)
    success_rates = [100 * np.mean(pos_errors < t) for t in thresholds]
    
    ax.plot(thresholds, success_rates, linewidth=2, color='steelblue')
    ax.axhline(95, color='green', linestyle='--', alpha=0.7, label='95% success')
    ax.axhline(99, color='orange', linestyle='--', alpha=0.7, label='99% success')
    ax.set_xlabel('Position Threshold (mm)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Position Error Cumulative Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Orientation threshold curve
    ax = axes[1]
    thresholds = np.linspace(0, np.percentile(ori_errors, 99), 100)
    success_rates = [100 * np.mean(ori_errors < t) for t in thresholds]
    
    ax.plot(thresholds, success_rates, linewidth=2, color='darkorange')
    ax.axhline(95, color='green', linestyle='--', alpha=0.7, label='95% success')
    ax.axhline(99, color='orange', linestyle='--', alpha=0.7, label='99% success')
    ax.set_xlabel('Orientation Threshold (degrees)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Orientation Error Cumulative Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_workspace_heatmap(
    positions: np.ndarray,
    errors: np.ndarray,
    plane: str = 'xy',
    grid_size: int = 50,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot error heatmap across workspace.
    
    Parameters
    ----------
    positions : np.ndarray
        End-effector positions of shape (N, 3).
    errors : np.ndarray
        Errors corresponding to each position.
    plane : str
        Projection plane: 'xy', 'xz', or 'yz'.
    grid_size : int
        Number of grid cells per dimension.
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    plane_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    axis_labels = {'xy': ('X', 'Y'), 'xz': ('X', 'Z'), 'yz': ('Y', 'Z')}
    
    if plane not in plane_map:
        raise ValueError(f"Invalid plane: {plane}. Must be 'xy', 'xz', or 'yz'.")
    
    idx1, idx2 = plane_map[plane]
    x, y = positions[:, idx1], positions[:, idx2]
    xlabel, ylabel = axis_labels[plane]
    
    # Create grid
    x_bins = np.linspace(x.min(), x.max(), grid_size + 1)
    y_bins = np.linspace(y.min(), y.max(), grid_size + 1)
    
    # Compute mean error in each cell
    heatmap = np.zeros((grid_size, grid_size))
    counts = np.zeros((grid_size, grid_size))
    
    x_indices = np.digitize(x, x_bins) - 1
    y_indices = np.digitize(y, y_bins) - 1
    
    # Clip to valid range
    x_indices = np.clip(x_indices, 0, grid_size - 1)
    y_indices = np.clip(y_indices, 0, grid_size - 1)
    
    for i, (xi, yi, err) in enumerate(zip(x_indices, y_indices, errors)):
        heatmap[yi, xi] += err
        counts[yi, xi] += 1
    
    # Avoid division by zero
    mask = counts > 0
    heatmap[mask] /= counts[mask]
    heatmap[~mask] = np.nan
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
    im = ax.imshow(
        heatmap, origin='lower', extent=extent,
        cmap='RdYlGn_r', aspect='auto',
    )
    
    ax.set_xlabel(f'{xlabel} (m)')
    ax.set_ylabel(f'{ylabel} (m)')
    ax.set_title(f'Error Heatmap ({plane.upper()} Plane)')
    
    cbar = plt.colorbar(im, ax=ax, label='Mean Error')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    pos_errors: Optional[List[float]] = None,
    ori_errors: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot training curves.
    
    Parameters
    ----------
    train_losses : list
        Training losses per epoch.
    val_losses : list
        Validation losses per epoch.
    pos_errors : list, optional
        Position errors per epoch (mm).
    ori_errors : list, optional
        Orientation errors per epoch (degrees).
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    n_plots = 2 if pos_errors is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    # Loss curves
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error curves
    if pos_errors is not None and len(axes) > 1:
        ax = axes[1]
        ax.plot(epochs, pos_errors, label='Position Error (mm)', linewidth=2)
        if ori_errors is not None:
            ax2 = ax.twinx()
            ax2.plot(epochs, ori_errors, label='Orientation Error (°)', 
                    linewidth=2, color='darkorange')
            ax2.set_ylabel('Orientation Error (degrees)')
            ax2.legend(loc='upper right')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Position Error (mm)')
        ax.set_title('Validation Errors')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_inference_time(
    timing_results: Dict[int, Dict[str, float]],
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot inference time vs batch size.
    
    Parameters
    ----------
    timing_results : dict
        Results from GeNIKEvaluator.measure_inference_time().
    figsize : tuple
        Figure size.
    save_path : str or Path, optional
        Path to save figure.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    batch_sizes = sorted(timing_results.keys())
    latencies = [timing_results[bs]['mean_ms'] for bs in batch_sizes]
    throughputs = [timing_results[bs]['throughput_qps'] for bs in batch_sizes]
    
    # Latency
    ax = axes[0]
    ax.plot(batch_sizes, latencies, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Inference Latency')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Throughput
    ax = axes[1]
    ax.plot(batch_sizes, throughputs, 's-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (queries/sec)')
    ax.set_title('Inference Throughput')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
