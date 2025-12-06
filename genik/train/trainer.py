"""Training Loop for GeNIK.

This module provides the main training loop and utilities for training
the GeNIK inverse kinematics network.
"""

from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import TrainingConfig
from ..model.network import IKNetwork, create_model
from ..model.loss import GeNIKLoss
from ..model.diff_fk import DifferentiableFKLayer
from ..data.dataset import GeNIKDataset, create_data_splits, create_dataloaders


logger = logging.getLogger(__name__)


class LRSchedulerWithWarmup:
    """Learning rate scheduler with linear warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_name: str,
        num_epochs: int,
        warmup_epochs: int = 5,
        warmup_start_lr: float = 1e-5,
        min_lr: float = 1e-6,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
        # Create main scheduler
        if scheduler_name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=min_lr,
            )
        elif scheduler_name == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1),
            )
        elif scheduler_name == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=kwargs.get('gamma', 0.1),
                patience=kwargs.get('patience', 10),
            )
        else:
            self.scheduler = None
        
        self.current_epoch = 0
    
    def step(self, val_loss: Optional[float] = None):
        """Update learning rate."""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            progress = self.current_epoch / self.warmup_epochs
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_loss is not None:
                    self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
    
    def get_last_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    """GeNIK Training Manager.
    
    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
        
    Examples
    --------
    >>> config = TrainingConfig.load("config.yaml")
    >>> trainer = Trainer(config)
    >>> trainer.train()
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Setup output directory
        self.output_dir = config.get_experiment_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Save config
        config.save(self.output_dir / 'config.yaml')
        
        # Initialize components
        self.fk_layer = self._create_fk_layer()
        self.model = self._create_model()
        self.criterion = self._create_criterion()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.dataloaders = self._create_dataloaders()
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.global_step = 0
    
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.output_dir / 'training.log'
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        # Add to logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    def _create_fk_layer(self) -> DifferentiableFKLayer:
        """Create differentiable FK layer."""
        fk_layer = DifferentiableFKLayer(
            urdf_path=self.config.urdf_path,
            end_link=self.config.end_link if self.config.end_link else None,
        )
        return fk_layer.to(self.device)
    
    def _create_model(self) -> nn.Module:
        """Create IK network model."""
        model = create_model(
            n_joints=self.fk_layer.n_joints,
            model_type=self.config.model.model_type,
            hidden_size=self.config.model.hidden_size,
            num_blocks=self.config.model.num_blocks,
            dropout=self.config.model.dropout,
            use_batch_norm=self.config.model.use_batch_norm,
            use_residual_output=self.config.model.use_residual_output,
        )
        
        model = model.to(self.device)
        
        # Log model info
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {param_count:,}")
        
        return model
    
    def _create_criterion(self) -> GeNIKLoss:
        """Create loss function."""
        return GeNIKLoss(
            fk_layer=self.fk_layer,
            lambda_min=self.config.loss.lambda_min,
            lambda_max=self.config.loss.lambda_max,
            warmup_epochs=self.config.loss.lambda_warmup_epochs,
            joint_loss_type=self.config.loss.joint_loss_type,
            w_pos=self.config.loss.w_pos,
            w_ori=self.config.loss.w_ori,
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        opt_config = self.config.optimizer
        
        if opt_config.name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas,
            )
        elif opt_config.name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas,
            )
        elif opt_config.name == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                weight_decay=opt_config.weight_decay,
                momentum=opt_config.momentum,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.name}")
        
        return optimizer
    
    def _create_scheduler(self) -> LRSchedulerWithWarmup:
        """Create learning rate scheduler."""
        sched_config = self.config.scheduler
        
        return LRSchedulerWithWarmup(
            optimizer=self.optimizer,
            scheduler_name=sched_config.name,
            num_epochs=self.config.num_epochs,
            warmup_epochs=sched_config.warmup_epochs,
            warmup_start_lr=sched_config.warmup_start_lr,
            min_lr=sched_config.min_lr,
            step_size=sched_config.step_size,
            gamma=sched_config.gamma,
            patience=sched_config.patience,
        )
    
    def _create_dataloaders(self) -> Dict[str, DataLoader]:
        """Create data loaders."""
        data_config = self.config.data
        
        # Load dataset
        dataset = GeNIKDataset(
            data_path=data_config.train_path,
            normalize_pose=data_config.normalize_pose,
        )
        
        logger.info(f"Dataset size: {len(dataset):,}")
        
        # Create splits
        train_dataset, val_dataset, test_dataset = create_data_splits(
            dataset,
            train_ratio=data_config.train_ratio,
            val_ratio=data_config.val_ratio,
            test_ratio=1 - data_config.train_ratio - data_config.val_ratio,
            seed=self.config.seed,
        )
        
        logger.info(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")
        
        # Create loaders
        return create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_joint_loss = 0.0
        total_pose_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.dataloaders['train'],
            desc=f"Epoch {self.current_epoch}",
            leave=False,
        )
        
        for batch_idx, (q_ref, x_target, q_star) in enumerate(pbar):
            # Move to device
            q_ref = q_ref.to(self.device)
            x_target = x_target.to(self.device)
            q_star = q_star.to(self.device)
            
            # Forward pass
            q_pred = self.model(q_ref, x_target)
            
            # Compute loss
            loss, metrics = self.criterion(
                q_pred, q_star, x_target,
                epoch=self.current_epoch,
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_joint_loss += metrics['joint_loss'].item()
            total_pose_loss += metrics['pose_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'joint': metrics['joint_loss'].item(),
                'pose': metrics['pose_loss'].item(),
            })
            
            # Log to TensorBoard
            if self.global_step % 100 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/joint_loss', metrics['joint_loss'].item(), self.global_step)
                self.writer.add_scalar('train/pose_loss', metrics['pose_loss'].item(), self.global_step)
                self.writer.add_scalar('train/lambda', metrics['lambda'].item(), self.global_step)
            
            self.global_step += 1
        
        return {
            'loss': total_loss / num_batches,
            'joint_loss': total_joint_loss / num_batches,
            'pose_loss': total_pose_loss / num_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0.0
        total_joint_loss = 0.0
        total_pose_loss = 0.0
        total_pos_error = 0.0
        total_ori_error = 0.0
        num_batches = 0
        
        for q_ref, x_target, q_star in self.dataloaders['val']:
            # Move to device
            q_ref = q_ref.to(self.device)
            x_target = x_target.to(self.device)
            q_star = q_star.to(self.device)
            
            # Forward pass
            q_pred = self.model(q_ref, x_target)
            
            # Compute loss
            loss, metrics = self.criterion(
                q_pred, q_star, x_target,
                epoch=self.current_epoch,
            )
            
            # Accumulate metrics
            total_loss += loss.item()
            total_joint_loss += metrics['joint_loss'].item()
            total_pose_loss += metrics['pose_loss'].item()
            total_pos_error += metrics['pos_error'].item()
            total_ori_error += metrics['ori_error'].item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'joint_loss': total_joint_loss / num_batches,
            'pose_loss': total_pose_loss / num_batches,
            'pos_error': total_pos_error / num_batches,
            'ori_error': total_ori_error / num_batches,
        }
    
    def save_checkpoint(self, name: str = 'checkpoint'):
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / f'{name}.pt'
        
        self.model.save_checkpoint(
            path=checkpoint_path,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            metrics={'best_val_loss': self.best_val_loss},
            config=self.config.to_dict(),
        )
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint for resume training."""
        model, checkpoint = IKNetwork.load_checkpoint(path, device=self.device)
        self.model = model
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']
        
        if 'metrics' in checkpoint and 'best_val_loss' in checkpoint['metrics']:
            self.best_val_loss = checkpoint['metrics']['best_val_loss']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Run full training loop."""
        logger.info("Starting training...")
        logger.info(f"Output directory: {self.output_dir}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log training metrics
            logger.info(
                f"Epoch {epoch}: "
                f"loss={train_metrics['loss']:.4f}, "
                f"joint={train_metrics['joint_loss']:.4f}, "
                f"pose={train_metrics['pose_loss']:.4f}, "
                f"lr={self.scheduler.get_last_lr():.2e}"
            )
            
            # Validation
            if (epoch + 1) % self.config.val_every_n_epochs == 0:
                val_metrics = self.validate()
                
                # Log validation metrics
                logger.info(
                    f"Val: loss={val_metrics['loss']:.4f}, "
                    f"pos_error={val_metrics['pos_error']*1000:.2f}mm, "
                    f"ori_error={np.degrees(val_metrics['ori_error']):.2f}°"
                )
                
                # TensorBoard
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('val/pos_error_mm', val_metrics['pos_error']*1000, epoch)
                self.writer.add_scalar('val/ori_error_deg', np.degrees(val_metrics['ori_error']), epoch)
                
                # Best model tracking
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model')
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                
                # Early stopping
                if (self.config.early_stopping_patience > 0 and 
                    self.early_stop_counter >= self.config.early_stopping_patience):
                    logger.info("Early stopping triggered")
                    break
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}')
            
            # Log learning rate
            self.writer.add_scalar('train/lr', self.scheduler.get_last_lr(), epoch)
        
        # Final save
        self.save_checkpoint('final_model')
        
        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed/3600:.1f} hours")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        self.writer.close()
