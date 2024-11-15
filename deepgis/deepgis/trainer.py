import os
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict
from functools import partial
from torch.utils.data import DataLoader

from .base_runner import BaseRunner
from .utils.accuracy import output_metrics
from .utils.dataloader import MyDataset
from .utils.utils import (
    get_classes, 
    my_transform, 
    store_result, 
    draw_result_visualization
)
from .train import train_epoch, valid_epoch
from .test import test_epoch

def save_weights(model: nn.Module, save_path: Path, save_state_dict_only: bool = True) -> None:
    """
    Save model weights safely.
    
    Args:
        model: Model to save
        save_path: Path to save the weights
        save_state_dict_only: If True, only save state dict instead of full model
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_to_save = model.module if hasattr(model, 'module') else model
    
    try:
        if save_state_dict_only:
            torch.save(model_to_save.state_dict(), str(save_path))
        else:
            torch.save(model_to_save, str(save_path))
    except Exception as e:
        print(f"Warning: Failed to save model to {save_path}: {e}")

class Trainer(BaseRunner):
    """
    Trainer class for managing the complete training pipeline.

    """
    
    def __init__(self, config):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration object containing training parameters
        """
        super().__init__(config)
        self.setup()
        
    def setup(self) -> None:
        """
        Initialize all components required for training.

        """
        # Create directories with timestamp
        time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.logs_folder = Path(self.config.logs_dir) / time_now
        self.checkpoints_folder = Path(self.config.checkpoints_dir) / time_now
        self.logs_folder.mkdir(parents=True, exist_ok=True)
        self.checkpoints_folder.mkdir(parents=True, exist_ok=True)
        
        # Get class information
        self.class_names, self.num_classes = get_classes(
            self.config.dataset.classes_path
        )
        
        # Setup model and components
        self.model = self._setup_model(self.num_classes)
        self._setup_data_loaders()
        self._setup_training_tools()
        
    def _setup_data_loaders(self) -> None:
        """
        Initialize data loaders for train, validation, and test sets.
        """
        image_transform = partial(my_transform, IsRandomRotation=True)
        
        # Load dataset files
        with open(self.config.dataset.train_annotation_path, encoding='utf-8') as f:
            train_lines = f.readlines()
        with open(self.config.dataset.val_annotation_path, encoding='utf-8') as f:
            val_lines = f.readlines()
        with open(self.config.dataset.test_annotation_path, encoding='utf-8') as f:
            test_lines = f.readlines()
            
        # Shuffle training data
        np.random.seed(3047)
        np.random.shuffle(train_lines)
        
        # Create datasets
        train_dataset = MyDataset(
            train_lines,
            input_shape=self.config.dataset.input_shape,
            transform=image_transform,
            expected_channels=self.config.dataset.in_channels,
            save_path=self.config.error_folder
        )
        
        val_dataset = MyDataset(
            val_lines,
            input_shape=self.config.dataset.input_shape,
            transform=image_transform,
            expected_channels=self.config.dataset.in_channels,
            save_path=self.config.error_folder
        )
        
        test_dataset = MyDataset(
            test_lines,
            input_shape=self.config.dataset.input_shape,
            transform=image_transform,
            expected_channels=self.config.dataset.in_channels,
            save_path=self.config.error_folder
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        print(
            f"Dataset sizes - Train: {len(train_lines)}, "
            f"Val: {len(val_lines)}, "
            f"Test: {len(test_lines)}"
        )
        
    def _setup_training_tools(self) -> None:
        """
        Initialize training components (criterion, optimizer, scheduler).
        """
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.train.lr,
            momentum=self.config.train.momentum,
            weight_decay=self.config.train.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.train.epoch // 2,
            gamma=0.9
        )
        
    def train(self) -> None:
        """
        Execute complete training pipeline.

        """
        print("Starting training pipeline...")
        epoch_results = np.zeros([4, self.config.train.epoch])
        
        for epoch in range(self.config.train.epoch):
            # Train one epoch
            train_acc1, train_acc3, train_loss = train_epoch(
                model=self.model,
                train_loader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                current_epoch=epoch,
                total_epochs=self.config.train.epoch,
                device=self.device
            )
            self.scheduler.step()
            
            # Record results
            epoch_results[0][epoch] = epoch + 1
            epoch_results[1][epoch] = train_loss
            epoch_results[2][epoch] = train_acc1
            epoch_results[3][epoch] = train_acc3
            
            print(
                f"Epoch: {epoch+1:03d} | Loss: {train_loss:.4f} | "
                f"Top-1: {train_acc1:.2f}% | Top-3: {train_acc3:.2f}%"
            )
            
            # Periodic validation and checkpointing
            if ((epoch + 1) % self.config.train.save_period == 0 or 
                epoch == self.config.train.epoch - 1):
                self.validate_and_save(epoch)
                
        # Finalize training
        draw_result_visualization(self.logs_folder, epoch_results)
        print("Training completed!")
        self.final_test()
        
    def validate_and_save(self, epoch: int) -> None:
        """
        Perform validation and save model checkpoint.
        
        Args:
            epoch: Current epoch number
        """
        print("Running validation...")
        val_acc1, val_acc3, val_loss, val_prediction, val_label = valid_epoch(
            model=self.model,
            valid_loader=self.val_loader,
            criterion=self.criterion,
            device=self.device
        )
        
        metrics = output_metrics(val_label, val_prediction)
        val_CM, val_weighted_recall, val_weighted_precision, val_weighted_f1 = metrics
        
        if epoch != self.config.train.epoch - 1:
            print(
                f"Validation => Top-1: {val_acc1:.2f}% | "
                f"W-Recall: {val_weighted_recall:.4f} | "
                f"W-Precision: {val_weighted_precision:.4f} | "
                f"W-F1: {val_weighted_f1:.4f}"
            )
            
        # Save checkpoints
        checkpoint_base = self.checkpoints_folder / f"model_loss{val_loss:.4f}_epoch{epoch+1}"
        
        try:
            # Save full model
            save_weights(
                model=self.model,
                save_path=checkpoint_base.with_suffix('.pth'),
                save_state_dict_only=False
            )
            
            # Save state dict only
            save_weights(
                model=self.model,
                save_path=checkpoint_base.with_name(f"{checkpoint_base.stem}_state_dict.pth"),
                save_state_dict_only=True
            )
            
            print(f"Checkpoints saved to {self.checkpoints_folder}")
        except Exception as e:
            print(f"Warning: Failed to save checkpoints: {e}")
        
    def final_test(self) -> None:
        """Perform final testing and store results."""
        print("Running final test evaluation...")
        test_acc1, test_acc3, test_prediction, test_label = test_epoch(
            model=self.model,
            test_loader=self.test_loader,
            device=self.device
        )
        
        # Calculate metrics
        metrics = output_metrics(test_label, test_prediction)
        test_CM, test_weighted_recall, test_weighted_precision, test_weighted_f1 = metrics
        
        # Print results
        print(
            f"Test Results => Top-1: {test_acc1:.2f}% | "
            f"W-Recall: {test_weighted_recall:.4f} | "
            f"W-Precision: {test_weighted_precision:.4f} | "
            f"W-F1: {test_weighted_f1:.4f}"
        )
        
        # Store results
        store_result(
            save_dir=self.logs_folder,
            accuracy=test_acc1,
            weighted_recall=test_weighted_recall,
            weighted_precision=test_weighted_precision,
            weighted_f1=test_weighted_f1,
            confusion_matrix=test_CM,
            epochs=self.config.train.epoch,
            batch_size=self.config.train.batch_size,
            learning_rate=self.config.train.lr,
            weight_decay=self.config.train.weight_decay
        )