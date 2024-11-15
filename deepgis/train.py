import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, List
from .utils.utils import AverageMeter
from .utils.accuracy import accuracy

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    current_epoch: int,
    total_epochs: int,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        current_epoch: Current epoch number
        total_epochs: Total number of epochs
        device: Device to train on
        
    Returns:
        Tuple of (top1_accuracy, top3_accuracy, average_loss)
    """
    # Initialize meters
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc3_meter = AverageMeter()
    
    # Create progress bar
    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f'Train Epoch [{current_epoch + 1}/{total_epochs}]',
        ncols=120
    )
    
    # Set model to training mode
    model.train()
    
    for batch_idx, (batch_data, batch_label) in progress_bar:
        # Move data to device
        batch_data = batch_data.to(device).float()
        batch_label = batch_label.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        batch_outputs = model(batch_data)
        loss = criterion(batch_outputs, batch_label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        batch_size = batch_data.size(0)
        batch_accuracies = accuracy(batch_outputs, batch_label, topk=(1, 3))
        
        # Update meters
        loss_meter.update(loss.item(), batch_size)
        acc1_meter.update(batch_accuracies[0], batch_size)
        acc3_meter.update(batch_accuracies[1], batch_size)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss_meter.get_average():.4f}",
            'acc1': f"{acc1_meter.get_average():.2f}%",
            'acc3': f"{acc3_meter.get_average():.2f}%"
        })
    
    return acc1_meter.get_average(), acc3_meter.get_average(), loss_meter.get_average()

def valid_epoch(
    model: nn.Module,
    valid_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Validate model performance.
    
    Args:
        model: Model to validate
        valid_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (top1_accuracy, top3_accuracy, average_loss, predictions, labels)
    """
    # Initialize meters
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc3_meter = AverageMeter()
    
    # Initialize arrays for predictions and labels
    predictions = np.array([])
    labels = np.array([])
    
    # Create progress bar
    progress_bar = tqdm(
        enumerate(valid_loader),
        total=len(valid_loader),
        desc='Validation',
        ncols=120
    )
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (batch_data, batch_label) in progress_bar:
            # Move data to device
            batch_data = batch_data.to(device).float()
            batch_label = batch_label.to(device)
            
            # Forward pass
            batch_outputs = model(batch_data)
            loss = criterion(batch_outputs, batch_label)
            
            # Calculate metrics
            batch_size = batch_data.size(0)
            batch_accuracies = accuracy(batch_outputs, batch_label, topk=(1, 3))
            batch_predictions = torch.argmax(batch_outputs, dim=1)
            
            # Update meters
            loss_meter.update(loss.item(), batch_size)
            acc1_meter.update(batch_accuracies[0], batch_size)
            acc3_meter.update(batch_accuracies[1], batch_size)
            
            # Store predictions and labels
            predictions = np.append(predictions, batch_predictions.cpu().numpy())
            labels = np.append(labels, batch_label.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_meter.get_average():.4f}",
                'acc1': f"{acc1_meter.get_average():.2f}%",
                'acc3': f"{acc3_meter.get_average():.2f}%"
            })
    
    # Return results
    return (
        acc1_meter.get_average(),
        acc3_meter.get_average(),
        loss_meter.get_average(),
        predictions,
        labels
    )