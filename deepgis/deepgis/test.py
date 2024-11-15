import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, List
from .utils.utils import AverageMeter
from .utils.accuracy import accuracy

def test_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Tuple containing:
        - top1_accuracy: Top-1 accuracy percentage
        - top3_accuracy: Top-3 accuracy percentage
        - predictions: Array of model predictions
        - labels: Array of ground truth labels
    """
    # Initialize meters
    acc1_meter = AverageMeter()
    acc3_meter = AverageMeter()
    
    # Initialize lists for predictions and labels
    predictions = []
    labels = []
    
    # Create progress bar
    progress_bar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc='Test',
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
            
            # Calculate accuracies
            batch_size = batch_data.size(0)
            batch_accuracies = accuracy(batch_outputs, batch_label, topk=(1, 3))
            batch_predictions = torch.argmax(batch_outputs, dim=1)
            
            # Update meters
            acc1_meter.update(batch_accuracies[0], batch_size)
            acc3_meter.update(batch_accuracies[1], batch_size)
            
            # Store predictions and labels
            predictions.extend(batch_predictions.cpu().numpy())
            labels.extend(batch_label.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'top1': f"{acc1_meter.get_average():.2f}%",
                'top3': f"{acc3_meter.get_average():.2f}%"
            })
    
    # Convert lists to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    return (
        acc1_meter.get_average(),
        acc3_meter.get_average(),
        predictions,
        labels
    )