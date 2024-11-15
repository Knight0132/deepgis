import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import os
import torchvision.transforms.functional as F
from typing import Union, Tuple
from pathlib import Path

def get_classes(classes_path):
   """
   Load class names from file.
   
   Args:
       classes_path: Path to file containing class names
       
   Returns:
       Tuple of (class_names, num_classes)
   """
   with open(classes_path, encoding='utf-8') as f:
       class_names = [c.strip() for c in f.readlines()]
   return class_names, len(class_names)

class AverageMeter:
   """Computes and stores the average and current value."""
   
   def __init__(self):
       self.reset()
       
   def reset(self):
       """Reset all statistics."""
       self._value = 0
       self._sum = 0
       self._count = 0
       
   def update(self, value, n=1):
       """
       Update statistics.
       
       Args:
           value: Value to update with (can be tensor or number)
           n: Number of items this value represents
       """
       if hasattr(value, 'item'):
           value = value.item()
       
       self._value = value
       self._sum += value * n
       self._count += n
   
   def get_value(self) -> float:
       """Get current value."""
       return float(self._value)
   
   def get_average(self) -> float:
       """Get current average."""
       return float(self._sum / max(1, self._count))

def store_result(save_dir: Union[str, Path],
               accuracy: float,
               weighted_recall: float,
               weighted_precision: float,
               weighted_f1: float,
               confusion_matrix: np.ndarray,
               epochs: int,
               batch_size: int,
               learning_rate: float,
               weight_decay: float) -> None:
   """
   Store training results and metrics.
   
   Args:
       save_dir: Directory to save results
       accuracy: Model accuracy
       weighted_recall: Weighted recall score
       weighted_precision: Weighted precision score
       weighted_f1: Weighted F1 score
       confusion_matrix: Confusion matrix
       epochs: Number of training epochs
       batch_size: Training batch size
       learning_rate: Learning rate used
       weight_decay: Weight decay used
   """
   save_dir = Path(save_dir)
   save_dir.mkdir(parents=True, exist_ok=True)
   
   # Save text results
   with open(save_dir / 'test_results.txt', 'w') as f:
       f.write(f"Test Results:\n")
       f.write(f"Accuracy: {accuracy:.2f}%\n")
       f.write(f"Weighted Recall: {weighted_recall:.4f}\n")
       f.write(f"Weighted Precision: {weighted_precision:.4f}\n")
       f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
       f.write(f"Training Parameters:\n") 
       f.write(f"Epochs: {epochs}\n")
       f.write(f"Batch Size: {batch_size}\n")
       f.write(f"Learning Rate: {learning_rate}\n")
       f.write(f"Weight Decay: {weight_decay}\n")
   
   # Plot confusion matrix
   plt.figure(figsize=(10, 8))
   plt.imshow(confusion_matrix, cmap='Blues')
   plt.colorbar()
   plt.title('Confusion Matrix')
   plt.xlabel('Predicted')
   plt.ylabel('True')
   
   for i in range(confusion_matrix.shape[0]):
       for j in range(confusion_matrix.shape[1]):
           plt.text(j, i, str(confusion_matrix[i, j]),
                   ha='center', va='center')
   
   plt.savefig(save_dir / 'confusion_matrix.png')
   plt.close()
   
   print(f"Results saved to {save_dir}")

def draw_result_visualization(save_dir: Union[str, Path],
                           results: np.ndarray) -> None:
   """
   Draw training results visualization.
   
   Args:
       save_dir: Directory to save visualization
       results: Array containing epoch results [epoch_num, loss, acc1, acc3]
   """
   save_dir = Path(save_dir)
   save_dir.mkdir(parents=True, exist_ok=True)
   
   # Plot metrics
   plt.figure(figsize=(12, 8))
   fig, ax1 = plt.subplots()
   
   # Plot loss
   color = 'tab:red'
   ax1.set_xlabel('Epoch')
   ax1.set_ylabel('Loss', color=color)
   ax1.plot(results[0], results[1], color=color, label='Loss')
   ax1.tick_params(axis='y', labelcolor=color)
   
   # Plot accuracy
   ax2 = ax1.twinx()
   color = 'tab:blue'
   ax2.set_ylabel('Accuracy (%)', color=color)
   ax2.plot(results[0], results[2], color=color, label='Top-1 Acc')
   ax2.plot(results[0], results[3], color='tab:green', label='Top-3 Acc')
   ax2.tick_params(axis='y', labelcolor=color)
   
   lines1, labels1 = ax1.get_legend_handles_labels()
   lines2, labels2 = ax2.get_legend_handles_labels()
   ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
   
   plt.title('Training Results')
   plt.tight_layout()
   plt.savefig(save_dir / 'training_results.png')
   plt.close()
   
   print(f"Training visualization saved to {save_dir}")

def transform(IsResize, Resize_size, IsTotensor, IsNormalize, Norm_mean,
             Norm_std, IsRandomGrayscale, IsColorJitter, brightness, contrast,
             hue, saturation, IsCentercrop, Centercrop_size, IsRandomCrop,
             RandomCrop_size, IsRandomResizedCrop, RandomResizedCrop_size,
             Grayscale_rate, IsRandomHorizontalFlip, HorizontalFlip_rate,
             IsRandomVerticalFlip, VerticalFlip_rate, IsRandomRotation,
             degrees):
   """
   Build transformation pipeline based on configuration.
   
   Args contain various configuration flags and parameters for different transforms.
   Returns a composed transform pipeline.
   """
   transform_list = []

   # Add rotation transforms
   if IsRandomRotation:
       transform_list.append(transforms.RandomRotation(degrees))
   if IsRandomHorizontalFlip:
       transform_list.append(transforms.RandomHorizontalFlip(HorizontalFlip_rate))
   if IsRandomVerticalFlip:
       transform_list.append(transforms.RandomHorizontalFlip(VerticalFlip_rate))

   # Add color transforms  
   if IsColorJitter:
       transform_list.append(transforms.ColorJitter(brightness, contrast, saturation, hue))
   if IsRandomGrayscale:
       transform_list.append(transforms.RandomGrayscale(Grayscale_rate))

   # Add resize/crop transforms
   if IsResize:
       transform_list.append(transforms.Resize(Resize_size))
   if IsCentercrop:
       transform_list.append(transforms.CenterCrop(Centercrop_size))
   if IsRandomCrop:
       transform_list.append(transforms.RandomCrop(RandomCrop_size))
   if IsRandomResizedCrop:
       transform_list.append(transforms.RandomResizedCrop(RandomResizedCrop_size))

   # Add tensor conversion and normalization
   if IsTotensor:
       transform_list.append(transforms.ToTensor())
   if IsNormalize:
       transform_list.append(transforms.Normalize(Norm_mean, Norm_std))

   return transforms.Compose(transform_list)

def get_transform(size=[200, 200],
                mean=[0, 0, 0],
                std=[1, 1, 1],
                IsResize=False,
                IsCentercrop=False,
                IsRandomCrop=False,
                IsRandomResizedCrop=False,
                IsTotensor=False,
                IsNormalize=False,
                IsRandomGrayscale=False,
                IsColorJitter=False,
                IsRandomVerticalFlip=False,
                IsRandomHorizontalFlip=False,
                IsRandomRotation=False):
   """
   Create transform pipeline with default parameters.
   
   Args:
       size: Target image size
       mean: Normalization mean
       std: Normalization std
       Various boolean flags to enable/disable transforms
       
   Returns:
       Composed transform pipeline
   """
   return transform(
       IsResize=IsResize,
       Resize_size=size,
       IsCentercrop=IsCentercrop,
       Centercrop_size=size,
       IsRandomCrop=IsRandomCrop,
       RandomCrop_size=size,
       IsRandomResizedCrop=IsRandomResizedCrop,
       RandomResizedCrop_size=size,
       IsTotensor=IsTotensor,
       IsNormalize=IsNormalize,
       Norm_mean=mean,
       Norm_std=std,
       IsRandomGrayscale=IsRandomGrayscale,
       Grayscale_rate=0.5,
       IsColorJitter=IsColorJitter,
       brightness=0.5,
       contrast=0.5,
       hue=0.5,
       saturation=0.5,
       IsRandomVerticalFlip=IsRandomVerticalFlip,
       VerticalFlip_rate=0.5,
       IsRandomHorizontalFlip=IsRandomHorizontalFlip,
       HorizontalFlip_rate=0.5,
       IsRandomRotation=IsRandomRotation,
       degrees=10
   )

def my_transform(image, IsRandomRotation=False):
   """
   Apply custom transforms to image.
   
   Args:
       image: Input image
       IsRandomRotation: Whether to apply random rotation
       
   Returns:
       Transformed image
   """
   if IsRandomRotation:
       angle = np.random.uniform(-10, 10)
       image = F.rotate(image, angle)
   return image