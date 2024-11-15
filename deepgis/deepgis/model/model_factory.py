import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
import warnings
from typing import Optional, Dict

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
   sys.path.insert(0, project_root)

from ..config.config import Config
from .ResNet import ResNet34, ResNet50
from .MobileNetV3 import MobileNetV3_Small, MobileNetV3_Large
from ..utils.model_utils import SafeUnpickler

class ModelFactory:
   """Factory class for creating and managing models."""
   
   _models = {
       'resnet34': ResNet34,
       'resnet50': ResNet50,
       'mobilenetv3_small': MobileNetV3_Small,
       'mobilenetv3_large': MobileNetV3_Large
   }

   @classmethod
   def get_weights_dir(cls) -> Path:
       """Get directory containing pre-trained weights."""
       return Path(__file__).parent.parent / 'weights'

   @staticmethod
   def safe_load_state_dict(weights_path: Path, device: str = 'cpu') -> Optional[Dict]:
       """
       Safely load state dict from file.
       
       Args:
           weights_path: Path to weights file
           device: Device to load weights to
           
       Returns:
           State dict if successful, None otherwise
       """
       if not weights_path.exists():
           raise FileNotFoundError(f"Weights file not found: {weights_path}")

       state_dict = torch.load(str(weights_path), map_location=device)
       
       # Handle different formats of saved weights
       if isinstance(state_dict, nn.Module):
           state_dict = state_dict.state_dict()
       elif isinstance(state_dict, dict):
           if 'state_dict' in state_dict:
               state_dict = state_dict['state_dict']
           elif 'model_state_dict' in state_dict:
               state_dict = state_dict['model_state_dict']
       
       # Remove DataParallel prefix
       new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
       return new_state_dict

   @classmethod
   def create_model(cls, config: Config) -> nn.Module:
       """
       Create and configure model instance.
       
       Args:
           config: Configuration object containing model specifications
           
       Returns:
           Configured model instance
       """
       if config.model.name not in cls._models:
           raise ValueError(f"Unknown model: {config.model.name}")
       
       # Create model instance
       model_cls = cls._models[config.model.name]
       model = model_cls(
           num_classes=config.dataset.num_classes,
           in_channels=config.dataset.in_channels
       )
       
       def load_weights(weights_path: Path, weight_type: str = "pretrained") -> bool:
           """
           Load weights from file and apply to model.
           
           Args:
               weights_path: Path to weights file
               weight_type: Type of weights being loaded
               
           Returns:
               Success status of weight loading
           """
           if not weights_path.exists():
               print(f"{weight_type} weights not found at {weights_path}")
               return False
           
           print(f"Loading {weight_type} weights from {weights_path}")
           state_dict = SafeUnpickler.safe_load(weights_path)
           
           if state_dict is None:
               print(f"Failed to load {weight_type} weights")
               return False
           
           print(f"\nFound {len(state_dict)} layers in {weight_type} weights")
           print("Sample keys:", list(state_dict.keys())[:5])
           
           try:
               missing = model.load_state_dict(state_dict, strict=False)
               if missing.missing_keys or missing.unexpected_keys:
                   print(f"\nWeight loading details for {weight_type}:")
                   if missing.missing_keys:
                       print(f"Missing keys: {missing.missing_keys}")
                   if missing.unexpected_keys:
                       print(f"Unexpected keys: {missing.unexpected_keys}")
               print(f"Successfully loaded {weight_type} weights")
               return True
           except Exception as e:
               print(f"Error loading {weight_type} weights: {str(e)}")
               return False
       
       # Load pretrained weights if specified
       if config.model.pretrained:
           weights_dir = cls.get_weights_dir()
           weights_path = weights_dir / f"{config.model.name}.pth"
           load_weights(weights_path, "pretrained")
       
       # Load custom weights if specified
       if config.pretrained_model_path:
           weights_path = Path(config.pretrained_model_path)
           load_weights(weights_path, "custom")
       
       # Configure device and data parallelism
       if config.model.cuda and torch.cuda.is_available():
           model = model.cuda()
           if config.model.dp and torch.cuda.device_count() > 1:
               model = nn.DataParallel(model)
       
       return model

def load_model(config: Config) -> nn.Module:
   """
   Convenience function to create model instance.
   
   Args:
       config: Configuration object
   
   Returns:
       Configured model instance
   """
   return ModelFactory.create_model(config)