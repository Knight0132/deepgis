import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Union, Any
import warnings
import pickle

class SafeUnpickler:
    """Custom unpickler for safely loading model weights."""
    
    @staticmethod
    def safe_load(file_path: Union[str, Path], map_location: str = 'cpu') -> Optional[Dict[str, torch.Tensor]]:
        """
        Safely load PyTorch weights with fallback options.
        
        Args:
            file_path: Path to weights file
            map_location: Device to map tensors to
            
        Returns:
            State dictionary if successful, None otherwise
        """
        file_path = Path(file_path)
        
        def try_load_weights():
            """Attempt to load weights and handle different formats"""
            state_dict = torch.load(str(file_path), map_location=map_location)
            
            if isinstance(state_dict, nn.Module):
                return state_dict.state_dict()
            elif isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    return state_dict['state_dict']
                elif 'model_state_dict' in state_dict:
                    return state_dict['model_state_dict']
                return state_dict
            return None

        def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """
            Clean state dict keys by removing common prefixes.
            
            Args:
                state_dict: Original state dictionary
                
            Returns:
                Cleaned state dictionary
            """
            new_state_dict = {}
            prefixes_to_remove = ['module.', 'model.', 'backbone.', 'encoder.']
            
            for key, value in state_dict.items():
                new_key = key
                for prefix in prefixes_to_remove:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                new_state_dict[new_key] = value
            return new_state_dict

        try:
            state_dict = try_load_weights()
            if state_dict is not None:
                return clean_state_dict(state_dict)
            print("Failed to load weights: Invalid format")
            return None
        except Exception as e:
            print(f"Failed to load weights: {str(e)}")
            return None

def save_weights(
    model: nn.Module,
    save_path: Union[str, Path],
    save_metadata: bool = False
) -> None:
    """
    Save model weights in a standardized format.
    
    Args:
        model: Model to save
        save_path: Path to save weights
        save_metadata: Whether to save additional metadata
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get state dictionary
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    # Clean state dictionary
    cleaned_dict = {k.replace('module.', '').replace('model.', ''): v 
                   for k, v in state_dict.items()}
    
    save_dict = {
        'state_dict': cleaned_dict,
        'model_type': model.__class__.__name__,
        'format_version': '1.0'
    } if save_metadata else cleaned_dict
    
    try:
        torch.save(save_dict, save_path)
        print(f"Successfully saved weights to {save_path}")
    except Exception as e:
        warnings.warn(f"Failed to save weights: {str(e)}")

def convert_weights(
    source_path: Union[str, Path],
    target_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Convert weights to a standard format.
    
    Args:
        source_path: Source weights file
        target_path: Target path to save converted weights
    """
    source_path = Path(source_path)
    target_path = target_path or source_path.with_stem(f"{source_path.stem}_converted")
    target_path = Path(target_path)
    
    state_dict = SafeUnpickler.safe_load(source_path)
    if state_dict is None:
        raise RuntimeError("Failed to load source weights")
    
    try:
        torch.save(state_dict, target_path)
        print(f"Successfully converted weights to {target_path}")
    except Exception as e:
        print(f"Error converting weights: {str(e)}")