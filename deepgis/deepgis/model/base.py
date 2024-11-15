import torch
import torch.nn as nn
from pathlib import Path
from ..config.config import Config

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    @classmethod
    def get_weights_dir(cls) -> Path:
        """get the directory to store the model weights"""
        return Path(__file__).parent.parent / 'checkpoints'
    
    @classmethod
    def get_pretrained_path(cls, model_name: str) -> Path:
        """get the path to the pretrained model weights"""
        weights_dir = cls.get_weights_dir()
        return weights_dir / f"{model_name}_model.pth"