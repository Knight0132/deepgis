import os
import torch
import json
from typing import Dict, Union, Optional, List, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
from enum import Enum

class ProcessMode(str, Enum):
    TRAIN = 'train'
    INFERENCE = 'inference'

@dataclass
class DatasetConfig:
    train_annotation_path: str
    val_annotation_path: str
    test_annotation_path: str
    classes_path: str
    input_shape: List[int]
    in_channels: int
    num_classes: int

    def __post_init__(self):
        if len(self.input_shape) != 2:
            raise ValueError("input_shape must be a list of 2 integers")

@dataclass
class ModelConfig:
    name: str
    pretrained: bool
    cuda: bool
    dp: bool

    def __post_init__(self):
        valid_models = ["resnet34", "resnet50", "mobilenetv3_small", "mobilenetv3_large"]
        if self.name not in valid_models:
            raise ValueError(f"Model name must be one of {valid_models}")

@dataclass
class TrainConfig:
    epoch: int
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    save_period: int

    def __post_init__(self):
        if self.epoch <= 0:
            raise ValueError("epoch must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0:
            raise ValueError("learning rate must be positive")

@dataclass
class ApplicationConfig:
    raster_image_initial_path: str
    raster_image_final_path: str
    time_period: int

class Config:
    def __init__(self):
        """Initialize configuration with default values."""
        self.version = "1.0.0"
        self._base_dir = Path.cwd()
        self.mode: ProcessMode = ProcessMode.TRAIN
        
        # Initialize paths
        self.logs_dir = 'logs'
        self.checkpoints_dir = "checkpoints"
        self.pretrained_model_path: Optional[str] = None
        self.error_folder = 'images/error_Images'

        # Initialize configurations
        self.dataset = DatasetConfig(
            train_annotation_path="images/cls_train.txt",
            val_annotation_path="images/cls_val.txt",
            test_annotation_path="images/cls_test.txt",
            classes_path='dataset/cls_classes.txt',
            input_shape=[17, 17],
            in_channels=4, 
            num_classes=5
        )

        self.model = ModelConfig(
            name="resnet34",
            pretrained=False,
            cuda=True,
            dp=False
        )

        self.train = TrainConfig(
            epoch=10,
            batch_size=64,
            lr=0.00001,
            momentum=0.9,
            weight_decay=1e-2,
            save_period=5
        )

        self.inference_image_path: Optional[str] = None
        self.output_image_path: Optional[str] = None

        self.application = ApplicationConfig(
            raster_image_initial_path='dataset/raw_data/raster_image_initial.tif',
            raster_image_final_path='dataset/predition/raster_image_final.tif',
            time_period=5
        )

    def get_device(self) -> torch.device:
        """Get the computing device for model operations."""
        if self.model.cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
        
    @property
    def device(self) -> torch.device:
        """Property for backward compatibility."""
        return self.get_device()

    def _resolve_path(self, path: str) -> Path:
        """Resolve relative path to absolute path."""
        if os.path.isabs(path):
            return Path(path)
        return self._base_dir / path

    def _validate_paths(self) -> None:
        """Validate and create necessary directories."""
        # Validate directories
        os.makedirs(self._resolve_path(self.logs_dir), exist_ok=True)
        os.makedirs(self._resolve_path(self.checkpoints_dir), exist_ok=True)
        
        # Validate required files
        required_files = [
            self.dataset.classes_path
        ]
        
        for file_path in required_files:
            path = self._resolve_path(file_path)
            if not path.exists():
                warnings.warn(f"Required file not found: {path}")

    def update(self, config_update: Union[str, Dict, None] = None) -> None:
        """Update configuration with new values."""
        if config_update is None:
            return

        if isinstance(config_update, str):
            if not os.path.exists(config_update):
                raise FileNotFoundError(f"Config file not found: {config_update}")
            
            try:
                with open(config_update, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file: {str(e)}")
        else:
            config_dict = config_update

        if not isinstance(config_dict, dict):
            raise TypeError("Configuration must be a dictionary")

        if 'mode' in config_dict:
            self.mode = ProcessMode(config_dict['mode'])

        dataclass_mapping = {
            'dataset': DatasetConfig,
            'model': ModelConfig,
            'train': TrainConfig,
            'application': ApplicationConfig
        }

        for key, value in config_dict.items():
            if key in dataclass_mapping:
                current_value = getattr(self, key)
                if isinstance(current_value, dataclass_mapping[key]):
                    current_dict = current_value.__dict__.copy()
                    current_dict.update(value)
                    setattr(self, key, dataclass_mapping[key](**current_dict))
                else:
                    setattr(self, key, dataclass_mapping[key](**value))
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Unknown configuration key '{key}' ignored")

        self._validate_paths()

    def to_json(self, save_path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            'version': self.version,
            'mode': self.mode.value,
            'logs_dir': self.logs_dir,
            'checkpoints_dir': self.checkpoints_dir,
            'pretrained_model_path': self.pretrained_model_path,
            'error_folder': self.error_folder,
            'dataset': self.dataset.__dict__,
            'model': self.model.__dict__,
            'train': self.train.__dict__,
            'inference_image_path': self.inference_image_path,
            'output_image_path': self.output_image_path,
            'application': self.application.__dict__
        }

        save_path = self._resolve_path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4)
            print(f"Configuration saved to {save_path}")
        except Exception as e:
            raise IOError(f"Failed to save configuration: {str(e)}")

    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """Create configuration from JSON file."""
        config = cls()
        config.update(json_path)
        return config

    def __str__(self) -> str:
        """Get string representation of configuration."""
        return f"Config(version={self.version}, mode={self.mode}, model={self.model.name})"

    def __repr__(self) -> str:
        return self.__str__()