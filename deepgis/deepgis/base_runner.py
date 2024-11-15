# deepgis/base_runner.py
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
from typing import Optional

from .model.ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .model.MobileNetV3 import MobileNetV3_Small, MobileNetV3_Large

class BaseRunner:
    """Base class for all runners (trainer, predictor, etc.)."""
    
    def __init__(self, config):
        """
        Initialize base runner.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = config.get_device()
        self._setup_device()
        
    def _setup_device(self) -> None:
        """Setup computing device and optimization."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
    def _setup_model(self, num_classes: int) -> nn.Module:
        """
        Create and setup model.
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Configured model
        """
        # Create model instance
        model = self._create_model(num_classes)
        
        # Load weights if specified
        model = self._load_weights(model)
        
        # Setup device and parallel processing
        if self.config.model.cuda and torch.cuda.is_available():
            model = model.to(self.device)
            if self.config.model.dp:
                model = nn.DataParallel(model)
                cudnn.benchmark = True
                print("Using DataParallel")
                
        return model
    
    def _create_model(self, num_classes: int) -> nn.Module:
        """
        Create model instance.
        
        Args:
            num_classes: Number of output classes
            
        Returns:
            Created model instance
        """
        if self.config.model.name == "resnet18":
            model = ResNet18(
                num_classes=num_classes,
                in_channels=self.config.dataset.in_channels
            )
        elif self.config.model.name == "resnet34":
            model = ResNet34(
                num_classes=num_classes,
                in_channels=self.config.dataset.in_channels
            )
        elif self.config.model.name == "resnet50":
            model = ResNet50(
                num_classes=num_classes,
                in_channels=self.config.dataset.in_channels
            )
        elif self.config.model.name == "resnet101":
            model = ResNet101(
                num_classes=num_classes,
                in_channels=self.config.dataset.in_channels
            )
        elif self.config.model.name == "resnet152":
            model = ResNet152(
                num_classes=num_classes,
                in_channels=self.config.dataset.in_channels
            )
        elif self.config.model.name == "mobilenetv3_small":
            model = MobileNetV3_Small(
                num_classes=num_classes,
                in_channels=self.config.dataset.in_channels
            )
        elif self.config.model.name == "mobilenetv3_large":
            model = MobileNetV3_Large(
                num_classes=num_classes,
                in_channels=self.config.dataset.in_channels
            )
        else:
            raise ValueError(f"Unsupported model: {self.config.model.name}")
            
        return model
    
    def _load_weights(self, model: nn.Module) -> nn.Module:
        """
        Load model weights if specified in config.
        
        Args:
            model: Model instance to load weights into
            
        Returns:
            Model with loaded weights
        """
        # Load pretrained weights if specified
        if self.config.model.pretrained:
            weights_dir = Path(__file__).parent / 'weights'
            weights_path = weights_dir / f"{self.config.model.name}.pth"
            
            if weights_path.exists():
                try:
                    state_dict = torch.load(weights_path, map_location=self.device)
                    if isinstance(state_dict, nn.Module):
                        state_dict = state_dict.state_dict()
                    model.load_state_dict(state_dict, strict=False)
                    print(f"Loaded pretrained weights from: {weights_path}")
                except Exception as e:
                    print(f"Failed to load pretrained weights: {e}")
            else:
                print(f"Pretrained weights not found at: {weights_path}")
        
        # Load custom weights if specified
        if self.config.pretrained_model_path:
            weights_path = Path(self.config.pretrained_model_path)
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Custom weights not found: {weights_path}"
                )
            
            try:
                if weights_path.suffix == '.pth':
                    state_dict = torch.load(weights_path, map_location=self.device)
                    if isinstance(state_dict, nn.Module):
                        state_dict = state_dict.state_dict()
                    model.load_state_dict(state_dict, strict=False)
                else:
                    model.load_state_dict(
                        torch.load(weights_path, map_location=self.device)
                    )
                print(f"Loaded custom weights from: {weights_path}")
            except Exception as e:
                print(f"Failed to load custom weights: {e}")
        else:
            self._initialize_weights(model)
            
        return model
    
    def _initialize_weights(self, model: nn.Module) -> None:
        """
        Initialize model weights.
        
        Args:
            model: Model to initialize
        """
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')