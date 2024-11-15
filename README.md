# deepgis
The repository is a python library for land use change analysis using deep learning model.

Deepgis is a deep learning library for geospatial image classification, built on PyTorch. It provides flexible tools for training and deploying deep learning models on multi-band satellite/aerial imagery.

## Features

- Support for multi-band geospatial image data (e.g. satellite imagery)
- Multiple model architectures (ResNet, MobileNetV3)
- Flexible data augmentation pipeline
- Easy-to-use training and inference APIs
- Built-in support for common metrics and visualizations
- Pre-trained models for common remote sensing tasks

## Installation
Install from source:

```bash
git clone https://github.com/yourusername/deepgis.git
cd deepgis
pip install -e .
```

## Quick Start
### Training a Model

```python
from deepgis import Config
from deepgis import load_model
from deepgis import Trainer

# Load configuration
config = Config()
config.update({
    'model': {
        'name': 'resnet34',
        'pretrained': True
    },
    'dataset': {
        'num_classes': 5,
        'in_channels': 4
    }
})

# Create model
model = load_model(config)

# Train model
trainer = Trainer(config)
trainer.train()
```

### Using Pre-trained Models

```python
from deepgis import Config
from deepgis import load_model

# Load configuration
config = Config()
config.update({
    'model': {
        'name': 'resnet34',
        'pretrained': True
    }
})

# Load pre-trained model
model = load_model(config)
```

## Data Format
The library expects data in the following format:
- Images: Multi-band GeoTIFF files
- Annotations: Text file with format class_id;image_path

Example annotation file:

```txt
0;/path/to/image1.tif
1;/path/to/image2.tif
```

## Configuration
Configuration is managed through a central Config class. The default configuration is provided in [default_json](https://github.com/Knight0132/deepgis/blob/main/deepgis/deepgis/config/default_config.json) and Key configuration options include:

```python
config = Config()
config.update({
    'model': {
        'name': 'resnet34',        # Model architecture
        'pretrained': True,        # Use pretrained weights in our project. If you don't want to use the pretrained weight provided in our project, you should input False 
        'cuda': True,             # Use GPU
        'dp': False               # Use DataParallel
    },
    'dataset': {
        'input_shape': [17, 17],  # Input image size
        'in_channels': 4,         # Number of input channels
        'num_classes': 5          # Number of classes
    },
    'train': {
        'epoch': 10,              # Number of training epochs
        'batch_size': 64,         # Batch size
        'lr': 0.00001,           # Learning rate
        'momentum': 0.9,         # SGD momentum
        'weight_decay': 1e-2     # Weight decay
    }
})
```

## Model Architectures
Currently supported architectures:
- ResNet18
- ResNet34
- ResNet50
- ResNet101
- ResNet152
- MobileNetV3-Small
- MobileNetV3-Large

But only supported pretrained architectures:
- ResNet34
- ResNet50
- MobileNetV3-Small
- MobileNetV3-Large

## License
This project is licensed under the MIT License - see the LICENSE file for details.


