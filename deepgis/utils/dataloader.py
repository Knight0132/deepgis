from PIL import Image
import numpy as np
import os
import torch
import torch.utils.data as data
from torchvision.io import read_image
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

class MyDataset(data.Dataset):
   """
   Custom dataset class for loading and processing multi-band images.
   
   Args:
       annotation_lines: List of annotation strings in format 'label;path' 
       input_shape: Expected input shape of images
       transform: Optional transforms to be applied to images
       expected_channels: Expected number of channels in images
       save_path: Optional path to save anomalous images
   """
   
   def __init__(self, annotation_lines, input_shape, transform=None, expected_channels=3, save_path=None):
       self.annotation_lines = annotation_lines
       self.input_shape = input_shape
       self.transform = transform
       self.expected_channels = expected_channels
       self.save_path = save_path
       if save_path and not os.path.exists(save_path):
           os.makedirs(save_path)

   def __len__(self):
       """Return the number of images in the dataset."""
       return len(self.annotation_lines)

   def __getitem__(self, index):
       """
       Get image and label at the given index.
       
       Args:
           index: Index of the image to retrieve
           
       Returns:
           tuple: (image, label) where image is a tensor and label is an integer
       """
       annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
       
       # Read multi-band image using rasterio
       with rasterio.open(annotation_path) as src:
           image = src.read()
           
       # Check channel count
       if image.shape[0] != self.expected_channels:
           print(f"Warning: Image at {annotation_path} has {image.shape[0]} channels instead of {self.expected_channels}")
           if self.save_path:
               # Save anomalous images for inspection
               save_file = os.path.join(self.save_path, f"anomaly_{index}_{image.shape[0]}ch.tif")
               with rasterio.open(save_file, 'w', driver='GTiff', 
                                height=image.shape[1], width=image.shape[2], 
                                count=image.shape[0], dtype=image.dtype) as dst:
                   dst.write(image)
               print(f"Saved anomaly image to {save_file}")
       
       # Convert to tensor and apply transforms if any
       image = torch.from_numpy(image).float()
       if self.transform is not None:
           image = self.transform(image)

       label = int(self.annotation_lines[index].split(';')[0])
       return image, label
   
   def check_all_images(self):
       """Verify all images in the dataset can be loaded correctly."""
       for i in range(len(self)):
           self.__getitem__(i)
       print("Finished checking all images.")