from typing import Tuple, Optional
import numpy as np
import rasterio
from functools import partial

from .base_runner import BaseRunner
from .pred import predict_image, save_prediction
from .utils.utils import get_classes, my_transform

class Predictor(BaseRunner):
    """
    Predictor class for remote sensing image classification.
    
    Inherits from BaseRunner for common model and device setup functionality.
    """
    
    def __init__(self, config):
        """
        Initialize the predictor with configuration.
        
        Args:
            config: Configuration object containing model and prediction parameters
        """
        super().__init__(config)
        self.setup()
        
    def setup(self) -> None:
        """
        Initialize components required for prediction.

        """
        self.class_names, self.num_classes = get_classes(
            self.config.dataset.classes_path
        )
        self.model = self._setup_model(self.num_classes)
        
    def predict(self) -> Tuple[np.ndarray, dict, dict]:
        """
        Execute prediction on the input image.
        
        Returns:
            Tuple containing:
            - prediction_map: numpy array of class predictions
            - src_crs: Coordinate reference system of source image
            - src_transform: Geotransform of source image
            
        Notes:
            - Input image path is taken from config.inference_image_path
            - Output path (if saving) is taken from config.output_image_path
        """
        # Setup image transformation
        image_transform = partial(my_transform, IsRandomRotation=True)
        
        # Run prediction
        prediction_map, src_crs, src_transform = predict_image(
            model=self.model,
            image_path=self.config.inference_image_path,
            input_shape=self.config.dataset.input_shape,
            device=self.device,
            patch_size=self.config.dataset.input_shape[0],
            batch_size=self.config.train.batch_size,
            transform=image_transform
        )
        
        # Save prediction if output path is specified
        if self.config.output_image_path:
            save_prediction(
                prediction_map=prediction_map,
                output_path=self.config.output_image_path,
                src_crs=src_crs,
                src_transform=src_transform
            )
            print(f"Prediction saved to: {self.config.output_image_path}")
            
        return prediction_map, src_crs, src_transform
    
    @property
    def class_mapping(self) -> dict:
        """
        Get mapping between class indices and class names.
        
        Returns:
            Dictionary mapping class indices to class names
        """
        return {i: name for i, name in enumerate(self.class_names)}