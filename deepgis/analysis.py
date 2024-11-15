import numpy as np
import matplotlib.pyplot as plt
import rasterio
from PIL import Image
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class ApplicationAnalyzer:
    """
    Analyzer class for land use change and impact analysis.
    
    """
    
    def __init__(self, config):
        """
        Initialize analyzer with configuration.
        
        Args:
            config: Configuration object containing application parameters
        """
        self.raster_image_initial_path = Path(config.application.raster_image_initial_path)
        self.raster_image_final_path = Path(config.application.raster_image_final_path)
        self.time_period = config.application.time_period

    def calculate_pixel_percentages(self) -> Dict[int, float]:
        """
        Calculate area percentages for each land use class.
        
        Returns:
            Dictionary mapping class values to their percentage coverage
        """
        # Load and process image
        img = np.array(Image.open(self.raster_image_final_path))
        total_pixels = img.size
        
        # Calculate percentages for each unique value
        percentages = {
            int(value): round((np.sum(img == value) / total_pixels) * 100, 2)
            for value in sorted(np.unique(img))
        }
        
        return percentages

    def linear_regression_analysis(
        self, 
        x: List[List[float]], 
        y: List[float], 
        title: str
    ) -> None:
        """
        Perform linear regression to analyze land type impacts on carbon emissions.
        
        Args:
            x: List of lists containing land type proportions for each year
               Format: [[bare_land, built_up, vegetation, water_body], ...]
            y: List of carbon emission values for each year
            title: Title for the plot
        """
        if len(x) != len(y):
            raise ValueError('Input dimensions mismatch: features vs target')

        # Perform regression
        model = LinearRegression()
        model.fit(np.array(x), np.array(y))
        weights = model.coef_

        # Normalize weights for visualization
        min_nonzero_weight = np.min(np.abs(weights[np.nonzero(weights)]))
        scale_factor = 10**int(np.floor(-np.log10(min_nonzero_weight)))
        scaled_weights = weights * scale_factor

        # Create visualization
        land_types = ['Bare_Land', 'Built-up_Area', 'Vegetation', 'Water_Body']
        plt.figure(figsize=(10, 6))
        plt.bar(land_types, scaled_weights, color='skyblue')
        plt.xlabel("Land Use Type")
        plt.ylabel("Impact Weight")
        plt.title(f"Impact Weights of Land Types on {title}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def calculate_land_use_dynamic_index(self) -> Dict[int, Optional[float]]:
        """
        Calculate dynamic index for each land use type.
        
        Dynamic Index Formula:
        K = ((Ui - Uj) / Ui) * (1 / T) * 100
        where:
        - Ui: Initial area
        - Uj: Final area
        - T: Time period
        
        Returns:
            Dictionary mapping land use types to their dynamic indices
        """
        initial_areas, final_areas = self._load_and_count_areas()
        
        dynamic_indices = {}
        for land_type, initial_area in initial_areas.items():
            if land_type in final_areas:
                final_area = final_areas[land_type]
                dynamic_indices[land_type] = (
                    ((initial_area - final_area) / initial_area) * 
                    (1 / self.time_period) * 100
                )
            else:
                dynamic_indices[land_type] = None
                
        return dynamic_indices

    def calculate_comprehensive_dynamic_index(self) -> float:
        """
        Calculate comprehensive land use dynamic index.
        
        Comprehensive Index Formula:
        L = (ΣΔ / Σu) * (1 / T) * 100
        where:
        - ΣΔ: Sum of absolute area changes
        - Σu: Sum of initial areas
        - T: Time period
        
        Returns:
            Comprehensive dynamic index value
        """
        initial_areas, final_areas = self._load_and_count_areas()
        
        sum_changes = sum(
            abs(initial_areas[t] - final_areas.get(t, 0))
            for t in initial_areas
        )
        sum_initial = sum(initial_areas.values())
        
        return (sum_changes / sum_initial) * (1 / self.time_period) * 100

    def _load_and_count_areas(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Load and count areas from initial and final raster images.
        
        Returns:
            Tuple containing:
            - Dictionary of initial areas
            - Dictionary of final areas
        """
        with rasterio.open(self.raster_image_initial_path) as src_initial:
            initial_data = src_initial.read(1)
        with rasterio.open(self.raster_image_final_path) as src_final:
            final_data = src_final.read(1)
            
        initial_unique, initial_counts = np.unique(initial_data, return_counts=True)
        final_unique, final_counts = np.unique(final_data, return_counts=True)
        
        return (
            dict(zip(initial_unique, initial_counts)),
            dict(zip(final_unique, final_counts))
        )