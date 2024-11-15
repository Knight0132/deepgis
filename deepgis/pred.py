import torch
import numpy as np
from tqdm import tqdm
import rasterio
from typing import Tuple, Optional, Callable
from .utils.utils import my_transform

def predict_image(
    model: torch.nn.Module,
    image_path: str,
    input_shape: list,
    device: torch.device,
    patch_size: int = 17,
    batch_size: int = 1024,
    transform: Optional[Callable] = None
) -> Tuple[np.ndarray, dict, dict]:
    """
    Stable GPU-accelerated prediction function for remote sensing image classification.
    
    Args:
        model: PyTorch model for prediction
        image_path: Path to the input raster image
        input_shape: List containing [height, width] of input patches
        device: PyTorch device (CPU or GPU)
        patch_size: Size of image patches to process (default: 17)
        batch_size: Number of patches to process in each batch (default: 1024)
        transform: Optional transform to apply to input patches
        
    Returns:
        Tuple containing:
        - prediction_map: numpy array of class predictions
        - src_crs: Coordinate reference system of source image
        - src_transform: Geotransform of source image
    """
    # Load the source image with rasterio
    print("Loading image data...")
    with rasterio.open(image_path) as src:
        image = src.read()
        src_crs = src.crs
        src_transform = src.transform

    # Convert to HWC format and get dimensions
    image = np.transpose(image, (1, 2, 0))
    height, width, num_bands = image.shape
    pad_size = patch_size // 2
    
    # Pad image with reflection
    padded_image = np.pad(
        image,
        ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        mode='reflect'
    )

    # Initialize prediction map
    prediction_map = np.zeros((height, width), dtype=np.uint8)

    print(f"Starting prediction... Image size: {height}x{width}")
    model.eval()

    with torch.no_grad():
        # Process image row by row
        for i in tqdm(range(height)):
            patches = []
            positions = []
            
            # Process patches in current row
            for j in range(width):
                # Extract and transform patch
                patch = padded_image[i:i + patch_size, j:j + patch_size, :]
                patch_tensor = torch.from_numpy(patch.astype(np.float32)).permute(2, 0, 1)
                if transform is not None:
                    patch_tensor = transform(patch_tensor)
                patches.append(patch_tensor)
                positions.append((i, j))
                
                # Process batch when full or at row end
                if len(patches) == batch_size or j == width - 1:
                    # Move batch to device and predict
                    batch_patches = torch.stack(patches).to(device)
                    outputs = model(batch_patches)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    
                    # Update prediction map
                    for (x, y), pred in zip(positions, preds):
                        prediction_map[x, y] = pred
                    
                    # Clear memory
                    patches = []
                    positions = []
                    del batch_patches, outputs
                    torch.cuda.empty_cache()

    return prediction_map, src_crs, src_transform

def save_prediction(
    prediction_map: np.ndarray,
    output_path: str,
    src_crs: dict,
    src_transform: dict
) -> None:
    """
    Save prediction map as a GeoTIFF file with preserved geospatial metadata.
    
    Args:
        prediction_map: Numpy array of predictions
        output_path: Path to save the output GeoTIFF
        src_crs: Coordinate reference system from source image
        src_transform: Geotransform from source image
    """
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=prediction_map.shape[0],
        width=prediction_map.shape[1],
        count=1,
        dtype=prediction_map.dtype,
        crs=src_crs,
        transform=src_transform,
    ) as dst:
        dst.write(prediction_map, 1)