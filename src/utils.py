"""
Utility functions for hair segmentation pipeline.
"""

import os
import yaml
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_image(image_path):
    """
    Load image from file as PIL Image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image in RGB format
    """
    image = Image.open(image_path)

    image = ImageOps.exif_transpose(image)
    
    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def save_image(image, output_path):
    """
    Save image to file.
    
    Args:
        image: PIL Image or numpy array
        output_path: Path to save the image
    """
    # Get directory from output path
    output_dir = os.path.dirname(output_path)
    
    # Only create directory if it's not empty (i.e., path contains a directory)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        # Check if it's RGBA or RGB
        if image.shape[-1] == 4:
            image = Image.fromarray(image, 'RGBA')
        else:
            image = Image.fromarray(image, 'RGB')
    
    # Ensure we have a PIL Image
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image or numpy array, got {type(image)}")
    
    # Save the image
    image.save(output_path)


def ensure_dir(directory):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_image_files(directory, recursive=True):
    """
    Get all image files in a directory.
    
    Args:
        directory: Directory path
        recursive: Whether to search recursively
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(directory, file))
    
    return sorted(image_files)


def create_output_path(input_path, output_dir, suffix="", extension=None):
    """
    Create output path from input path.
    
    Args:
        input_path: Input file path
        output_dir: Output directory
        suffix: Suffix to add to filename
        extension: New extension (keep original if None)
        
    Returns:
        Output file path
    """
    input_name = Path(input_path).stem
    
    if extension is None:
        extension = Path(input_path).suffix
    
    output_name = f"{input_name}{suffix}{extension}"
    return os.path.join(output_dir, output_name)


def get_relative_path(file_path, base_dir):
    """
    Get relative path from base directory.
    
    Args:
        file_path: Full file path
        base_dir: Base directory
        
    Returns:
        Relative path
    """
    return os.path.relpath(file_path, base_dir)


def visualize_results(original, mask, output):
    """
    Visualize segmentation results.
    
    Args:
        original: Original image (PIL Image or numpy array)
        mask: Segmentation mask (numpy array)
        output: Output image (PIL Image)
    """
    import matplotlib.pyplot as plt
    
    # Convert to numpy arrays for visualization
    if isinstance(original, Image.Image):
        original = np.array(original)
    if isinstance(output, Image.Image):
        output = np.array(output)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Hair Mask')
    axes[1].axis('off')
    
    # Output with transparent background
    axes[2].imshow(output)
    axes[2].set_title('Hair Only (Transparent BG)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()