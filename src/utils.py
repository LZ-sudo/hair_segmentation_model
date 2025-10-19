"""
Utility functions for file I/O and visualization.
"""

import os
import cv2
import yaml
import matplotlib.pyplot as plt
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_image(image_path):
    """
    Load image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image in BGR format (OpenCV default)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def save_image(image, output_path):
    """
    Save image to file.
    
    Args:
        image: Image to save
        output_path: Output file path
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    success = cv2.imwrite(output_path, image)
    if not success:
        raise IOError(f"Failed to save image: {output_path}")


def get_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp'), recursive=False):
    """
    Get all image files in a directory.

    Args:
        directory: Path to directory
        extensions: Tuple of valid image extensions
        recursive: If True, search subdirectories recursively

    Returns:
        List of image file paths
    """
    image_files = set()  # Use set to avoid duplicates
    pattern_prefix = "**/" if recursive else ""

    for ext in extensions:
        # Add both lowercase and uppercase versions
        image_files.update(Path(directory).glob(f"{pattern_prefix}*{ext}"))
        image_files.update(Path(directory).glob(f"{pattern_prefix}*{ext.upper()}"))

    return sorted([str(f) for f in image_files])


def get_relative_path(file_path, base_dir):
    """
    Get relative path of a file from a base directory.

    Args:
        file_path: Path to file
        base_dir: Base directory path

    Returns:
        Relative path string
    """
    return str(Path(file_path).relative_to(Path(base_dir)))


def create_output_path(input_path, output_dir, suffix="", extension=None):
    """
    Create output path based on input path.
    
    Args:
        input_path: Input file path
        output_dir: Output directory
        suffix: Suffix to add to filename
        extension: Output extension (default: same as input)
        
    Returns:
        Output file path
    """
    input_path = Path(input_path)
    filename = input_path.stem + suffix
    
    if extension is None:
        extension = input_path.suffix
    else:
        extension = extension if extension.startswith('.') else f'.{extension}'
    
    output_path = Path(output_dir) / (filename + extension)
    return str(output_path)


def visualize_results(image, mask, output=None, figsize=(15, 5)):
    """
    Visualize segmentation results.
    
    Args:
        image: Original image (BGR)
        mask: Binary mask (0-255)
        output: Output image (BGR), optional
        figsize: Figure size tuple
    """
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    num_plots = 3 if output is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    # Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Hair Mask')
    axes[1].axis('off')
    
    # Output
    if output is not None:
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        axes[2].imshow(output_rgb)
        axes[2].set_title('Segmented Hair')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_comparison(images_list, titles_list, figsize=(20, 5)):
    """
    Visualize multiple images side by side.
    
    Args:
        images_list: List of images to display
        titles_list: List of titles for each image
        figsize: Figure size tuple
    """
    num_images = len(images_list)
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    if num_images == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images_list, titles_list):
        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def print_progress(current, total, prefix='Progress:', suffix='Complete'):
    """
    Print progress bar.
    
    Args:
        current: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
    """
    bar_length = 50
    filled_length = int(bar_length * current / total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = f"{100 * (current / total):.1f}"
    
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    
    if current == total:
        print()


def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)


def get_file_size(file_path):
    """
    Get file size in MB.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb