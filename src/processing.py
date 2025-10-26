"""
Pre/post-processing utilities for hair segmentation.
"""

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class ImagePreprocessor:
    """Preprocess images for SegFormer model input."""
    
    def __init__(self, input_size=(512, 512), mean=None, std=None):
        """
        Initialize preprocessor.
        
        Note: SegFormer has its own internal preprocessing via SegformerImageProcessor,
        so this class mainly handles image resizing for consistency.
        
        Args:
            input_size: Target size (height, width) - optional for SegFormer
            mean: Not used (kept for API compatibility)
            std: Not used (kept for API compatibility)
        """
        self.input_size = input_size
        print("ImagePreprocessor initialized for SegFormer")
        print("Note: SegFormer handles normalization internally")
    
    def __call__(self, image):
        """
        Preprocess image - minimal processing since SegFormer handles it.
        
        Args:
            image: PIL Image
            
        Returns:
            PIL Image (optionally resized)
        """
        # SegFormer can handle any input size
        return image


class MaskPostprocessor:
    """Post-process segmentation masks."""
    
    def __init__(self, config):
        """
        Initialize post-processor.
        
        Args:
            config: Post-processing configuration dict
        """
        self.morph_kernel_size = config.get('morph_kernel_size', 5)
        self.morph_iterations = config.get('morph_iterations', 2)
        self.apply_blur = config.get('apply_gaussian_blur', True)
        self.blur_kernel_size = config.get('gaussian_kernel_size', 5)
        self.threshold = config.get('confidence_threshold', 0.5)
    
    def process(self, mask):
        """
        Apply post-processing operations to mask.
        
        Args:
            mask: Binary mask [H, W] as numpy array
            
        Returns:
            Processed mask [H, W] as numpy array
        """
        # Ensure mask is numpy array
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        mask = mask.astype(np.uint8)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        
        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                               iterations=self.morph_iterations)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 
                               iterations=1)
        
        # Optional Gaussian blur for smoother edges
        if self.apply_blur:
            mask_float = mask.astype(np.float32)
            mask_float = cv2.GaussianBlur(
                mask_float, 
                (self.blur_kernel_size, self.blur_kernel_size), 
                0
            )
            mask = (mask_float > self.threshold).astype(np.uint8)
        
        return mask


class OutputGenerator:
    """Generate output images from segmentation results."""
    
    def create_transparent_hair(self, image, mask):
        """
        Create image with only hair visible on transparent background.
        
        Args:
            image: Original image as numpy array [H, W, 3] or PIL Image
            mask: Binary hair mask [H, W] as numpy array
            
        Returns:
            PIL Image in RGBA format with transparent background
        """
        # Convert image to numpy if it's PIL
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure mask is numpy array and binary
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)
        
        # Create RGBA image
        rgba = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
        
        # Copy RGB channels
        rgba[:, :, :3] = image
        
        # Set alpha channel from mask
        rgba[:, :, 3] = mask * 255
        
        return Image.fromarray(rgba, 'RGBA')