"""
Image preprocessing and postprocessing utilities.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms


class ImagePreprocessor:
    """Handles image preprocessing for model input."""
    
    def __init__(self, input_size=(512, 512), mean=None, std=None):
        """
        Initialize preprocessor.
        
        Args:
            input_size: Tuple of (height, width) for model input
            mean: RGB mean values for normalization
            std: RGB std values for normalization
        """
        self.input_size = input_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def preprocess(self, image):
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (BGR format from cv2)
            
        Returns:
            tensor: Preprocessed image tensor [1, 3, H, W]
            original_size: Tuple of (height, width) of original image
        """
        original_size = image.shape[:2]
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(image_rgb, (self.input_size[1], self.input_size[0]))
        
        # Apply transforms (normalize and convert to tensor)
        tensor = self.transform(resized)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor, original_size


class MaskPostprocessor:
    """Handles mask refinement and postprocessing."""
    
    def __init__(self, config):
        """
        Initialize postprocessor.
        
        Args:
            config: Dictionary with postprocessing parameters
        """
        self.kernel_size = config.get('morph_kernel_size', 5)
        self.iterations = config.get('morph_iterations', 2)
        self.apply_blur = config.get('apply_gaussian_blur', True)
        self.blur_kernel = config.get('gaussian_kernel_size', 5)
        self.threshold = config.get('confidence_threshold', 0.5)
    
    def refine_mask(self, mask):
        """
        Refine binary mask using morphological operations.
        
        Args:
            mask: Binary mask [H, W] with values 0 or 1
            
        Returns:
            Refined binary mask
        """
        # Convert to uint8 if needed
        mask = (mask * 255).astype(np.uint8)
        
        # Morphological opening (remove noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (self.kernel_size, self.kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 
                               iterations=self.iterations)
        
        # Morphological closing (fill holes)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                               iterations=self.iterations)
        
        # Optional Gaussian blur for smooth edges
        if self.apply_blur:
            mask = cv2.GaussianBlur(mask, (self.blur_kernel, self.blur_kernel), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def resize_mask(self, mask, target_size):
        """
        Resize mask to target size.
        
        Args:
            mask: Input mask
            target_size: Tuple of (height, width)
            
        Returns:
            Resized mask
        """
        return cv2.resize(mask, (target_size[1], target_size[0]), 
                         interpolation=cv2.INTER_LINEAR)


class OutputGenerator:
    """Generates hair-only output with transparent background."""
    
    def __init__(self):
        """Initialize output generator."""
        pass
    
    def create_hair_only(self, image, mask):
        """
        Create image with only hair visible on transparent background.
        
        Args:
            image: Original image (BGR)
            mask: Binary mask (0-255)
            
        Returns:
            Hair-only image with alpha channel (BGRA)
        """
        # Create BGRA image
        result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = mask  # Set alpha channel to mask
        
        return result
    
    def save_output(self, image, mask, output_path):
        """
        Save hair-only output to file.
        
        Args:
            image: Original image (BGR)
            mask: Binary mask (0-255)
            output_path: Path to save output PNG file
            
        Returns:
            Path to saved file
        """
        output = self.create_hair_only(image, mask)
        cv2.imwrite(output_path, output)
        return output_path