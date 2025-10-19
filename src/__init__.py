"""
Hair segmentation package.
"""

from .model import HairSegmentationModel
from .processing import ImagePreprocessor, MaskPostprocessor, OutputGenerator
from .utils import load_config, load_image, save_image

__version__ = "1.0.0"

__all__ = [
    'HairSegmentationModel',
    'ImagePreprocessor',
    'MaskPostprocessor',
    'OutputGenerator',
    'load_config',
    'load_image',
    'save_image'
]