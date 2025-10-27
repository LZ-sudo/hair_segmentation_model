"""
Main hair segmentation pipeline.
"""

import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.model import HairSegmentationModel
from src.processing import ImagePreprocessor, MaskPostprocessor, OutputGenerator
from src.utils import (
    load_config, load_image, save_image, get_image_files,
    create_output_path, visualize_results, ensure_dir, get_relative_path
)


class HairSegmentationPipeline:
    """Complete pipeline for hair segmentation using SegFormer."""
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize the hair segmentation pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        print("Initializing SegFormer hair segmentation pipeline...")
        
        # Model
        model_config = self.config['model']
        self.model = HairSegmentationModel(
            model_id=model_config['model_id'],
            num_classes=model_config['num_classes'],
            device=self.config['device']
        )
        self.hair_class_idx = model_config.get('hair_class_index', 17)
        
        # Preprocessor
        img_config = self.config['image']
        self.preprocessor = ImagePreprocessor(
            input_size=tuple(img_config['input_size'])
        )
        
        # Postprocessor
        self.postprocessor = MaskPostprocessor(self.config['postprocess'])
        
        # Output generator
        self.output_generator = OutputGenerator()
        
        print("Pipeline initialized successfully!")
    
    def process_single(self, image_path, output_path=None, visualize=False):
        """
        Process a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
            visualize: Whether to visualize results
            
        Returns:
            Dict with processing results
        """
        # Load image
        image = load_image(image_path)
        original_size = image.size
        
        # Preprocess (SegFormer handles most preprocessing internally)
        processed_image = self.preprocessor(image)
        
        # Run model inference - SegFormer expects PIL Image
        output = self.model.predict(processed_image)
        
        # Extract hair mask
        hair_mask = self.model.extract_hair_mask(output, self.hair_class_idx)
        
        # Postprocess mask
        hair_mask = self.postprocessor.process(hair_mask)
        
        # Generate output with transparent background
        hair_only = self.output_generator.create_transparent_hair(
            np.array(image), hair_mask
        )
        
        # Save if output path provided
        if output_path:
            save_image(hair_only, output_path)
            print(f"Saved result to {output_path}")
        
        # Visualize if requested
        if visualize:
            visualize_results(image, hair_mask, hair_only)
        
        return {
            'original': image,
            'mask': hair_mask,
            'hair_only': hair_only,
            'success': True
        }
    
    def process_batch(self, input_dir, output_dir, visualize=False, recursive=True):
        """
        Process all images in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            visualize: Whether to show visualization for each image
            recursive: If True, process subdirectories and preserve structure

        Returns:
            List of processed results
        """
        # Get all image files (recursive if specified)
        image_files = get_image_files(input_dir, recursive=recursive)

        if not image_files:
            print(f"No images found in {input_dir}")
            return []

        print(f"Found {len(image_files)} images to process")

        # Create output directory
        ensure_dir(output_dir)

        # Process each image
        results = []
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                # Preserve subdirectory structure
                if recursive:
                    # Get relative path from input_dir
                    rel_path = get_relative_path(image_path, input_dir)
                    # Create output path preserving directory structure
                    output_path = Path(output_dir) / rel_path
                    # Change extension to .png
                    output_path = output_path.with_suffix('.png')
                    # Ensure parent directory exists
                    ensure_dir(output_path.parent)
                    output_path = str(output_path)
                else:
                    # Flat structure - use original behavior
                    output_path = create_output_path(
                        image_path, output_dir, suffix="", extension=".png"
                    )

                # Process image
                result = self.process_single(
                    image_path, output_path, visualize=visualize
                )
                results.append({
                    'input_path': image_path,
                    'output_path': output_path,
                    'success': True,
                    'result': result
                })

            except Exception as e:
                print(f"\nError processing {image_path}: {e}")
                results.append({
                    'input_path': image_path,
                    'success': False,
                    'error': str(e)
                })

        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\nProcessing complete: {successful}/{len(results)} images processed successfully")

        return results
    
    def process_from_list(self, image_paths, output_dir, visualize=False):
        """
        Process a list of image paths.
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory path
            visualize: Whether to show visualization
            
        Returns:
            List of processed results
        """
        ensure_dir(output_dir)
        
        results = []
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                output_path = create_output_path(
                    image_path, output_dir, suffix="", extension=".png"
                )
                
                result = self.process_single(
                    image_path, output_path, visualize=visualize
                )
                results.append({
                    'input_path': image_path,
                    'output_path': output_path,
                    'success': True,
                    'result': result
                })
                
            except Exception as e:
                print(f"\nError processing {image_path}: {e}")
                results.append({
                    'input_path': image_path,
                    'success': False,
                    'error': str(e)
                })
        
        return results


def main():
    """Example usage of the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Hair Segmentation Pipeline')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file or directory')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file path')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize results')
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='Process subdirectories recursively (default: True)')
    parser.add_argument('--no-recursive', dest='recursive', action='store_false',
                       help='Process only files in the input directory (flat structure)')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = HairSegmentationPipeline(config_path=args.config)

    # Process single image or batch
    if os.path.isfile(args.input):
        # Single image
        pipeline.process_single(args.input, args.output, visualize=args.visualize)
    elif os.path.isdir(args.input):
        # Batch processing
        pipeline.process_batch(args.input, args.output,
                             visualize=args.visualize,
                             recursive=args.recursive)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == '__main__':
    main()