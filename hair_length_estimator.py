"""
Hair Length Estimator - Main orchestrator script.

This script estimates hair length from images using:
1. A4 paper detection for calibration
2. Hair segmentation using SegFormer
3. Length calculation based on pixels-per-cm ratio

Usage:
    # Process single image
    python hair_length_estimator.py --input_dir image.jpg --output_dir output_folder/
    
    # Process directory of images
    python hair_length_estimator.py --input_dir input_folder/ --output_dir output_folder/

Output:
    - JSON file with measurement data (no visualizations by default)
    - To enable visualizations, set 'visualize: true' in config.yaml
"""

import argparse
import sys
import json
from pathlib import Path
import cv2
import numpy as np

from src.hair_length_cv import HairLengthCV, create_default_config
from src.model import HairSegmentationModel
from src.utils import load_config, load_image, ensure_dir


class HairLengthEstimator:
    """Main orchestrator for hair length estimation."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the hair length estimator.
        
        Args:
            config_path: Path to configuration file
        """
        print("="*60)
        print("Hair Length Estimator")
        print("="*60)
        
        # Load configuration
        try:
            self.config = load_config(config_path)
        except FileNotFoundError:
            print(f"Warning: Config file '{config_path}' not found. Using defaults.")
            self.config = self._create_default_config()
        
        # Initialize hair segmentation model
        print("\n[Initializing Hair Segmentation Model]")
        model_config = self.config.get('model', {})
        self.segmentation_model = HairSegmentationModel(
            model_id=model_config.get('model_id', 'jonathandinu/face-parsing'),
            num_classes=model_config.get('num_classes', 19),
            device=self.config.get('device', 'cuda')
        )
        
        # Get hair class index
        self.hair_class_idx = model_config.get('hair_class_index', 17)
        
        # Initialize CV module
        print("\n[Initializing Computer Vision Module]")
        cv_config = self.config.get('hair_length', create_default_config())
        self.cv_module = HairLengthCV(cv_config)
        
        print("\n" + "="*60)
        print("Initialization Complete!")
        print("="*60 + "\n")
    
    def _create_default_config(self) -> dict:
        """Create default configuration."""
        return {
            'device': 'cuda',
            'model': {
                'model_id': 'jonathandinu/face-parsing',
                'num_classes': 19,
                'hair_class_index': 17
            },
            'hair_length': create_default_config(),
            'processing': {
                'file_pattern': '*.jpg',
                'visualize': True,
                'save_mask': False
            }
        }
    
    def estimate_length(
        self,
        image_path: str,
        output_path: str = None
    ) -> dict:
        """
        Estimate hair length from an image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            
        Returns:
            Dictionary with measurement results
        """
        # Get processing options from config
        processing_config = self.config.get('processing', {})
        visualize = processing_config.get('visualize', True)
        save_mask = processing_config.get('save_mask', False)
        
        print(f"\nProcessing: {image_path}")
        print("="*60)
        
        # Load image
        image = load_image(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        print(f"Image size: {image.size}")
        
        # Convert PIL to numpy array
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Step 1: Detect A4 paper for calibration
        reference = self.cv_module.detect_a4_paper(image_bgr)
        
        if reference is None:
            print("\n✗ ERROR: Could not detect A4 paper in image")
            print("  Make sure:")
            print("  - A4 paper is clearly visible")
            print("  - Paper forms a clear rectangle")
            print("  - Paper is not folded or wrinkled")
            return {
                'success': False,
                'error': 'A4 paper not detected',
                'image_path': image_path
            }
        
        # Step 2: Segment hair
        print("\n[Hair Segmentation]")
        print("Running SegFormer model...")
        
        logits = self.segmentation_model.predict(image)
        hair_mask = self.segmentation_model.extract_hair_mask(
            logits, 
            self.hair_class_idx
        )
        
        # Convert to binary mask (0 or 255)
        hair_mask_binary = (hair_mask > 0.5).astype(np.uint8) * 255
        
        # Count hair pixels
        hair_pixels = np.sum(hair_mask_binary > 0)
        total_pixels = hair_mask_binary.size
        hair_percentage = (hair_pixels / total_pixels) * 100
        
        print(f"  ✓ Hair segmented")
        print(f"    Hair pixels: {hair_pixels:,} ({hair_percentage:.2f}%)")
        
        if hair_pixels == 0:
            print("\n✗ ERROR: No hair detected in image")
            return {
                'success': False,
                'error': 'No hair detected',
                'image_path': image_path
            }
        
        # Step 3: Find hair extremes
        top_point, bottom_point = self.cv_module.find_hair_extremes(hair_mask_binary)
        
        if top_point is None or bottom_point is None:
            print("\n✗ ERROR: Could not find hair boundaries")
            return {
                'success': False,
                'error': 'Hair boundaries not found',
                'image_path': image_path
            }
        
        # Step 4: Calculate hair length
        measurement = self.cv_module.calculate_hair_length(
            top_point,
            bottom_point,
            reference.pixels_per_cm,
            confidence=reference.confidence
        )
        
        # Validate measurement
        is_valid, validation_msg = self.cv_module.validate_measurement(measurement)
        
        if not is_valid:
            print(f"\n⚠ WARNING: {validation_msg}")
        
        # Create results dictionary
        results = {
            'success': True,
            'image_path': image_path,
            'hair_length_cm': round(measurement.length_cm, 2),
            'hair_length_pixels': round(measurement.length_pixels, 1),
            'pixels_per_cm': round(reference.pixels_per_cm, 2),
            'calibration_confidence': round(reference.confidence, 3),
            'top_point': measurement.top_point,
            'bottom_point': measurement.bottom_point,
            'validation': {
                'is_valid': is_valid,
                'message': validation_msg
            },
            'a4_paper': {
                'detected': True,
                'size_pixels': (int(reference.width_pixels), int(reference.height_pixels)),
                'size_cm': (reference.known_width_cm, reference.known_height_cm)
            }
        }
        
        # Create visualization
        if visualize or output_path:
            vis_image = self.cv_module.visualize_measurement(
                image_bgr,
                reference,
                measurement,
                hair_mask_binary
            )
            
            if output_path:
                # Ensure output directory exists
                ensure_dir(Path(output_path).parent)
                cv2.imwrite(output_path, vis_image)
                print(f"\n✓ Visualization saved: {output_path}")
                results['output_path'] = output_path
        
        # Save hair mask if requested
        if save_mask:
            mask_path = str(Path(image_path).stem) + "_hair_mask.png"
            if output_path:
                mask_path = str(Path(output_path).parent / mask_path)
            cv2.imwrite(mask_path, hair_mask_binary)
            print(f"✓ Hair mask saved: {mask_path}")
            results['mask_path'] = mask_path
        
        # Print summary
        print("\n" + "="*60)
        print("MEASUREMENT SUMMARY")
        print("="*60)
        print(f"Hair Length: {measurement.length_cm:.2f} cm")
        print(f"Calibration: {reference.pixels_per_cm:.2f} pixels/cm")
        print(f"Confidence: {reference.confidence:.1%}")
        print(f"Validation: {validation_msg}")
        print("="*60 + "\n")
        
        return results
    
    def batch_process(
        self,
        input_dir: str,
        output_dir: str = None,
        pattern: str = None
    ) -> list:
        """
        Process multiple images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results (not used, kept for compatibility)
            pattern: File pattern to match (overrides config if provided)
            
        Returns:
            List of result dictionaries
        """
        # Get file pattern from config or use provided
        if pattern is None:
            processing_config = self.config.get('processing', {})
            pattern = processing_config.get('file_pattern', '*.jpg')
        
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Find all matching images
        image_files = list(input_path.glob(pattern))
        
        if len(image_files) == 0:
            print(f"No images found matching pattern: {pattern}")
            return []
        
        print(f"\nFound {len(image_files)} images to process")
        print("="*60)
        
        results = []
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}]")
            
            # Process image
            try:
                result = self.estimate_length(
                    str(image_file)
                )
                results.append(result)
            except Exception as e:
                print(f"✗ Error processing {image_file.name}: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'image_path': str(image_file)
                })
        
        # Print batch summary
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        
        successful = sum(1 for r in results if r['success'])
        print(f"Total images: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        if successful > 0:
            lengths = [r['hair_length_cm'] for r in results if r['success']]
            print(f"\nHair lengths:")
            print(f"  Min: {min(lengths):.2f} cm")
            print(f"  Max: {max(lengths):.2f} cm")
            print(f"  Mean: {np.mean(lengths):.2f} cm")
            print(f"  Median: {np.median(lengths):.2f} cm")
        
        print("="*60 + "\n")
        
        return results


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Estimate hair length from images using A4 paper reference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python hair_length_estimator.py --input_dir image.jpg --output_dir output_folder/
  
  # Process directory of images
  python hair_length_estimator.py --input_dir input_folder/ --output_dir output_folder/

Note: All other settings (file patterns, visualization options, etc.) 
      are configured in config.yaml
      
Output: Results are saved as JSON file in the output directory
        """
    )
    
    # Arguments with flags
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image file or directory'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Directory to save results (JSON files only)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize estimator (requires config.yaml to exist)
        estimator = HairLengthEstimator(config_path='config.yaml')
        
        # Get processing options from config
        processing_config = estimator.config.get('processing', {})
        file_pattern = processing_config.get('file_pattern', '*.jpg')
        
        # Ensure output directory exists
        ensure_dir(Path(args.output_dir))
        
        # Determine if input is a file or directory
        input_path = Path(args.input_dir)
        
        if not input_path.exists():
            print(f"✗ Error: Input path does not exist: {args.input_dir}")
            return 1
        
        results = []
        json_filename = "results.json"
        
        if input_path.is_file():
            # Process single image
            print(f"\nProcessing single image: {args.input_dir}")
            print(f"Output directory: {args.output_dir}")
            print()
            
            result = estimator.estimate_length(
                str(input_path)
            )
            results.append(result)
            json_filename = input_path.stem + "_result.json"
        
        elif input_path.is_dir():
            # Process directory
            print(f"\nProcessing directory: {args.input_dir}")
            print(f"Output directory: {args.output_dir}")
            print(f"File pattern: {file_pattern}")
            print()
            
            results = estimator.batch_process(
                str(input_path),
                output_dir=args.output_dir,
                pattern=file_pattern
            )
        
        else:
            print(f"✗ Error: Input path must be a file or directory")
            return 1
        
        # Save results as JSON
        json_path = Path(args.output_dir) / json_filename
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {json_path}")
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        print(f"\nSummary:")
        print(f"  Total: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(results) - successful}")
        
        if successful > 0:
            return 0
        else:
            return 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())