# Hair Segmentation & Length Estimation Pipeline

A modular pipeline for segmenting hair and estimating hair length from images using **SegFormer**, 
a state-of-the-art transformer-based semantic segmentation model, combined with computer vision 
techniques for precise length measurement.

## Features

### Hair Segmentation
- **High-precision hair segmentation** using SegFormer transformer architecture
- **Superior boundary detection** - Better edge precision compared to CNN-based models
- **Transparent background output** - Hair isolated on transparent PNG
- **Batch processing** for multiple images
- **Configurable** via YAML configuration
- **Post-processing refinement** with morphological operations
- **Easy-to-use API** for integration into other projects
- **Automatic EXIF orientation handling** - Images are automatically rotated to correct orientation

### Hair Length Estimation 
- **Automatic A4 paper detection** - Uses reference object (A4 paper) for calibration
- **Color-based detection** - Robust detection of colored paper on colored backgrounds
- **Precise measurements** - Calculates pixels-per-cm ratio for accurate length estimation
- **JSON output** - Structured data output for easy integration
- **Multiple detection methods** - Color-based, edge-based, or automatic fallback
- **Configurable validation** - Ensures measurements are within reasonable ranges

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended, but CPU works too)

### Step 1: Clone the Repository

```bash
git clone https://github.com/LZ-sudo/hair_segmentation_model.git
cd hair_segmentation_model
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `torch` - Deep learning framework
- `torchvision` - Computer vision utilities
- `transformers>=4.30.0` - HuggingFace library for SegFormer
- `Pillow` - Image processing
- `opencv-python` - Post-processing operations
- `PyYAML` - Configuration management
- `numpy` - Numerical operations

### Step 3: Verify Installation

```bash
python -c "import torch; from transformers import SegformerForSemanticSegmentation; print('Installation successful!')"
```

## Model Information

This pipeline uses the **SegFormer** model fine-tuned on the **CelebAMask-HQ** dataset for face parsing.

- **Model**: [jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing)
- **Architecture**: SegFormer (Semantic Segmentation with Transformers)
- **Base Model**: NVIDIA MIT-B5
- **Training Dataset**: CelebAMask-HQ (30,000 high-resolution face images)
- **Classes**: 19 facial components including hair, skin, eyes, nose, etc.

### Why SegFormer?

SegFormer offers several advantages over traditional CNN-based models like BiSeNet:
- **Better boundary precision**: Transformer architecture captures fine details more accurately
- **Superior generalization**: Performs well across diverse face images
- **No positional encoding needed**: Handles various image resolutions naturally
- **Lightweight decoder**: Efficient while maintaining high accuracy

## Project Structure

```
hair_segmentation_model/
├── src/
│   ├── __init__.py
│   ├── model.py                     # SegFormer model wrapper
│   ├── processing.py                # Pre/post-processing utilities
│   ├── utils.py                     # File I/O and helper functions
│   └── hair_length_cv.py            # Hair length CV operations 
├── hair_segmentation_pipeline.py    # Main pipeline orchestrator
├── hair_length_estimator.py         # Hair length estimation orchestrator
├── config.yaml                      # Configuration file
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Configuration

Edit `config.yaml` to customize:

### Hair Segmentation Settings
```yaml
# Model settings
model:
  name: "segformer"
  model_id: "jonathandinu/face-parsing"
  num_classes: 19
  hair_class_index: 17

# Image processing
image:
  input_size: [512, 512]

# Post-processing
postprocess:
  morph_kernel_size: 5
  morph_iterations: 2
  apply_gaussian_blur: true
  gaussian_kernel_size: 5
  confidence_threshold: 0.5
```

### Hair Length Estimation Settings (NEW)
```yaml
hair_length:
  # Detection method
  detection_method: 'color'        # 'color', 'edges', or 'auto'
  paper_color: 'yellow'            # Color of A4 paper reference
  
  # Detection parameters
  min_area_pixels: 10000           # Minimum paper area
  aspect_ratio_tolerance: 0.15     # A4 aspect ratio tolerance
  min_confidence: 0.7              # Minimum detection confidence
  
  # Measurement validation
  min_length_cm: 1.0               # Minimum hair length
  max_length_cm: 200.0             # Maximum hair length

# Processing options
processing:
  file_pattern: "*.jpg"            # File pattern for batch processing
  visualize: false                 # Create visualization images
  save_mask: false                 # Save hair segmentation masks
```

### Adjusting Post-Processing

- **Increase `morph_kernel_size`**: More aggressive smoothing
- **Increase `morph_iterations`**: Cleaner mask, removes more noise
- **Enable `apply_gaussian_blur`**: Smoother edges
- **Adjust `confidence_threshold`**: Lower = more hair detected, Higher = stricter detection

## Usage

### Command Line

**Process a single image:**
```bash
python hair_segmentation_pipeline.py --input path/to/image.jpg --output path/to/output.png
```

**Process a directory of images:**
```bash
python hair_segmentation_pipeline.py --input path/to/images/ --output path/to/outputs/
```

**Process with subdirectories (recursive):**
```bash
python hair_segmentation_pipeline.py --input path/to/directory/ --output path/to/output_directory/ --recursive
```

**Process only top-level directory (non-recursive):**
```bash
python hair_segmentation_pipeline.py --input path/to/directory/ --output path/to/output_directory/ --no-recursive
```

**With visualization:**
```bash
python hair_segmentation_pipeline.py --input image.jpg --output output.png --visualize
```

## Usage - Hair Length Estimation

### Setup Requirements

For accurate hair length measurement, you need:
1. **A4 paper** (21 cm × 29.7 cm) - Any color works, yellow recommended
2. **Fixed camera position** - Camera should be 1m from wall
3. **Subject positioning** - Subject stands against wall with A4 paper next to them
4. **Good lighting** - Ensure paper is well-lit and clearly visible

### Single Image Processing
```bash
python hair_length_estimator.py --input subject.jpg --output results/
```

**Output:**
- `results/subject_result.json` - JSON file with measurement data

### Batch Processing
```bash
python hair_length_estimator.py --input photos/ --output results/
```

**Output:**
- `results/results.json` - JSON file with all measurements

### Hair Segmentation in Your Code (API)

**Single image processing:**
```python
from pipeline import HairSegmentationPipeline

# Initialize pipeline
pipeline = HairSegmentationPipeline(config_path='config.yaml')

# Process single image
result = pipeline.process_single(
    image_path='input.jpg',
    output_path='output.png',
    visualize=True
)

# Access results
original = result['original']      # PIL Image
mask = result['mask']              # numpy array [H, W]
hair_only = result['hair_only']    # PIL Image (RGBA with transparent background)
```

**Batch processing:**
```python
# Process all images in a directory
results = pipeline.process_batch(
    input_dir='data/images/',
    output_dir='data/outputs/',
    visualize=False,
    recursive=True  # Include subdirectories
)

# Check results
for r in results:
    if r['success']:
        print(f"✓ {r['input_path']}")
    else:
        print(f"✗ {r['input_path']}: {r['error']}")
```

**Process specific images:**
```python
# Process a list of images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = pipeline.process_from_list(
    image_paths=image_paths,
    output_dir='outputs/'
)
```

### Hair Length Estimation in Your Code (API)

```python
from hair_length_estimator import HairLengthEstimator

# Initialize estimator
estimator = HairLengthEstimator(config_path='config.yaml')

# Estimate hair length from single image
result = estimator.estimate_length('photo.jpg')

if result['success']:
    print(f"Hair length: {result['hair_length_cm']} cm")
    print(f"Confidence: {result['calibration_confidence']}")
else:
    print(f"Error: {result['error']}")
```

### Batch Processing in Code
```python
# Process directory
results = estimator.batch_process(
    input_dir='photos/',
    output_dir='results/',
    pattern='*.jpg'
)

# Analyze results
successful = [r for r in results if r['success']]
print(f"Processed {len(successful)} images successfully")

# Get statistics
lengths = [r['hair_length_cm'] for r in successful]
print(f"Average length: {sum(lengths)/len(lengths):.2f} cm")
```

## Citations

If you use this code or the SegFormer model in your research, please cite:

### SegFormer Model
```bibtex
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

### Pre-trained Model
The pre-trained face parsing model used in this project:
- **Model**: [jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing)
- **Base**: Fine-tuned from NVIDIA MIT-B5
- **Dataset**: CelebAMask-HQ

### CelebAMask-HQ Dataset
```bibtex
@inproceedings{CelebAMask-HQ,
  title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

## Acknowledgments

- **SegFormer**: Thank you to the authors of SegFormer for the excellent transformer-based segmentation architecture
- **HuggingFace**: For hosting the pre-trained model and providing the transformers library
- **jonathandinu**: For fine-tuning and sharing the face-parsing model on HuggingFace
- **CelebAMask-HQ**: For providing the high-quality face parsing dataset
- **PyTorch & HuggingFace Teams**: For the deep learning frameworks and model hub

## License

This project is intended for research and educational purposes. Please ensure you comply with:
- The licenses of the underlying models (SegFormer, pre-trained weights)
- The CelebAMask-HQ dataset license (non-commercial research use)
- Any applicable usage restrictions from HuggingFace

