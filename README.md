# Hair Segmentation Pipeline

A modular pipeline for segmenting and isolating hair from images using **SegFormer**, a state-of-the-art transformer-based semantic segmentation model.

## Features

- **High-precision hair segmentation** using SegFormer transformer architecture
- **Superior boundary detection** - Better edge precision compared to CNN-based models
- **Transparent background output** - Hair isolated on transparent PNG
- **Batch processing** for multiple images
- **Configurable** via YAML configuration
- **Post-processing refinement** with morphological operations
- **Easy-to-use API** for integration into other projects
- **Automatic EXIF orientation handling** - Images are automatically rotated to correct orientation

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
│   ├── model.py              # SegFormer model wrapper
│   ├── processing.py         # Pre/post-processing utilities
│   └── utils.py              # File I/O and helper functions
├── pipeline.py               # Main pipeline orchestrator
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Model settings
model:
  name: "segformer"
  model_id: "jonathandinu/face-parsing"  # HuggingFace model ID
  num_classes: 19
  hair_class_index: 17  # Auto-detected if not specified

# Image processing
image:
  input_size: [512, 512]  # Target size (SegFormer can handle any size)

# Post-processing
postprocess:
  morph_kernel_size: 5        # Morphological operation kernel size
  morph_iterations: 2         # Number of iterations for cleaning
  apply_gaussian_blur: true   # Smooth edges
  gaussian_kernel_size: 5     # Blur kernel size
  confidence_threshold: 0.5   # Threshold for binary mask

# Output settings
output:
  save_alpha: true  # Save with transparent background

# Processing
device: "cuda"  # "cuda" or "cpu"
batch_size: 1
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
python pipeline.py --input path/to/image.jpg --output path/to/output.png
```

**Process a directory of images:**
```bash
python pipeline.py --input path/to/images/ --output path/to/outputs/
```

**Process with subdirectories (recursive):**
```bash
python pipeline.py --input path/to/directory/ --output path/to/output_directory/ --recursive
```

**Process only top-level directory (non-recursive):**
```bash
python pipeline.py --input path/to/directory/ --output path/to/output_directory/ --no-recursive
```

**With visualization:**
```bash
python pipeline.py --input image.jpg --output output.png --visualize
```

### Python API

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

## Output Format

The pipeline generates PNG files with alpha channel (transparency):

- **Hair-only with transparent background**: Only the hair is visible, everything else is transparent
- **Maintains original resolution**: Output has the same dimensions as input
- **Correct orientation**: Automatically handles EXIF rotation data

Output files maintain the same filename as input with `.png` extension for transparency support.

## Examples

### Input → Output Comparison

The pipeline can handle various hair types and styles:
- ✅ Straight, wavy, and curly hair
- ✅ Different hair colors (black, brown, blonde, red, etc.)
- ✅ Various hairstyles (long, short, updos, etc.)
- ✅ Images from different angles
- ✅ Various lighting conditions

### Common Use Cases

1. **Hair color analysis**: Isolate hair for color classification or extraction
2. **Hair style classification**: Segment hair for texture/style analysis
3. **Virtual try-on**: Replace hair color or style in photos
4. **Dataset preparation**: Create masked hair images for training ML models
5. **Photo editing**: Professional hair isolation for compositing

## Troubleshooting

**Images appear rotated:**
- ✓ Fixed! The pipeline automatically handles EXIF orientation data

**Out of memory errors:**
- Switch to CPU: Set `device: "cpu"` in config.yaml
- Reduce image size before processing
- Process images one at a time

**Poor segmentation quality:**
- Adjust `confidence_threshold` in config (try 0.3 - 0.7 range)
- Tune post-processing parameters:
  - Increase `morph_iterations` for cleaner masks
  - Adjust `gaussian_kernel_size` for edge smoothing

**Slow processing:**
- Enable GPU: Set `device: "cuda"` (requires CUDA installation)
- Process in batches
- Use smaller input images

**Hair not fully detected:**
- Lower `confidence_threshold` (e.g., 0.3)
- Reduce `morph_iterations` to preserve fine details
- Disable `apply_gaussian_blur` for sharper edges

## Technical Details

### Model Architecture

**SegFormer** consists of:
1. **Hierarchical Transformer Encoder**: Captures multi-scale features
2. **Lightweight MLP Decoder**: Efficient segmentation head
3. **No Positional Encoding**: Resolution-independent design

### Processing Pipeline

1. **Load Image**: Read and convert to RGB with EXIF orientation correction
2. **Preprocessing**: Minimal preprocessing (SegFormer handles internally)
3. **Inference**: Run SegFormer model to get segmentation logits
4. **Extract Hair Mask**: Select hair class from multi-class output
5. **Post-processing**: Morphological operations and edge smoothing
6. **Generate Output**: Create RGBA image with transparent background

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

