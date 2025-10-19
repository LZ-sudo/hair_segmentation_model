# Hair Segmentation Pipeline

A modular pipeline for segmenting and isolating hair from images using BiSeNet.

## Features

- **Accurate hair segmentation** using BiSeNet architecture
- **Transparent background output** - Hair isolated on transparent PNG
- **Batch processing** for multiple images
- **Configurable** via YAML configuration
- **Post-processing refinement** with morphological operations
- **Easy-to-use API** for integration into other projects

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LZ-sudo/hair_segmentation_model.git
cd hair_segmentation_model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pretrained weights:
   - Download BiSeNet pretrained weights from [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
   - Place the weights file (e.g., `79999_iter.pth`) in the `pretrained/` directory
   - Update the path in `config.yaml` if needed

## Project Structure

```
hair-segmentation/
├── src/
│   ├── __init__.py
│   ├── model.py              # BiSeNet model and inference
│   ├── processing.py         # Pre/post-processing utilities
│   └── utils.py              # File I/O and visualization
│   └── 79999_iter.pth        # BiSeNet pretrained weights 
├── pipeline.py               # Main pipeline orchestrator
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Configuration

Edit `config.yaml` to customize:

- **Model settings**: path to pretrained weights, number of classes
- **Image processing**: input size, normalization parameters
- **Post-processing**: morphological operations, blur settings
- **Output formats**: which outputs to generate, background colors
- **Device**: CPU or CUDA

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

**Process a directory with subdirectories of images:**
```bash
python pipeline.py --input path/to/directory/ --output path/to/output_directory/
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
original = result['original']
mask = result['mask']
hair_only = result['hair_only']  # This is your hair with transparent background
```

**Batch processing:**
```python
# Process all images in a directory
results = pipeline.process_batch(
    input_dir='data/images/',
    output_dir='data/outputs/',
    visualize=False
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

The pipeline generates a single output format:

- **Hair-only with transparent background**: PNG file with alpha channel where only the hair is visible

Output files maintain the same filename as input with `.png` extension for transparency support.

## Customization

### Adjust Post-Processing

Edit `config.yaml` to tune refinement:

```yaml
postprocess:
  morph_kernel_size: 5        # Larger = more aggressive smoothing
  morph_iterations: 2         # More iterations = cleaner mask
  apply_gaussian_blur: true   # Smooth edges
  gaussian_kernel_size: 5     # Blur strength
```

### Use Different Device

```yaml
device: "cuda"  # Use GPU
# device: "cpu"   # Use CPU
```

### Visualization

```python
from src.utils import visualize_results, visualize_comparison

# Visualize single result
visualize_results(image, mask, hair_only)

# Compare multiple outputs
visualize_comparison(
    [image, mask, hair_only],
    ['Original', 'Mask', 'Hair Only']
)
```

## Troubleshooting

**Out of memory errors:**
- Reduce `batch_size` in config
- Use smaller `input_size`
- Switch to CPU if GPU memory is limited

**Poor segmentation quality:**
- Ensure you're using pretrained weights trained on face parsing
- Adjust `confidence_threshold` in config
- Tune post-processing parameters
- Try different `morph_kernel_size` values

**Slow processing:**
- Enable GPU (`device: "cuda"`)
- Increase `batch_size` if memory allows
- Reduce `input_size` (trade-off with quality)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- NumPy
- PIL/Pillow
- PyYAML

See `requirements.txt` for complete list.

## Acknowledgments

- BiSeNet architecture based on [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
- Pretrained models trained on CelebAMask-HQ dataset