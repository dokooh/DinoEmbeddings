# DINOv2 Full Segmentation Script

This script converts the Jupyter notebook `fg_segmentation.ipynb` into a standalone Python script that performs advanced foreground segmentation and object analysis using Facebook's DINOv2 vision transformer.

## Features

🎯 **Foreground Segmentation**: Automatically separate foreground objects from background using PCA on DINOv2 features

🌈 **RGB Object Visualization**: Generate colorful visualizations where similar semantic regions have similar colors

🧠 **Transfer Learning**: Train on a few images and test on new unseen images

📊 **Multiple Model Sizes**: Support for DINOv2 small, base, large, and giant models

## How It Works

1. **Feature Extraction**: Uses DINOv2 to extract patch-level features from images
2. **Foreground Detection**: Applies PCA to separate foreground from background
3. **Object Analysis**: Uses 3-component PCA on foreground patches for RGB visualization  
4. **Generalization**: Applies learned patterns to new test images

## Installation

```bash
# Install required packages
pip install -r requirements_segmentation.txt
```

## Quick Start

```bash
# Show usage examples
python example_segmentation.py

# Basic usage with your images
python dinov2_full_segmentation.py --training_images image1.jpg image2.jpg image3.jpg

# With test image
python dinov2_full_segmentation.py --training_images train1.jpg train2.jpg --test_image test.jpg
```

## Using Sample Images

If you have run the PDF extractor, you can use the generated page images:

```bash
# Use PDF-generated images for segmentation
python dinov2_full_segmentation.py --training_images pages/page_001.png pages/page_002.png pages/page_003.png --test_image pages/page_004.png
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--training_images` | List of training image paths | Required |
| `--test_image` | Path to test image | Optional |
| `--model_size` | DINOv2 model: small/base/large/giant | base |
| `--threshold` | Foreground threshold (0.0-1.0) | 0.6 |
| `--target_size` | Training image size | 448 |
| `--test_size` | Test image size | 672 |
| `--save_dir` | Output directory | segmentation_results |

## Output

The script generates:

1. **`training_results.png`**: Shows original images, foreground masks, object features, and combined views
2. **`test_result.png`**: Shows test image and segmentation result (if test image provided)

## Example Results

- **Original Images**: Your input training images
- **Foreground Segmentation**: Highlights detected foreground regions
- **Object Features**: RGB visualization where colors represent semantic similarity
- **Test Results**: Segmentation applied to new unseen images

## Model Performance

| Model Size | Speed | Memory | Quality |
|------------|-------|--------|---------|
| Small | ⚡⚡⚡ | 💾 | ⭐⭐ |
| Base | ⚡⚡ | 💾💾 | ⭐⭐⭐ |
| Large | ⚡ | 💾💾💾 | ⭐⭐⭐⭐ |
| Giant | 🐌 | 💾💾💾💾 | ⭐⭐⭐⭐⭐ |

## Tips for Best Results

1. **Similar Objects**: Use training images containing similar objects for better generalization
2. **Good Lighting**: Well-lit images with clear foreground/background separation work best
3. **Threshold Tuning**: Adjust `--threshold` (0.4-0.8) based on your images
4. **Model Size**: Start with `base`, upgrade to `large` for better quality
5. **Image Size**: Larger images (672+) generally give better results

## Troubleshooting

**CUDA Out of Memory**: Use `--model_size small` or ensure GPU has sufficient memory

**Poor Segmentation**: Try adjusting `--threshold` or using more training images

**Import Errors**: Install missing packages with `pip install -r requirements_segmentation.txt`

## Technical Details

- **Architecture**: Vision Transformer (ViT) with DINOv2 self-supervised learning
- **Patch Size**: 14x14 pixels
- **Features**: 384-1536 dimensions depending on model size
- **PCA Components**: 1 for foreground/background, 3 for RGB visualization
- **Normalization**: Min-max scaling for stable results

## Original Research

This implementation is based on the DINOv2 paper and demonstrates the remarkable capability of self-supervised vision transformers for semantic understanding without any task-specific training.

## Related Files

- `dinov2_full_segmentation.py` - Main segmentation script
- `example_segmentation.py` - Usage examples and image checker
- `requirements_segmentation.txt` - Required Python packages
- `fg_segmentation.ipynb` - Original Jupyter notebook