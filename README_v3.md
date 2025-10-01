# PDF Document Embedding Extractor using DinoV3

This project extracts document embeddings from PDF files using Facebook's DinoV3 vision transformer model via Hugging Face Transformers. It converts each PDF page to an image, applies DinoV3 feature extraction, and generates PCA visualizations that highlight semantic regions in the document.

## Features

- Convert PDF pages to high-quality images
- Extract DinoV3 embeddings from document pages using Hugging Face Transformers
- Generate PCA visualizations highlighting semantic regions
- Automatic foreground/background separation
- GPU acceleration support
- Multiple DinoV3 model variants support
- Register tokens for improved performance
- Batch processing of multiple pages

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv dinoembeddingv3
dinoembeddingv3\Scripts\activate  # On Windows
# source dinoembeddingv3/bin/activate  # On Linux/macOS
```

2. Install required packages:
```bash
pip install -r requirements_v3.txt
```

## Usage

### Basic Usage

Process a PDF document using the example script:
```bash
python example_usage_v3.py path/to/your/document.pdf
```

### Advanced Usage

Use the main script with custom options:
```bash
python pdf_embedding_extractor_v3.py path/to/your/document.pdf --pages-dir custom_pages --results-dir custom_results --model facebook/dinov3-vitb16-pretrain-lvd1689m --device cuda
```

### Command Line Options

- `pdf_path`: Path to the PDF file (required)
- `--pages-dir`: Directory to save page images (default: "pages")
- `--results-dir`: Directory to save embedding visualizations (default: "results")
- `--model`: DinoV3 model name from Hugging Face (default: "facebook/dinov3-vits16-pretrain-lvd1689m")
- `--device`: Device to run the model on - "auto", "cpu", or "cuda" (default: "auto")

## Available DinoV3 Models

You can use different DinoV3 model variants:

- **Small models**: `facebook/dinov3-vits16-pretrain-lvd1689m` (default)
- **Base models**: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- **Large models**: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- **Huge models**: `facebook/dinov3-vith16plus-pretrain-lvd1689m`
- **Giant models**: `facebook/dinov3-vit7b16-pretrain-lvd1689m`

Larger models provide better performance but require more memory and computation.

## Output

The script creates two directories:

1. **pages/**: Contains individual page images extracted from the PDF
   - Format: `page_001.png`, `page_002.png`, etc.
   - High resolution (300 DPI) PNG images

2. **results/**: Contains DinoV3 embedding visualizations for each page
   - Format: `page_001_dinov3_embedding.png`, `page_002_dinov3_embedding.png`, etc.
   - RGB visualizations where colors represent semantic similarity
   - Foreground regions are highlighted with PCA colors
   - Background regions appear black

## How It Works

1. **PDF Conversion**: Each PDF page is converted to a high-resolution image using PyMuPDF
2. **Preprocessing**: Images are processed using DinoV3's AutoImageProcessor (typically 224x224 pixels)
3. **Feature Extraction**: DinoV3 model extracts patch embeddings (16x16 patches with register tokens)
4. **Foreground Detection**: First PCA component separates foreground from background
5. **Semantic Visualization**: Three PCA components of foreground embeddings create RGB visualization
6. **Output**: Visualizations are saved as images showing semantic regions

## Technical Details

- **Model**: DinoV3 variants (ViT architecture with 16x16 patch size)
- **Framework**: Hugging Face Transformers
- **Input Size**: 224x224 pixels (default for DinoV3)
- **Patch Size**: 16x16 pixels (14x14 patches total)
- **Register Tokens**: Special tokens that improve attention maps and performance
- **Embedding Dimension**: Varies by model (384 for ViT-S, 768 for ViT-B, etc.)
- **Output**: RGB visualization using top 3 PCA components

## Differences from DinoV2

DinoV3 improvements over DinoV2:
- **Register Tokens**: Special learnable tokens that act as "memory slots"
- **Better Attention Maps**: Cleaner attention with reduced high-norm artifacts
- **Improved Performance**: Better results on dense prediction tasks
- **Enhanced Training**: More sophisticated self-supervised learning
- **Hugging Face Integration**: Native support in transformers library

## Dependencies

- `torch>=2.8.0` - PyTorch deep learning framework
- `torchvision>=0.23.0` - Computer vision utilities
- `transformers>=4.40.0` - Hugging Face Transformers library
- `einops==0.7.0` - Tensor operations library
- `PyMuPDF` - PDF processing library
- `Pillow` - Image processing library

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for faster processing)
- Sufficient RAM for large documents (>4GB recommended)
- Internet connection for downloading DinoV3 models (first run only)

## Example Output

The DinoV3 embedding visualizations show:
- **Colored regions**: Semantically similar areas (text, images, diagrams)
- **Different colors**: Different semantic content types
- **Black regions**: Background or low-confidence areas
- **Spatial organization**: Preserved layout structure from original document
- **Improved quality**: Better semantic understanding compared to DinoV2

## Troubleshooting

- **CUDA out of memory**: Use `--device cpu` or switch to a smaller model variant
- **Model download errors**: Ensure internet connection for first-time model download
- **Large PDF files**: Process pages in batches or increase system RAM
- **Import errors**: Ensure all dependencies are installed in the virtual environment
- **Transformers version**: Ensure transformers>=4.40.0 for DinoV3 support

## Comparison with DinoV2 Version

| Feature | DinoV2 | DinoV3 |
|---------|---------|---------|
| Framework | torch.hub | Hugging Face Transformers |
| Input Size | 672x672 | 224x224 |
| Patches | 48x48 | 14x14 |
| Register Tokens | No | Yes (4 tokens) |
| Model Loading | Direct download | AutoModel.from_pretrained |
| Preprocessing | Manual normalization | AutoImageProcessor |
| Performance | Good | Better (especially dense tasks) |

## License

This project uses the DinoV3 model from Facebook Research via Hugging Face. Please refer to their license terms for commercial usage.