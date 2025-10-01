# PDF Document Embedding Extractor using DinoV2

This project extracts document embeddings from PDF files using Facebook's DinoV2 vision transformer model. It converts each PDF page to an image, applies DinoV2 feature extraction, and generates PCA visualizations that highlight semantic regions in the document.

## Features

- Convert PDF pages to high-quality images
- Extract DinoV2 embeddings from document pages
- Generate PCA visualizations highlighting semantic regions
- Automatic foreground/background separation
- GPU acceleration support
- Batch processing of multiple pages

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv dinoembedding
dinoembedding\Scripts\activate  # On Windows
# source dinoembedding/bin/activate  # On Linux/macOS
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Process a PDF document using the example script:
```bash
python example_usage.py path/to/your/document.pdf
```

### Advanced Usage

Use the main script with custom options:
```bash
python pdf_embedding_extractor.py path/to/your/document.pdf --pages-dir custom_pages --results-dir custom_results --device cuda
```

### Command Line Options

- `pdf_path`: Path to the PDF file (required)
- `--pages-dir`: Directory to save page images (default: "pages")
- `--results-dir`: Directory to save embedding visualizations (default: "results")
- `--device`: Device to run the model on - "auto", "cpu", or "cuda" (default: "auto")

## Output

The script creates two directories:

1. **pages/**: Contains individual page images extracted from the PDF
   - Format: `page_001.png`, `page_002.png`, etc.
   - High resolution (300 DPI) PNG images

2. **results/**: Contains embedding visualizations for each page
   - Format: `page_001_embedding.png`, `page_002_embedding.png`, etc.
   - RGB visualizations where colors represent semantic similarity
   - Foreground regions are highlighted with PCA colors
   - Background regions appear black

## How It Works

1. **PDF Conversion**: Each PDF page is converted to a high-resolution image using PyMuPDF
2. **Preprocessing**: Images are resized to 672x672 pixels and normalized using ImageNet statistics
3. **Feature Extraction**: DinoV2 ViT-B/14 model extracts patch embeddings (48x48 patches)
4. **Foreground Detection**: First PCA component separates foreground from background
5. **Semantic Visualization**: Three PCA components of foreground embeddings create RGB visualization
6. **Output**: Visualizations are saved as images showing semantic regions

## Technical Details

- **Model**: DinoV2 ViT-B/14 (Vision Transformer Base, 14x14 patch size)
- **Input Size**: 672x672 pixels
- **Patch Size**: 14x14 pixels (48x48 patches total)
- **Embedding Dimension**: 768 features per patch
- **Output**: RGB visualization using top 3 PCA components

## Dependencies

- `torch>=2.8.0` - PyTorch deep learning framework
- `torchvision>=0.23.0` - Computer vision utilities
- `einops==0.7.0` - Tensor operations library
- `PyMuPDF` - PDF processing library
- `Pillow` - Image processing library

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for faster processing)
- Sufficient RAM for large documents (>4GB recommended)

## Example Output

The embedding visualizations show:
- **Colored regions**: Semantically similar areas (text, images, diagrams)
- **Different colors**: Different semantic content types
- **Black regions**: Background or low-confidence areas
- **Spatial organization**: Preserved layout structure from original document

## Troubleshooting

- **CUDA out of memory**: Use `--device cpu` to run on CPU
- **Large PDF files**: Process pages in batches or increase system RAM
- **Import errors**: Ensure all dependencies are installed in the virtual environment

## License

This project uses the DinoV2 model from Facebook Research. Please refer to their license terms for commercial usage.