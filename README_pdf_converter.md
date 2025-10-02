# PDF to PNG Converter

A high-quality, command-line PDF to PNG converter that extracts pages from PDF documents and saves them as PNG images with customizable settings.

## Features

📄 **PDF Processing**: Extract any page range from PDF documents  
🖼️ **High Quality**: Customizable DPI (72-600) for different quality needs  
📁 **Flexible Output**: Custom output directories and filename patterns  
📊 **Detailed Info**: View PDF metadata and document information  
⚡ **Fast Processing**: Efficient page-by-page extraction with progress logging  
🔧 **Batch Options**: Extract specific page ranges or entire documents  

## Installation

```bash
# Install required packages
pip install -r requirements_pdf_converter.txt

# Or install manually
pip install PyMuPDF Pillow
```

## Quick Start

```bash
# Basic usage - extract all pages to 'pages' folder
python pdf_to_png.py document.pdf

# Custom output directory and DPI
python pdf_to_png.py document.pdf --output_dir images --dpi 300

# Extract specific pages (1-based indexing)
python pdf_to_png.py document.pdf --start_page 2 --end_page 5

# Show PDF information without extracting
python pdf_to_png.py document.pdf --info
```

## Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `pdf_file` | - | Input PDF file path | Required |
| `--output_dir` | `-o` | Output directory for PNG images | `pages` |
| `--dpi` | `-d` | Resolution in DPI (72-600) | `300` |
| `--prefix` | `-p` | Filename prefix | `page` |
| `--digits` | - | Number of digits for page numbering | `3` |
| `--start_page` | `-s` | First page to extract (1-based) | `1` |
| `--end_page` | `-e` | Last page to extract (1-based) | Last page |
| `--quality` | `-q` | PNG compression quality (1-100) | `95` |
| `--info` | - | Show PDF information and exit | False |
| `--quiet` | - | Reduce output verbosity | False |

## Usage Examples

### Basic Extraction
```bash
# Extract all pages with default settings
python pdf_to_png.py document.pdf

# Output: pages/page_001.png, pages/page_002.png, ...
```

### Custom Quality and Resolution
```bash
# High-quality extraction for printing
python pdf_to_png.py document.pdf --dpi 600 --quality 100

# Web-optimized extraction
python pdf_to_png.py document.pdf --dpi 150 --quality 85
```

### Page Range Extraction
```bash
# Extract pages 5-10
python pdf_to_png.py document.pdf --start_page 5 --end_page 10

# Extract only first page
python pdf_to_png.py document.pdf --start_page 1 --end_page 1
```

### Custom Output Format
```bash
# Custom filename pattern
python pdf_to_png.py document.pdf --prefix "doc_page" --digits 4
# Output: doc_page_0001.png, doc_page_0002.png, ...

# Custom output directory
python pdf_to_png.py document.pdf --output_dir "extracted_images"
```

### PDF Information
```bash
# View PDF metadata without extraction
python pdf_to_png.py document.pdf --info

# Example output:
# PDF Information for: document.pdf
#   Pages: 25
#   Title: Annual Report 2024
#   Author: John Smith
#   Page Size: 612 x 792 points
#   File Size: 2.45 MB
```

## Output Quality Guidelines

| DPI | Use Case | File Size | Quality |
|-----|----------|-----------|---------|
| 72-96 | Web display, thumbnails | Small | Basic |
| 150 | Standard viewing | Medium | Good |
| 300 | High quality, printing | Large | Excellent |
| 600 | Professional printing | Very Large | Maximum |

## File Naming Convention

Default naming pattern: `{prefix}_{page_number:0{digits}d}.png`

Examples:
- `page_001.png, page_002.png, ...` (default)
- `doc_0001.png, doc_0002.png, ...` (--prefix doc --digits 4)
- `slide_01.png, slide_02.png, ...` (--prefix slide --digits 2)

## Error Handling

The script includes comprehensive error handling for:
- ✅ **Invalid PDF files**: Validation before processing
- ✅ **Missing files**: Clear error messages for file not found
- ✅ **Permission issues**: Directory creation and file writing errors
- ✅ **Corrupted PDFs**: Graceful handling of damaged documents
- ✅ **Invalid parameters**: Range and value validation
- ✅ **Memory issues**: Efficient page-by-page processing

## Performance Tips

1. **DPI Selection**: Use appropriate DPI for your needs
   - 150 DPI: Good balance of quality and file size
   - 300 DPI: High quality for most uses
   - 600 DPI: Only for professional printing

2. **Batch Processing**: Extract page ranges instead of entire large documents when possible

3. **Memory Management**: The script processes pages individually to handle large PDFs efficiently

## Example Session

```bash
$ python pdf_to_png.py "Annual Report.pdf" --dpi 200 --start_page 1 --end_page 5

2025-10-02 10:22:14,048 - INFO - PDF validation successful: 25 pages found
2025-10-02 10:22:14,048 - INFO - Output directory ready: pages
2025-10-02 10:22:14,049 - INFO - Extracting pages 1 to 5 from 25 total pages
2025-10-02 10:22:14,234 - INFO - Saved page 1: page_001.png (1400x1980, 0.32MB)
2025-10-02 10:22:14,456 - INFO - Saved page 2: page_002.png (1400x1980, 0.45MB)
2025-10-02 10:22:14,678 - INFO - Saved page 3: page_003.png (1400x1980, 0.38MB)
2025-10-02 10:22:14,892 - INFO - Saved page 4: page_004.png (1400x1980, 0.41MB)
2025-10-02 10:22:15,103 - INFO - Saved page 5: page_005.png (1400x1980, 0.36MB)
2025-10-02 10:22:15,104 - INFO - Successfully extracted 5 pages

============================================================
PDF TO PNG EXTRACTION SUMMARY
============================================================
Input PDF: Annual Report.pdf
  Title: Annual Report 2024
  Pages: 25
  Page Size: 612 x 792 points
  File Size: 3.42 MB

Output Directory: pages
Images Created: 5
First Image: page_001.png
Last Image: page_005.png
Total Output Size: 1.92 MB
============================================================
```

## Related Files

- `pdf_to_png.py` - Main converter script
- `example_pdf_converter.py` - Usage examples and demos
- `requirements_pdf_converter.txt` - Required Python packages

## Dependencies

- **PyMuPDF (fitz)**: PDF processing and rendering
- **Pillow (PIL)**: Image processing and PNG optimization

## Troubleshooting

**Import Error**: Install missing packages with `pip install PyMuPDF Pillow`

**Permission Denied**: Check write permissions in output directory

**Large Files**: For very large PDFs, use page ranges or lower DPI

**Memory Issues**: Process pages in smaller batches for huge documents

**Poor Quality**: Increase DPI for better quality (trade-off with file size)

## Technical Details

- **PDF Processing**: Uses PyMuPDF for high-quality rendering
- **Image Format**: PNG with lossless compression
- **Color Space**: RGB color mode
- **Resolution**: Scalable DPI from 72 to 600
- **Memory Efficient**: Page-by-page processing prevents memory overflow