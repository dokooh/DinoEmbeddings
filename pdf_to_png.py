#!/usr/bin/env python3
"""
PDF to PNG Converter

This script extracts pages from PDF documents and saves them as high-quality PNG images.
Each page is saved as a separate PNG file with customizable naming and quality settings.

Usage:
    python pdf_to_png.py input.pdf
    python pdf_to_png.py input.pdf --output_dir pages --dpi 300 --prefix page

Dependencies:
    - PyMuPDF (fitz)
    - Pillow (PIL)
"""

import os
import sys
import argparse
import fitz  # PyMuPDF
from PIL import Image
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_pdf_file(pdf_path):
    """
    Validate that the PDF file exists and is readable
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return False
    
    if not pdf_path.lower().endswith('.pdf'):
        logger.error(f"File is not a PDF: {pdf_path}")
        return False
    
    try:
        # Try to open the PDF to check if it's valid
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        
        if page_count == 0:
            logger.error(f"PDF file has no pages: {pdf_path}")
            return False
            
        logger.info(f"PDF validation successful: {page_count} pages found")
        return True
        
    except Exception as e:
        logger.error(f"Invalid PDF file: {pdf_path}, Error: {str(e)}")
        return False

def create_output_directory(output_dir):
    """
    Create output directory if it doesn't exist
    
    Args:
        output_dir (str): Path to output directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to create output directory: {output_dir}, Error: {str(e)}")
        return False

def extract_pdf_pages(pdf_path, output_dir, dpi=300, prefix="page", format_digits=3, 
                     start_page=None, end_page=None, quality=95):
    """
    Extract pages from PDF and save as PNG images
    
    Args:
        pdf_path (str): Path to input PDF file
        output_dir (str): Directory to save PNG images
        dpi (int): Resolution for image extraction (default: 300)
        prefix (str): Prefix for output filenames (default: "page")
        format_digits (int): Number of digits for page numbering (default: 3)
        start_page (int): First page to extract (1-based, None for first page)
        end_page (int): Last page to extract (1-based, None for last page)
        quality (int): JPEG quality for PNG compression (default: 95)
        
    Returns:
        list: List of successfully created image files
    """
    try:
        # Open PDF document
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Determine page range
        start_idx = (start_page - 1) if start_page else 0
        end_idx = end_page if end_page else total_pages
        
        # Validate page range
        start_idx = max(0, start_idx)
        end_idx = min(total_pages, end_idx)
        
        if start_idx >= end_idx:
            logger.error(f"Invalid page range: {start_page}-{end_page}")
            return []
        
        logger.info(f"Extracting pages {start_idx + 1} to {end_idx} from {total_pages} total pages")
        
        created_files = []
        
        # Process each page
        for page_num in range(start_idx, end_idx):
            try:
                # Get page
                page = doc[page_num]
                
                # Create transformation matrix for desired DPI
                # Default DPI is 72, so scale factor is dpi/72
                mat = fitz.Matrix(dpi/72, dpi/72)
                
                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Generate filename
                page_number = page_num + 1
                filename = f"{prefix}_{page_number:0{format_digits}d}.png"
                output_path = os.path.join(output_dir, filename)
                
                # Save as PNG with high quality
                img.save(output_path, "PNG", optimize=True, quality=quality)
                
                # Get image info
                width, height = img.size
                file_size = os.path.getsize(output_path)
                file_size_mb = file_size / (1024 * 1024)
                
                logger.info(f"Saved page {page_number}: {filename} ({width}x{height}, {file_size_mb:.2f}MB)")
                created_files.append(output_path)
                
            except Exception as e:
                logger.error(f"Failed to extract page {page_num + 1}: {str(e)}")
                continue
        
        doc.close()
        
        logger.info(f"Successfully extracted {len(created_files)} pages")
        return created_files
        
    except Exception as e:
        logger.error(f"Failed to process PDF: {str(e)}")
        return []

def get_pdf_info(pdf_path):
    """
    Get information about the PDF document
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        dict: PDF information
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        page_count = len(doc)
        
        # Get first page dimensions
        if page_count > 0:
            page = doc[0]
            rect = page.rect
            page_width = rect.width
            page_height = rect.height
        else:
            page_width = page_height = 0
        
        doc.close()
        
        info = {
            'pages': page_count,
            'title': metadata.get('title', 'Unknown'),
            'author': metadata.get('author', 'Unknown'),
            'subject': metadata.get('subject', 'Unknown'),
            'page_width': page_width,
            'page_height': page_height,
            'file_size': os.path.getsize(pdf_path)
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get PDF info: {str(e)}")
        return {}

def print_summary(pdf_path, created_files, output_dir):
    """
    Print extraction summary
    
    Args:
        pdf_path (str): Input PDF path
        created_files (list): List of created image files
        output_dir (str): Output directory
    """
    print("\n" + "="*60)
    print("PDF TO PNG EXTRACTION SUMMARY")
    print("="*60)
    
    # PDF info
    pdf_info = get_pdf_info(pdf_path)
    if pdf_info:
        print(f"Input PDF: {pdf_path}")
        print(f"  Title: {pdf_info['title']}")
        print(f"  Pages: {pdf_info['pages']}")
        print(f"  Page Size: {pdf_info['page_width']:.0f} x {pdf_info['page_height']:.0f} points")
        print(f"  File Size: {pdf_info['file_size'] / (1024*1024):.2f} MB")
    
    print(f"\nOutput Directory: {output_dir}")
    print(f"Images Created: {len(created_files)}")
    
    if created_files:
        print(f"First Image: {os.path.basename(created_files[0])}")
        print(f"Last Image: {os.path.basename(created_files[-1])}")
        
        # Calculate total size of created images
        total_size = sum(os.path.getsize(f) for f in created_files)
        total_size_mb = total_size / (1024 * 1024)
        print(f"Total Output Size: {total_size_mb:.2f} MB")
    
    print("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Extract pages from PDF documents and save as PNG images',
        epilog='Example: python pdf_to_png.py document.pdf --output_dir images --dpi 300 --prefix page',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('pdf_file', 
                       help='Path to input PDF file')
    
    # Optional arguments
    parser.add_argument('--output_dir', '-o', 
                       default='pages',
                       help='Output directory for PNG images (default: pages)')
    
    parser.add_argument('--dpi', '-d', 
                       type=int, default=300,
                       help='Resolution in DPI for image extraction (default: 300)')
    
    parser.add_argument('--prefix', '-p', 
                       default='page',
                       help='Prefix for output filenames (default: page)')
    
    parser.add_argument('--digits', 
                       type=int, default=3,
                       help='Number of digits for page numbering (default: 3)')
    
    parser.add_argument('--start_page', '-s', 
                       type=int, default=None,
                       help='First page to extract (1-based indexing)')
    
    parser.add_argument('--end_page', '-e', 
                       type=int, default=None,
                       help='Last page to extract (1-based indexing)')
    
    parser.add_argument('--quality', '-q', 
                       type=int, default=95,
                       help='PNG compression quality 1-100 (default: 95)')
    
    parser.add_argument('--info', 
                       action='store_true',
                       help='Show PDF information and exit')
    
    parser.add_argument('--quiet', 
                       action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Validate input file
    if not validate_pdf_file(args.pdf_file):
        return 1
    
    # Show PDF info if requested
    if args.info:
        info = get_pdf_info(args.pdf_file)
        if info:
            print(f"PDF Information for: {args.pdf_file}")
            print(f"  Pages: {info['pages']}")
            print(f"  Title: {info['title']}")
            print(f"  Author: {info['author']}")
            print(f"  Subject: {info['subject']}")
            print(f"  Page Size: {info['page_width']:.0f} x {info['page_height']:.0f} points")
            print(f"  File Size: {info['file_size'] / (1024*1024):.2f} MB")
        return 0
    
    # Validate parameters
    if args.dpi < 72 or args.dpi > 600:
        logger.warning(f"DPI {args.dpi} may produce very large or poor-quality images")
    
    if args.quality < 1 or args.quality > 100:
        logger.error("Quality must be between 1 and 100")
        return 1
    
    if args.start_page and args.start_page < 1:
        logger.error("Start page must be >= 1")
        return 1
    
    if args.end_page and args.end_page < 1:
        logger.error("End page must be >= 1")
        return 1
    
    if args.start_page and args.end_page and args.start_page > args.end_page:
        logger.error("Start page must be <= end page")
        return 1
    
    # Create output directory
    if not create_output_directory(args.output_dir):
        return 1
    
    try:
        # Extract pages
        created_files = extract_pdf_pages(
            pdf_path=args.pdf_file,
            output_dir=args.output_dir,
            dpi=args.dpi,
            prefix=args.prefix,
            format_digits=args.digits,
            start_page=args.start_page,
            end_page=args.end_page,
            quality=args.quality
        )
        
        if not created_files:
            logger.error("No pages were extracted")
            return 1
        
        # Print summary
        if not args.quiet:
            print_summary(args.pdf_file, created_files, args.output_dir)
        
        logger.info("PDF to PNG conversion completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Extraction cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    import io  # Add missing import
    exit(main())