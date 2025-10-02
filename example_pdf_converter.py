#!/usr/bin/env python3
"""
Example usage for pdf_to_png.py

This script demonstrates various ways to use the PDF to PNG converter.
"""

import os
import subprocess
import sys

def run_command(cmd):
    """Run a command and show the output"""
    print(f"Running: {cmd}")
    print("-" * 50)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("Error:", result.stderr)
        print()
    except Exception as e:
        print(f"Error running command: {e}")
        print()

def main():
    print("PDF to PNG Converter - Usage Examples")
    print("=" * 60)
    print()
    
    # Check if we have sample PDFs
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if pdf_files:
        sample_pdf = pdf_files[0]
        print(f"Found sample PDF: {sample_pdf}")
        print()
        
        print("1. Basic Usage (default settings):")
        print(f"python pdf_to_png.py \"{sample_pdf}\"")
        print()
        
        print("2. Custom output directory and DPI:")
        print(f"python pdf_to_png.py \"{sample_pdf}\" --output_dir my_images --dpi 200")
        print()
        
        print("3. Extract specific page range:")
        print(f"python pdf_to_png.py \"{sample_pdf}\" --start_page 2 --end_page 5")
        print()
        
        print("4. Custom filename prefix and numbering:")
        print(f"python pdf_to_png.py \"{sample_pdf}\" --prefix document --digits 4")
        print()
        
        print("5. High quality extraction:")
        print(f"python pdf_to_png.py \"{sample_pdf}\" --dpi 300 --quality 100")
        print()
        
        print("6. Show PDF information:")
        print(f"python pdf_to_png.py \"{sample_pdf}\" --info")
        print()
        
        print("Would you like to run a demo? (y/n): ", end="")
        response = input().lower().strip()
        
        if response == 'y':
            print("\nRunning demo extraction...")
            demo_cmd = f'python pdf_to_png.py "{sample_pdf}" --output_dir demo_output --dpi 150 --prefix demo'
            run_command(demo_cmd)
            
            # Check results
            if os.path.exists('demo_output'):
                images = [f for f in os.listdir('demo_output') if f.endswith('.png')]
                print(f"Demo completed! Created {len(images)} images in demo_output/")
                if images:
                    print("Created files:")
                    for img in sorted(images)[:5]:  # Show first 5
                        print(f"  {img}")
                    if len(images) > 5:
                        print(f"  ... and {len(images) - 5} more")
    else:
        print("No PDF files found in current directory.")
        print()
        print("Usage examples with your own PDF:")
        print()
        print("1. Basic usage:")
        print("python pdf_to_png.py your_document.pdf")
        print()
        print("2. Custom settings:")
        print("python pdf_to_png.py your_document.pdf --output_dir images --dpi 300")
        print()
        print("3. Extract pages 1-10:")
        print("python pdf_to_png.py your_document.pdf --start_page 1 --end_page 10")
        print()
        print("4. High-resolution extraction:")
        print("python pdf_to_png.py your_document.pdf --dpi 600 --quality 100")
        print()
    
    print("Command Line Options:")
    print("-" * 30)
    print("--output_dir, -o    : Output directory (default: pages)")
    print("--dpi, -d          : Resolution in DPI (default: 300)")
    print("--prefix, -p       : Filename prefix (default: page)")
    print("--digits           : Number of digits for numbering (default: 3)")
    print("--start_page, -s   : First page to extract")
    print("--end_page, -e     : Last page to extract")
    print("--quality, -q      : PNG quality 1-100 (default: 95)")
    print("--info             : Show PDF information")
    print("--quiet            : Reduce output verbosity")
    print()
    
    print("Installation:")
    print("pip install -r requirements_pdf_converter.txt")

if __name__ == "__main__":
    main()