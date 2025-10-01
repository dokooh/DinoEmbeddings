#!/usr/bin/env python3
"""
Example usage script for DinoV3 PDF embedding extraction

This script demonstrates how to use the DinoV3 PDF embedding extractor with a sample PDF.
"""

import os
import sys
from pdf_embedding_extractor_v3 import process_document

def main():
    # Example usage
    print("DinoV3 PDF Document Embedding Extractor")
    print("=" * 50)
    
    # Check if a PDF file is provided as argument
    if len(sys.argv) < 2:
        print("Usage: python example_usage_v3.py <path_to_pdf>")
        print("\nExample:")
        print("python example_usage_v3.py sample_document.pdf")
        print("\nOptional arguments:")
        print("python example_usage_v3.py sample_document.pdf --model facebook/dinov3-vitb16-pretrain-lvd1689m")
        return
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    print(f"Processing PDF with DinoV3: {pdf_path}")
    print("This will:")
    print("1. Convert each PDF page to an image in the 'pages' folder")
    print("2. Apply DinoV3 embedding extraction to each page")
    print("3. Generate PCA visualizations in the 'results' folder")
    print("4. Use Hugging Face Transformers for DinoV3 model")
    print()
    
    # Parse additional arguments
    model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    device = "auto"
    
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--model" and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]
        elif arg == "--device" and i + 1 < len(sys.argv):
            device = sys.argv[i + 1]
    
    print(f"Using model: {model_name}")
    print(f"Using device: {device}")
    print()
    
    try:
        # Process the document
        process_document(
            pdf_path=pdf_path,
            pages_dir="pages",
            results_dir="results",
            model_name=model_name,
            device=device
        )
        
        print("\n" + "=" * 50)
        print("DinoV3 processing completed successfully!")
        print("Check the 'pages' folder for extracted page images.")
        print("Check the 'results' folder for DinoV3 embedding visualizations.")
        print("\nDinoV3 features:")
        print("- Uses Hugging Face Transformers")
        print("- 16x16 patches with register tokens")
        print("- Improved self-supervised learning")
        print("- Better semantic understanding")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()