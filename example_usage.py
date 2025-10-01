#!/usr/bin/env python3
"""
Example usage script for PDF embedding extraction

This script demonstrates how to use the PDF embedding extractor with a sample PDF.
"""

import os
import sys
from pdf_embedding_extractor import process_document

def main():
    # Example usage
    print("PDF Document Embedding Extractor")
    print("=" * 40)
    
    # Check if a PDF file is provided as argument
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <path_to_pdf>")
        print("\nExample:")
        print("python example_usage.py sample_document.pdf")
        return
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    print(f"Processing PDF: {pdf_path}")
    print("This will:")
    print("1. Convert each PDF page to an image in the 'pages' folder")
    print("2. Apply DinoV2 embedding extraction to each page")
    print("3. Generate PCA visualizations in the 'results' folder")
    print()
    
    try:
        # Process the document
        process_document(
            pdf_path=pdf_path,
            pages_dir="pages",
            results_dir="results",
            device="auto"  # Automatically choose GPU or CPU
        )
        
        print("\n" + "=" * 40)
        print("Processing completed successfully!")
        print("Check the 'pages' folder for extracted page images.")
        print("Check the 'results' folder for embedding visualizations.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()