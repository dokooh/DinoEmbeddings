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
        print("Usage: python example_usage_v3.py <path_to_pdf> [--image_size SIZE]")
        print("\nExample:")
        print("python example_usage_v3.py sample_document.pdf")
        print("python example_usage_v3.py sample_document.pdf --image_size 512")
        print("python example_usage_v3.py sample_document.pdf --image_size 768")
        print("\nOptional arguments:")
        print("  --model MODEL_NAME     DinoV3 model to use")
        print("  --image_size SIZE      Input/output image size (default: 518)")
        print("  --device DEVICE        Device to use (auto/cpu/cuda)")
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
    image_size = 518  # Increased from 224 to 518 (DinoV3 default)
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--device" and i + 1 < len(sys.argv):
            device = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--image_size" and i + 1 < len(sys.argv):
            try:
                image_size = int(sys.argv[i + 1])
                if image_size < 224:
                    print("Warning: Image size below 224 may cause issues. Using 224.")
                    image_size = 224
                elif image_size > 1024:
                    print("Warning: Very large image sizes may cause memory issues.")
            except ValueError:
                print("Warning: Invalid image size. Using default 518.")
                image_size = 518
            i += 2
        else:
            i += 1
    
    print(f"Using model: {model_name}")
    print(f"Using device: {device}")
    print(f"Using image size: {image_size}x{image_size}")
    print()
    
    try:
        # Process the document with configurable image size
        process_document(
            pdf_path=pdf_path,
            pages_dir="pages",
            results_dir="results",
            model_name=model_name,
            device=device,
            image_size=image_size  # Pass the configurable image size
        )
        
        print("\n" + "=" * 50)
        print("DinoV3 processing completed successfully!")
        print("Check the 'pages' folder for extracted page images.")
        print("Check the 'results' folder for DinoV3 embedding visualizations.")
        print(f"Output images generated at {image_size}x{image_size} resolution.")
        print("\nDinoV3 features:")
        print("- Uses Hugging Face Transformers")
        print("- 16x16 patches with register tokens")
        print("- Configurable input/output image sizes")
        print("- Improved self-supervised learning")
        print("- Better semantic understanding")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting:")
        print("- If you get CUDA errors, try: --device cpu")
        print("- If you get memory errors, try a smaller --image_size")
        print("- For gated models, authenticate with: huggingface-cli login")
        sys.exit(1)

if __name__ == "__main__":
    main()