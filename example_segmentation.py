#!/usr/bin/env python3
"""
Example usage for dinov2_full_segmentation.py

This script demonstrates how to use the DINOv2 segmentation pipeline
with sample images.
"""

import os
import sys

def create_sample_usage():
    """Print sample usage commands"""
    print("DINOv2 Full Segmentation - Usage Examples")
    print("=" * 50)
    print()
    
    print("1. Basic usage with training images:")
    print("python dinov2_full_segmentation.py --training_images image1.jpg image2.jpg image3.jpg")
    print()
    
    print("2. With test image:")
    print("python dinov2_full_segmentation.py --training_images train1.jpg train2.jpg --test_image test.jpg")
    print()
    
    print("3. Custom parameters:")
    print("python dinov2_full_segmentation.py \\")
    print("    --training_images crane1.jpg crane2.jpg crane3.jpg crane4.jpg \\")
    print("    --test_image crane_test.jpg \\")
    print("    --model_size base \\")
    print("    --threshold 0.6 \\")
    print("    --target_size 448 \\")
    print("    --test_size 672 \\")
    print("    --save_dir results")
    print()
    
    print("Available model sizes:")
    print("- small: dinov2_vits14 (fastest)")
    print("- base: dinov2_vitb14 (default)")
    print("- large: dinov2_vitl14 (better quality)")
    print("- giant: dinov2_vitg14 (best quality, slowest)")
    print()
    
    print("Required packages (install with: pip install -r requirements_segmentation.txt):")
    print("- torch, torchvision")
    print("- matplotlib, numpy")
    print("- opencv-python")
    print("- scikit-learn")
    print()

def check_sample_images():
    """Check if sample images exist"""
    sample_paths = [
        "pages/page_001.png",
        "pages/page_002.png", 
        "pages/page_003.png"
    ]
    
    available_images = [p for p in sample_paths if os.path.exists(p)]
    
    if available_images:
        print("Sample images found in pages/ directory:")
        for img in available_images:
            print(f"  {img}")
        print()
        print("You can test with these images:")
        print(f"python dinov2_full_segmentation.py --training_images {' '.join(available_images[:3])}")
        if len(available_images) > 3:
            print(f"python dinov2_full_segmentation.py --training_images {' '.join(available_images[:2])} --test_image {available_images[2]}")
    else:
        print("No sample images found in pages/ directory.")
        print("Please provide your own images or run the PDF extractor first to generate sample images.")

def main():
    create_sample_usage()
    check_sample_images()

if __name__ == "__main__":
    main()