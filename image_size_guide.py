#!/usr/bin/env python3
"""
Image Size Guide for DinoV3 PDF Processing

This script provides guidance on choosing appropriate image sizes.
"""

def print_size_guide():
    """Print comprehensive guide for image size selection"""
    
    print("DinoV3 Image Size Selection Guide")
    print("=" * 50)
    print()
    
    print("📏 Recommended Image Sizes:")
    print("  224x224  - Minimum size (may reduce quality)")
    print("  518x518  - DinoV3 default (recommended)")
    print("  672x672  - High quality (good balance)")
    print("  896x896  - Very high quality (more memory)")
    print("  1024x1024 - Maximum quality (high memory)")
    print()
    
    print("💾 Memory Requirements (approximate):")
    print("  224x224  - Low memory (~1GB GPU)")
    print("  518x518  - Medium memory (~2GB GPU)")
    print("  672x672  - Medium-high memory (~3GB GPU)")
    print("  896x896  - High memory (~4GB GPU)")
    print("  1024x1024 - Very high memory (~6GB+ GPU)")
    print()
    
    print("🎯 Use Cases:")
    print("  224x224  - Quick testing, limited resources")
    print("  518x518  - General document processing")
    print("  672x672  - Detailed analysis, research")
    print("  896x896  - High-quality publications")
    print("  1024x1024 - Maximum detail extraction")
    print()
    
    print("⚙️ Performance Tips:")
    print("  • Use CPU for sizes >672 if GPU memory is limited")
    print("  • Larger sizes provide better patch resolution")
    print("  • DinoV3 works best with sizes divisible by 16")
    print("  • Consider your document complexity")
    print()
    
    print("🚀 Usage Examples:")
    print("  # Default size (518x518)")
    print("  python example_usage_v3.py document.pdf")
    print()
    print("  # High quality (672x672)")  
    print("  python example_usage_v3.py document.pdf --image_size 672")
    print()
    print("  # Maximum quality (1024x1024)")
    print("  python example_usage_v3.py document.pdf --image_size 1024 --device cpu")
    print()

if __name__ == "__main__":
    print_size_guide()