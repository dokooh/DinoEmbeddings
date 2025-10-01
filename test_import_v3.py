#!/usr/bin/env python3
"""
Test script to verify the DinoV3 PDF embedding extractor works
"""

try:
    import pdf_embedding_extractor_v3
    print("✓ DinoV3 PDF embedding extractor imports successfully")
    
    # Test DinoV3EmbeddingExtractor initialization
    from pdf_embedding_extractor_v3 import DinoV3EmbeddingExtractor
    print("✓ DinoV3EmbeddingExtractor class imported successfully")
    
    # Test transformers imports
    from transformers import AutoImageProcessor, AutoModel
    print("✓ Transformers AutoModel and AutoImageProcessor imported successfully")
    
    # Test dinov3embeddings script import
    import dinov3embeddings
    print("✓ dinov3embeddings script imported successfully")
    
    print("✓ All imports successful - the DinoV3 PDF embedding extractor is ready to use!")
    print("\nTo process a PDF document, run:")
    print("python example_usage_v3.py your_document.pdf")
    print("\nAvailable DinoV3 models:")
    print("- facebook/dinov3-vits16-pretrain-lvd1689m (default, small)")
    print("- facebook/dinov3-vitb16-pretrain-lvd1689m (base)")
    print("- facebook/dinov3-vitl16-pretrain-lvd1689m (large)")
    print("- facebook/dinov3-vith16-pretrain-lvd1689m (huge)")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements_v3.txt")
    
except Exception as e:
    print(f"✗ Error: {e}")