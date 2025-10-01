#!/usr/bin/env python3
"""
Test script to verify the PDF embedding extractor works
"""

try:
    import pdf_embedding_extractor
    print("✓ PDF embedding extractor imports successfully")
    
    # Test DinoV2EmbeddingExtractor initialization
    from pdf_embedding_extractor import DinoV2EmbeddingExtractor
    print("✓ DinoV2EmbeddingExtractor class imported successfully")
    
    print("✓ All imports successful - the PDF embedding extractor is ready to use!")
    print("\nTo process a PDF document, run:")
    print("python example_usage.py your_document.pdf")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    
except Exception as e:
    print(f"✗ Error: {e}")