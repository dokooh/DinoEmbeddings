#!/usr/bin/env python3
"""
Test script for Gemini Table Detector

Quick test to verify the setup is working correctly.
"""

import os
import tempfile
from pathlib import Path

def test_dependencies():
    """Test if all dependencies are available"""
    print("🧪 Testing Dependencies")
    print("=" * 30)
    
    dependencies = [
        ('google.generativeai', 'Google Generative AI'),
        ('fitz', 'PyMuPDF'),
        ('PIL', 'Pillow'),
        ('dotenv', 'python-dotenv (optional)')
    ]
    
    all_good = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Run: pip install {name.lower().replace(' ', '-')}")
            if module != 'dotenv':  # dotenv is optional
                all_good = False
    
    return all_good

def test_api_key():
    """Test if API key is available"""
    print("\n🔑 Testing API Key")
    print("=" * 20)
    
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    if api_key:
        print("✅ GOOGLE_AI_API_KEY found in environment")
        print(f"   Key starts with: {api_key[:10]}...")
        return True
    else:
        print("❌ GOOGLE_AI_API_KEY not found")
        print("   Set it with: set GOOGLE_AI_API_KEY=your_key (Windows)")
        print("   Or: export GOOGLE_AI_API_KEY=your_key (Linux/Mac)")
        return False

def create_sample_table_image():
    """Create a sample image with a table for testing"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create test image with a simple table structure
        img = Image.new('RGB', (800, 600), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw table structure
        draw.rectangle([100, 100, 700, 500], outline='black', width=2)
        draw.line([100, 200, 700, 200], fill='black', width=1)  # Header line
        draw.line([300, 100, 300, 500], fill='black', width=1)  # Column divider
        draw.line([500, 100, 500, 500], fill='black', width=1)  # Column divider
        
        # Add text content
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Headers
        draw.text((150, 150), "Product", fill='black', font=font)
        draw.text((350, 150), "Price", fill='black', font=font)
        draw.text((550, 150), "Stock", fill='black', font=font)
        
        # Data rows
        draw.text((150, 250), "Laptop", fill='black', font=font)
        draw.text((350, 250), "$999", fill='black', font=font)
        draw.text((550, 250), "15", fill='black', font=font)
        
        draw.text((150, 300), "Mouse", fill='black', font=font)
        draw.text((350, 300), "$25", fill='black', font=font)
        draw.text((550, 300), "50", fill='black', font=font)
        
        draw.text((150, 350), "Keyboard", fill='black', font=font)
        draw.text((350, 350), "$75", fill='black', font=font)
        draw.text((550, 350), "30", fill='black', font=font)
        
        # Save test image
        test_image_path = "test_table_sample.png"
        img.save(test_image_path)
        print(f"✅ Created sample table image: {test_image_path}")
        
        return test_image_path
        
    except Exception as e:
        print(f"❌ Failed to create sample image: {e}")
        return None

def test_gemini_connection():
    """Test Gemini API connection"""
    print("\n🔍 Testing Gemini Connection")
    print("=" * 30)
    
    try:
        from gemini_table_detector import GeminiTableDetector
        
        # Initialize detector
        detector = GeminiTableDetector()
        print("✅ Gemini Flash initialized successfully")
        
        # Create and test with sample image
        sample_image = create_sample_table_image()
        if not sample_image:
            return False
        
        print("📊 Testing table detection...")
        result = detector.detect_tables_in_image(sample_image, detailed_analysis=False)
        
        print(f"✅ Detection complete!")
        print(f"   Tables found: {result.tables_found}")
        print(f"   Response preview: {result.raw_response[:100]}...")
        
        # Clean up
        os.remove(sample_image)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Gemini Table Detector Test Suite")
    print("=" * 40)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test API key
    api_ok = test_api_key()
    
    if not deps_ok:
        print("\n❌ Dependencies missing. Please install them first.")
        print("Run: pip install -r requirements_gemini_table_detector.txt")
        return False
    
    if not api_ok:
        print("\n❌ API key missing. Please set up your Google AI API key.")
        print("Run: python setup_gemini_detector.py")
        return False
    
    # Test Gemini connection
    connection_ok = test_gemini_connection()
    
    if connection_ok:
        print("\n🎉 All tests passed!")
        print("Your Gemini Table Detector is ready to use!")
        print("\n🚀 Try it out:")
        print("python gemini_table_detector.py your_document.pdf")
        return True
    else:
        print("\n❌ Connection test failed.")
        print("Check your API key and internet connection.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)