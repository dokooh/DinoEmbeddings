#!/usr/bin/env python3
"""
Setup helper for Gemini Table Detector

This script helps you set up the environment and API key for table detection.
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup environment and API key"""
    print("🔧 Gemini Table Detector Setup")
    print("=" * 40)
    
    # Check if API key is already set
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    
    if api_key:
        print("✅ Google AI API key found in environment")
    else:
        print("❌ Google AI API key not found")
        print("\n📝 To get your API key:")
        print("1. Go to: https://makersuite.google.com/app/apikey")
        print("2. Create a new API key")
        print("3. Set it as environment variable:")
        print("   Windows: set GOOGLE_AI_API_KEY=your_api_key_here")
        print("   Linux/Mac: export GOOGLE_AI_API_KEY=your_api_key_here")
        print("4. Or create a .env file with: GOOGLE_AI_API_KEY=your_api_key_here")
        
        # Offer to create .env file
        response = input("\nWould you like to create a .env file now? (y/n): ")
        if response.lower() == 'y':
            api_key = input("Enter your Google AI API key: ").strip()
            if api_key:
                with open('.env', 'w') as f:
                    f.write(f"GOOGLE_AI_API_KEY={api_key}\n")
                print("✅ Created .env file with your API key")
            else:
                print("❌ No API key provided")
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        ('google.generativeai', 'google-generativeai'),
        ('fitz', 'PyMuPDF'),
        ('PIL', 'Pillow'),
        ('dotenv', 'python-dotenv')
    ]
    
    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name}")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n🔧 Install missing packages:")
        print("pip install -r requirements_gemini_table_detector.txt")
        print("\nOr install individually:")
        for package in missing_packages:
            print(f"pip install {package}")
        return False
    
    print("\n🎉 Setup complete! Ready to detect tables with Gemini Flash.")
    print("\n🚀 Usage examples:")
    print("python gemini_table_detector.py document.pdf")
    print("python gemini_table_detector.py document.pdf --detailed --save-images")
    print("python gemini_table_detector.py document.pdf --output my_results --dpi 400")
    
    return True

def create_sample_env():
    """Create a sample .env file"""
    sample_content = """# Google AI API Key for Gemini Flash
# Get your key from: https://makersuite.google.com/app/apikey
GOOGLE_AI_API_KEY=your_api_key_here
"""
    
    with open('.env.sample', 'w') as f:
        f.write(sample_content)
    
    print("✅ Created .env.sample file")
    print("Copy it to .env and add your actual API key")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create-sample":
        create_sample_env()
    else:
        setup_environment()