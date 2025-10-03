#!/usr/bin/env python3
"""
Example usage of Gemini Table Detector

This script demonstrates how to use the GeminiTableDetector class
programmatically for batch processing or integration into other applications.
"""

import os
import sys
from pathlib import Path
from gemini_table_detector import GeminiTableDetector, TableDetectionResult

def example_single_pdf():
    """Example: Process a single PDF"""
    print("📄 Example: Single PDF Processing")
    print("=" * 40)
    
    # Check if sample PDF exists
    sample_pdf = "sample_document.pdf"
    if not os.path.exists(sample_pdf):
        print(f"⚠️  Sample PDF '{sample_pdf}' not found")
        print("Please provide a PDF file for testing")
        return
    
    try:
        # Initialize detector
        detector = GeminiTableDetector()
        
        # Process PDF with detailed analysis
        results = detector.process_pdf(
            pdf_path=sample_pdf,
            output_dir="example_results",
            detailed_analysis=True,
            save_images=True
        )
        
        # Display results
        total_tables = sum(r.tables_found for r in results)
        pages_with_tables = [r for r in results if r.tables_found > 0]
        
        print(f"✅ Processing complete!")
        print(f"   Total pages: {len(results)}")
        print(f"   Total tables: {total_tables}")
        print(f"   Pages with tables: {len(pages_with_tables)}")
        
        # Show detailed results for pages with tables
        for result in pages_with_tables:
            print(f"\n📊 Page {result.page_number}:")
            print(f"   Tables found: {result.tables_found}")
            for i, desc in enumerate(result.table_descriptions):
                confidence = result.confidence_scores[i] if i < len(result.confidence_scores) else 0.0
                print(f"   Table {i+1}: {desc} (confidence: {confidence:.2f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_batch_processing():
    """Example: Process multiple PDFs"""
    print("\n📚 Example: Batch Processing")
    print("=" * 40)
    
    # Find all PDFs in current directory
    pdf_files = list(Path(".").glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found in current directory")
        return
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    
    try:
        # Initialize detector once for batch processing
        detector = GeminiTableDetector()
        
        # Process each PDF
        batch_results = {}
        
        for pdf_file in pdf_files:
            print(f"\n🔍 Processing {pdf_file}...")
            
            # Create unique output directory for each PDF
            output_dir = f"batch_results_{pdf_file.stem}"
            
            # Process with quick mode for batch processing
            results = detector.process_pdf(
                pdf_path=str(pdf_file),
                output_dir=output_dir,
                detailed_analysis=False,  # Quick mode for batch
                save_images=False
            )
            
            # Store results
            total_tables = sum(r.tables_found for r in results)
            batch_results[str(pdf_file)] = {
                'pages': len(results),
                'tables': total_tables,
                'results': results
            }
            
            print(f"   ✅ {len(results)} pages, {total_tables} tables")
        
        # Summary report
        print(f"\n📊 Batch Processing Summary:")
        print("=" * 30)
        
        total_pages = sum(data['pages'] for data in batch_results.values())
        total_tables = sum(data['tables'] for data in batch_results.values())
        
        print(f"Files processed: {len(batch_results)}")
        print(f"Total pages: {total_pages}")
        print(f"Total tables: {total_tables}")
        
        # Per-file breakdown
        for pdf, data in batch_results.items():
            print(f"  {pdf}: {data['pages']} pages, {data['tables']} tables")
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")

def example_custom_analysis():
    """Example: Custom analysis with filtering"""
    print("\n🔬 Example: Custom Analysis")
    print("=" * 40)
    
    sample_pdf = "sample_document.pdf"
    if not os.path.exists(sample_pdf):
        print(f"⚠️  Sample PDF '{sample_pdf}' not found")
        return
    
    try:
        # Initialize detector
        detector = GeminiTableDetector()
        
        # Process with detailed analysis
        results = detector.process_pdf(
            pdf_path=sample_pdf,
            output_dir="custom_analysis",
            detailed_analysis=True
        )
        
        # Custom analysis: Find pages with high-confidence tables
        high_confidence_pages = []
        
        for result in results:
            if result.tables_found > 0:
                avg_confidence = sum(result.confidence_scores) / len(result.confidence_scores)
                if avg_confidence > 0.8:  # High confidence threshold
                    high_confidence_pages.append((result.page_number, avg_confidence))
        
        # Custom analysis: Find pages with multiple tables
        multi_table_pages = [r for r in results if r.tables_found > 1]
        
        # Custom analysis: Extract table descriptions containing keywords
        financial_tables = []
        for result in results:
            for desc in result.table_descriptions:
                if any(keyword in desc.lower() for keyword in ['financial', 'budget', 'revenue', 'cost']):
                    financial_tables.append((result.page_number, desc))
        
        # Report custom analysis
        print("🎯 Custom Analysis Results:")
        
        print(f"\n📈 High-confidence tables (>0.8):")
        if high_confidence_pages:
            for page, confidence in high_confidence_pages:
                print(f"   Page {page}: {confidence:.2f} confidence")
        else:
            print("   None found")
        
        print(f"\n📊 Pages with multiple tables:")
        if multi_table_pages:
            for result in multi_table_pages:
                print(f"   Page {result.page_number}: {result.tables_found} tables")
        else:
            print("   None found")
        
        print(f"\n💰 Financial-related tables:")
        if financial_tables:
            for page, desc in financial_tables:
                print(f"   Page {page}: {desc}")
        else:
            print("   None found")
        
    except Exception as e:
        print(f"❌ Custom analysis error: {e}")

def main():
    """Run all examples"""
    print("🎯 Gemini Table Detector Examples")
    print("=" * 50)
    
    # Check API key
    if not os.getenv('GOOGLE_AI_API_KEY'):
        print("❌ GOOGLE_AI_API_KEY not set")
        print("Run: python setup_gemini_detector.py")
        return
    
    # Run examples
    example_single_pdf()
    example_batch_processing()
    example_custom_analysis()
    
    print("\n🎉 Examples complete!")
    print("Check the output directories for results.")

if __name__ == "__main__":
    main()