#!/usr/bin/env python3
"""
Gemini Flash PDF Table Detection Script

This script uses Google's Gemini Flash model to detect and analyze tables in PDF documents.
It extracts pages from PDFs as images and sends them to Gemini Flash for table detection.

Dependencies:
- google-generativeai
- PyMuPDF (fitz)
- Pillow (PIL)
- python-dotenv (for API key management)
"""

import os
import io
import json
import base64
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Using environment variables directly.")

@dataclass
class TableDetectionResult:
    """Result from table detection"""
    page_number: int
    tables_found: int
    table_descriptions: List[str]
    confidence_scores: List[float]
    raw_response: str
    image_path: Optional[str] = None

class GeminiTableDetector:
    """Gemini Flash-based table detector for PDF documents"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the Gemini table detector
        
        Args:
            api_key: Google AI API key (if None, will look for GOOGLE_AI_API_KEY env var)
            model_name: Gemini model to use (gemini-1.5-flash or gemini-1.5-pro)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Google AI API key required. Set GOOGLE_AI_API_KEY environment variable "
                "or pass api_key parameter"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        print(f"Initialized Gemini {model_name} for table detection")
    
    def extract_pdf_pages(self, pdf_path: str, output_dir: str = "temp_pages", 
                         dpi: int = 300) -> List[str]:
        """
        Extract pages from PDF as high-quality images
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save page images
            dpi: Image resolution (higher = better quality)
            
        Returns:
            List of paths to extracted page images
        """
        print(f"Extracting pages from PDF: {pdf_path}")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Open PDF
        doc = fitz.open(pdf_path)
        page_paths = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Convert page to image with high DPI for better table detection
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Save page image
            output_path = Path(output_dir) / f"page_{page_num + 1:03d}.png"
            img.save(output_path, "PNG", optimize=True)
            page_paths.append(str(output_path))
            
            print(f"  Extracted page {page_num + 1}: {output_path}")
        
        doc.close()
        print(f"Extracted {len(page_paths)} pages")
        return page_paths
    
    def detect_tables_in_image(self, image_path: str, 
                              detailed_analysis: bool = True) -> TableDetectionResult:
        """
        Detect tables in a single image using Gemini Flash
        
        Args:
            image_path: Path to image file
            detailed_analysis: Whether to perform detailed table analysis
            
        Returns:
            TableDetectionResult with detection results
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Create prompt for table detection
            if detailed_analysis:
                prompt = """
                Analyze this document page and detect all tables present. For each table found, provide:

                1. A brief description of the table content and structure
                2. The number of rows and columns (approximate)
                3. The table headers if visible
                4. A confidence score (0-1) for the detection
                5. The location/position of the table on the page (top, middle, bottom, etc.)

                Format your response as JSON with this structure:
                {
                    "tables_detected": number,
                    "tables": [
                        {
                            "description": "Brief description of table content",
                            "rows": estimated_number_of_rows,
                            "columns": estimated_number_of_columns,
                            "headers": ["header1", "header2", ...] or null,
                            "position": "position description",
                            "confidence": confidence_score_0_to_1
                        }
                    ],
                    "analysis_notes": "Additional observations about the document layout"
                }

                If no tables are found, return {"tables_detected": 0, "tables": [], "analysis_notes": "No tables detected"}.
                """
            else:
                prompt = """
                Quickly scan this document page and count how many tables are present.
                Provide a simple response with the number of tables found and a brief description of each.
                
                Format: "Found X table(s): [brief description of each table]"
                If no tables found, respond: "No tables detected"
                """
            
            # Generate response
            response = self.model.generate_content([prompt, img])
            response_text = response.text.strip()
            
            # Parse response
            if detailed_analysis:
                try:
                    # Try to parse JSON response
                    result_data = json.loads(response_text)
                    tables_found = result_data.get("tables_detected", 0)
                    tables = result_data.get("tables", [])
                    
                    descriptions = [table.get("description", "") for table in tables]
                    confidence_scores = [table.get("confidence", 0.0) for table in tables]
                    
                except json.JSONDecodeError:
                    # Fallback parsing if JSON fails
                    tables_found = self._extract_table_count_from_text(response_text)
                    descriptions = [response_text]
                    confidence_scores = [0.8]  # Default confidence
            else:
                tables_found = self._extract_table_count_from_text(response_text)
                descriptions = [response_text]
                confidence_scores = [0.8]
            
            # Extract page number from filename
            page_num = self._extract_page_number(image_path)
            
            return TableDetectionResult(
                page_number=page_num,
                tables_found=tables_found,
                table_descriptions=descriptions,
                confidence_scores=confidence_scores,
                raw_response=response_text,
                image_path=image_path
            )
            
        except Exception as e:
            print(f"Error detecting tables in {image_path}: {e}")
            page_num = self._extract_page_number(image_path)
            return TableDetectionResult(
                page_number=page_num,
                tables_found=0,
                table_descriptions=[f"Error: {str(e)}"],
                confidence_scores=[0.0],
                raw_response="",
                image_path=image_path
            )
    
    def _extract_table_count_from_text(self, text: str) -> int:
        """Extract table count from text response"""
        text_lower = text.lower()
        
        if "no table" in text_lower or "0 table" in text_lower:
            return 0
        
        # Look for patterns like "Found 2 table(s)", "2 tables detected", etc.
        import re
        patterns = [
            r'found (\d+) table',
            r'(\d+) table.*detected',
            r'detected (\d+) table',
            r'tables_detected["\']?\s*:\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return int(match.group(1))
        
        # Default: assume 1 table if text mentions tables but no count found
        if "table" in text_lower:
            return 1
        
        return 0
    
    def _extract_page_number(self, image_path: str) -> int:
        """Extract page number from image filename"""
        import re
        match = re.search(r'page_(\d+)', Path(image_path).stem)
        return int(match.group(1)) if match else 1
    
    def process_pdf(self, pdf_path: str, output_dir: str = "table_detection_results",
                   detailed_analysis: bool = True, save_images: bool = False,
                   dpi: int = 300) -> List[TableDetectionResult]:
        """
        Process entire PDF for table detection
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save results
            detailed_analysis: Whether to perform detailed analysis
            save_images: Whether to save extracted page images
            dpi: Image resolution for extraction
            
        Returns:
            List of TableDetectionResult for each page
        """
        print(f"Processing PDF with Gemini Flash: {pdf_path}")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Extract pages
        temp_dir = "temp_gemini_pages"
        page_paths = self.extract_pdf_pages(pdf_path, temp_dir, dpi)
        
        # Process each page
        results = []
        total_pages = len(page_paths)
        
        for i, page_path in enumerate(page_paths, 1):
            print(f"Analyzing page {i}/{total_pages} with Gemini Flash...")
            
            result = self.detect_tables_in_image(page_path, detailed_analysis)
            results.append(result)
            
            print(f"  Page {result.page_number}: {result.tables_found} table(s) detected")
            
            # Copy page image to results if requested
            if save_images:
                import shutil
                dest_path = Path(output_dir) / f"page_{result.page_number:03d}.png"
                shutil.copy2(page_path, dest_path)
                result.image_path = str(dest_path)
        
        # Clean up temporary images unless saving
        if not save_images:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Save results
        self._save_results(results, output_dir, pdf_path)
        
        return results
    
    def _save_results(self, results: List[TableDetectionResult], 
                     output_dir: str, pdf_path: str):
        """Save detection results to files"""
        # Create summary report
        total_tables = sum(r.tables_found for r in results)
        pages_with_tables = sum(1 for r in results if r.tables_found > 0)
        
        report_path = Path(output_dir) / "table_detection_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Gemini Flash Table Detection Report\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"PDF File: {pdf_path}\n")
            f.write(f"Total Pages: {len(results)}\n")
            f.write(f"Pages with Tables: {pages_with_tables}\n")
            f.write(f"Total Tables Detected: {total_tables}\n\n")
            
            for result in results:
                f.write(f"Page {result.page_number}:\n")
                f.write(f"  Tables Found: {result.tables_found}\n")
                
                if result.table_descriptions:
                    for i, desc in enumerate(result.table_descriptions):
                        confidence = result.confidence_scores[i] if i < len(result.confidence_scores) else 0.0
                        f.write(f"  Table {i+1}: {desc} (confidence: {confidence:.2f})\n")
                
                f.write(f"  Raw Response: {result.raw_response[:200]}...\n\n")
        
        # Create JSON results
        json_data = {
            "pdf_file": pdf_path,
            "total_pages": len(results),
            "pages_with_tables": pages_with_tables,
            "total_tables": total_tables,
            "results": [
                {
                    "page_number": r.page_number,
                    "tables_found": r.tables_found,
                    "table_descriptions": r.table_descriptions,
                    "confidence_scores": r.confidence_scores,
                    "raw_response": r.raw_response
                }
                for r in results
            ]
        }
        
        json_path = Path(output_dir) / "table_detection_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to:")
        print(f"  Report: {report_path}")
        print(f"  JSON: {json_path}")
        print(f"\nSummary:")
        print(f"  Total tables detected: {total_tables}")
        print(f"  Pages with tables: {pages_with_tables}/{len(results)}")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Detect tables in PDF documents using Gemini Flash",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gemini_table_detector.py document.pdf
  python gemini_table_detector.py document.pdf --detailed --save-images
  python gemini_table_detector.py document.pdf --output results --dpi 300
  python gemini_table_detector.py document.pdf --api-key YOUR_API_KEY
        """
    )
    
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", default="table_detection_results", 
                       help="Output directory (default: table_detection_results)")
    parser.add_argument("--api-key", help="Google AI API key (or set GOOGLE_AI_API_KEY env var)")
    parser.add_argument("--model", default="gemini-1.5-flash", 
                       choices=["gemini-1.5-flash", "gemini-1.5-pro"],
                       help="Gemini model to use (default: gemini-1.5-flash)")
    parser.add_argument("--detailed", action="store_true", 
                       help="Perform detailed table analysis")
    parser.add_argument("--save-images", action="store_true",
                       help="Save extracted page images")
    parser.add_argument("--dpi", type=int, default=300,
                       help="Image resolution for extraction (default: 300)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        return 1
    
    try:
        # Initialize detector
        detector = GeminiTableDetector(
            api_key=args.api_key,
            model_name=args.model
        )
        
        # Process PDF
        results = detector.process_pdf(
            pdf_path=args.pdf_path,
            output_dir=args.output,
            detailed_analysis=args.detailed,
            save_images=args.save_images,
            dpi=args.dpi
        )
        
        # Print summary
        total_tables = sum(r.tables_found for r in results)
        pages_with_tables = sum(1 for r in results if r.tables_found > 0)
        
        print(f"\n🎉 Table Detection Complete!")
        print(f"📄 Processed: {len(results)} pages")
        print(f"📊 Found: {total_tables} tables on {pages_with_tables} pages")
        print(f"💾 Results saved to: {args.output}/")
        
        if total_tables > 0:
            print(f"\n📋 Pages with tables:")
            for result in results:
                if result.tables_found > 0:
                    print(f"  Page {result.page_number}: {result.tables_found} table(s)")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())