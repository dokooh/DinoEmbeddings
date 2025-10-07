"""
DINOv2 PDF Visual Embeddings Extractor
Learns visual representations of tables and sections in PDF documents
"""

import torch
import numpy as np
from PIL import Image
import pdf2image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import pickle
import argparse
import sys
import json

# Check if transformers is available for DINOv2
try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    print("Please install transformers: pip install transformers")
    raise


class PDFVisualEmbedder:
    """Extract and embed visual regions from PDF documents using DINOv2"""
    
    def __init__(self, model_name="facebook/dinov2-base", device=None):
        """
        Initialize the visual embedder
        
        Args:
            model_name: DINOv2 model variant to use
            device: torch device (cuda/cpu)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load DINOv2 model and processor
        print(f"Loading {model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[Image.Image]:
        """
        Convert PDF pages to images
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            List of PIL Images
        """
        print(f"Converting PDF to images at {dpi} DPI...")
        images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
        print(f"Converted {len(images)} pages")
        return images
    
    def detect_regions(self, image: Image.Image) -> List[Dict]:
        """
        Detect potential table and section regions in the image
        This is a simple implementation using basic heuristics
        For production, consider using models like LayoutParser or table detection models
        
        Args:
            image: PIL Image of PDF page
            
        Returns:
            List of region dictionaries with bbox coordinates
        """
        # Convert to numpy for processing
        img_array = np.array(image.convert('L'))
        height, width = img_array.shape
        
        # Simple grid-based sectioning (replace with actual detection model)
        regions = []
        
        # Divide page into potential sections (top, middle, bottom)
        section_height = height // 3
        for i, section_name in enumerate(['top', 'middle', 'bottom']):
            regions.append({
                'type': 'section',
                'name': section_name,
                'bbox': (0, i * section_height, width, (i + 1) * section_height)
            })
        
        # Add some example table-like regions (in practice, use a table detection model)
        # These are placeholder regions - replace with actual table detection
        potential_tables = [
            (int(width * 0.1), int(height * 0.2), int(width * 0.9), int(height * 0.4)),
            (int(width * 0.1), int(height * 0.5), int(width * 0.9), int(height * 0.7)),
        ]
        
        for idx, bbox in enumerate(potential_tables):
            regions.append({
                'type': 'table',
                'name': f'table_{idx}',
                'bbox': bbox
            })
        
        return regions
    
    def extract_region(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        Extract a region from the image
        
        Args:
            image: PIL Image
            bbox: (x1, y1, x2, y2) coordinates
            
        Returns:
            Cropped PIL Image
        """
        return image.crop(bbox)
    
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Get DINOv2 embedding for an image region
        
        Args:
            image: PIL Image
            
        Returns:
            Embedding vector as numpy array
        """
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.squeeze()
    
    def process_pdf(self, pdf_path: str, dpi: int = 200) -> Dict:
        """
        Process entire PDF and extract embeddings for all regions
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path, dpi=dpi)
        
        results = {
            'pdf_path': pdf_path,
            'num_pages': len(images),
            'pages': []
        }
        
        # Process each page
        for page_idx, image in enumerate(images):
            print(f"\nProcessing page {page_idx + 1}/{len(images)}...")
            
            # Detect regions
            regions = self.detect_regions(image)
            
            page_data = {
                'page_num': page_idx + 1,
                'regions': []
            }
            
            # Extract embeddings for each region
            for region in regions:
                print(f"  Extracting {region['type']}: {region['name']}")
                
                # Crop region
                region_img = self.extract_region(image, region['bbox'])
                
                # Get embedding
                embedding = self.get_embedding(region_img)
                
                page_data['regions'].append({
                    'type': region['type'],
                    'name': region['name'],
                    'bbox': region['bbox'],
                    'embedding': embedding,
                    'embedding_dim': len(embedding)
                })
            
            results['pages'].append(page_data)
        
        return results
    
    def visualize_regions(self, image: Image.Image, regions: List[Dict], save_path: str = None):
        """
        Visualize detected regions on the page
        
        Args:
            image: PIL Image of the page
            regions: List of region dictionaries
            save_path: Optional path to save the visualization
        """
        fig, ax = plt.subplots(1, figsize=(12, 16))
        ax.imshow(image)
        
        colors = {'section': 'blue', 'table': 'red'}
        
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=colors.get(region['type'], 'green'),
                facecolor='none',
                label=f"{region['type']}: {region['name']}"
            )
            ax.add_patch(rect)
            
            # Add text label
            ax.text(x1, y1 - 10, region['name'], 
                   color=colors.get(region['type'], 'green'),
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.axis('off')
        plt.title('Detected Regions')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def save_region_images(self, image: Image.Image, regions: List[Dict], 
                          output_dir: str, page_num: int, 
                          save_with_bbox: bool = True):
        """
        Save individual region images with optional bounding boxes
        
        Args:
            image: PIL Image of the page
            regions: List of region dictionaries
            output_dir: Directory to save region images
            page_num: Page number for filename
            save_with_bbox: If True, save images with bounding boxes drawn
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        colors_rgb = {
            'section': (0, 0, 255),  # Blue
            'table': (255, 0, 0),     # Red
            'default': (0, 255, 0)    # Green
        }
        
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            region_type = region['type']
            region_name = region['name']
            
            # Crop the region
            region_img = image.crop((x1, y1, x2, y2))
            
            # Save cropped region without bbox
            filename_base = f"page{page_num:03d}_{region_type}_{region_name}"
            crop_path = output_path / f"{filename_base}_crop.png"
            region_img.save(crop_path)
            
            # Save with bounding box if requested
            if save_with_bbox:
                # Create a copy of the cropped image to draw on
                import PIL.ImageDraw as ImageDraw
                import PIL.ImageFont as ImageFont
                
                bbox_img = region_img.copy()
                draw = ImageDraw.Draw(bbox_img)
                
                # Draw border
                color = colors_rgb.get(region_type, colors_rgb['default'])
                border_width = max(2, min(bbox_img.width, bbox_img.height) // 100)
                
                # Draw rectangle border
                for i in range(border_width):
                    draw.rectangle(
                        [i, i, bbox_img.width - 1 - i, bbox_img.height - 1 - i],
                        outline=color,
                        width=1
                    )
                
                # Add label
                try:
                    # Try to use a nicer font if available
                    font_size = max(12, min(bbox_img.width, bbox_img.height) // 20)
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    # Fallback to default font
                    font = ImageFont.load_default()
                
                label = f"{region_type}: {region_name}"
                
                # Get text bounding box
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw background for text
                padding = 5
                draw.rectangle(
                    [padding, padding, text_width + 2*padding, text_height + 2*padding],
                    fill=(255, 255, 255, 200)
                )
                
                # Draw text
                draw.text((padding, padding), label, fill=color, font=font)
                
                bbox_path = output_path / f"{filename_base}_bbox.png"
                bbox_img.save(bbox_path)
        
        print(f"  Saved {len(regions)} region images to {output_dir}/")
    
    def save_embeddings(self, results: Dict, output_path: str):
        """
        Save embeddings to disk
        
        Args:
            results: Results dictionary from process_pdf
            output_path: Path to save the embeddings
        """
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nEmbeddings saved to {output_path}")
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Extract visual embeddings from PDF documents using DINOv2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python script.py document.pdf
  
  # Specify output path and DPI
  python script.py document.pdf --output embeddings.pkl --dpi 300
  
  # Use larger model and save visualization
  python script.py document.pdf --model facebook/dinov2-large --visualize
  
  # Process multiple pages with specific range
  python script.py document.pdf --pages 1-5 --no-save
  
  # Compare similarity between regions
  python script.py document.pdf --compare-regions
  
  # Export to JSON format
  python script.py document.pdf --format json
        """
    )
    
    parser.add_argument(
        'pdf_path',
        type=str,
        help='Path to the PDF file to process'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='pdf_embeddings.pkl',
        help='Output path for embeddings (default: pdf_embeddings.pkl)'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='facebook/dinov2-base',
        choices=['facebook/dinov2-small', 'facebook/dinov2-base', 
                 'facebook/dinov2-large', 'facebook/dinov2-giant'],
        help='DINOv2 model variant to use (default: facebook/dinov2-base)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='DPI resolution for PDF conversion (default: 200)'
    )
    
    parser.add_argument(
        '--pages',
        type=str,
        default=None,
        help='Page range to process (e.g., "1-5" or "1,3,5")'
    )
    
    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='Generate and save visualization of detected regions'
    )
    
    parser.add_argument(
        '--viz-output',
        type=str,
        default='regions_visualization.png',
        help='Output path for visualization (default: regions_visualization.png)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save embeddings to disk'
    )
    
    parser.add_argument(
        '--compare-regions',
        action='store_true',
        help='Compute and display similarity between regions'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['pickle', 'json'],
        default='pickle',
        help='Output format for embeddings (default: pickle)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to run model on (default: auto)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed processing information'
    )
    
    return parser.parse_args()


def parse_page_range(page_range: str, total_pages: int) -> List[int]:
    """
    Parse page range string into list of page indices
    
    Args:
        page_range: String like "1-5" or "1,3,5"
        total_pages: Total number of pages in document
        
    Returns:
        List of 0-indexed page numbers
    """
    if not page_range:
        return list(range(total_pages))
    
    pages = set()
    
    for part in page_range.split(','):
        if '-' in part:
            start, end = part.split('-')
            start = int(start.strip())
            end = int(end.strip())
            pages.update(range(start - 1, min(end, total_pages)))
        else:
            page = int(part.strip())
            if 1 <= page <= total_pages:
                pages.add(page - 1)
    
    return sorted(list(pages))


def save_results_json(results: Dict, output_path: str):
    """
    Save results to JSON format (without numpy arrays)
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    # Convert to JSON-serializable format
    json_results = {
        'pdf_path': results['pdf_path'],
        'num_pages': results['num_pages'],
        'pages': []
    }
    
    for page in results['pages']:
        page_data = {
            'page_num': page['page_num'],
            'regions': []
        }
        
        for region in page['regions']:
            region_data = {
                'type': region['type'],
                'name': region['name'],
                'bbox': region['bbox'],
                'embedding': region['embedding'].tolist(),
                'embedding_dim': region['embedding_dim']
            }
            page_data['regions'].append(region_data)
        
        json_results['pages'].append(page_data)
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_path} (JSON format)")


def main():
    """Main execution function"""
    args = parse_args()
    
    # Check if PDF exists
    if not Path(args.pdf_path).exists():
        print(f"Error: PDF file '{args.pdf_path}' not found!")
        sys.exit(1)
    
    # Set device
    if args.device == 'auto':
        device = None
    else:
        device = torch.device(args.device)
    
    # Initialize embedder
    print(f"Initializing DINOv2 embedder with {args.model}...")
    embedder = PDFVisualEmbedder(model_name=args.model, device=device)
    
    # Convert PDF to images first to determine page count
    print(f"\nConverting PDF: {args.pdf_path}")
    all_images = embedder.pdf_to_images(args.pdf_path, dpi=args.dpi)
    total_pages = len(all_images)
    
    # Parse page range
    page_indices = parse_page_range(args.pages, total_pages)
    
    if args.pages:
        print(f"Processing pages: {[p+1 for p in page_indices]} of {total_pages}")
    else:
        print(f"Processing all {total_pages} pages")
    
    # Filter images based on page range
    images = [all_images[i] for i in page_indices]
    
    # Process pages
    results = {
        'pdf_path': args.pdf_path,
        'num_pages': len(images),
        'pages': []
    }
    
    for page_idx, image in enumerate(images):
        actual_page_num = page_indices[page_idx] + 1
        print(f"\nProcessing page {actual_page_num}...")
        
        # Detect regions
        regions = embedder.detect_regions(image)
        
        page_data = {
            'page_num': actual_page_num,
            'regions': []
        }
        
        # Extract embeddings for each region
        for region in regions:
            if args.verbose:
                print(f"  Extracting {region['type']}: {region['name']}")
            
            # Crop region
            region_img = embedder.extract_region(image, region['bbox'])
            
            # Get embedding
            embedding = embedder.get_embedding(region_img)
            
            page_data['regions'].append({
                'type': region['type'],
                'name': region['name'],
                'bbox': region['bbox'],
                'embedding': embedding,
                'embedding_dim': len(embedding)
            })
        
        results['pages'].append(page_data)
    
    # Save embeddings
    if not args.no_save:
        if args.format == 'json':
            output_path = args.output.replace('.pkl', '.json')
            save_results_json(results, output_path)
        else:
            embedder.save_embeddings(results, args.output)
    
    # Visualize regions
    if args.visualize and len(images) > 0:
        print("\nGenerating visualization...")
        regions = embedder.detect_regions(images[0])
        embedder.visualize_regions(images[0], regions, args.viz_output)
    
    # Compare regions
    if args.compare_regions:
        print("\n" + "="*60)
        print("Region Similarity Analysis")
        print("="*60)
        
        for page in results['pages']:
            if len(page['regions']) < 2:
                continue
                
            print(f"\nPage {page['page_num']}:")
            regions = page['regions']
            
            # Compare all pairs
            for i in range(len(regions)):
                for j in range(i + 1, len(regions)):
                    emb1 = regions[i]['embedding']
                    emb2 = regions[j]['embedding']
                    similarity = embedder.compute_similarity(emb1, emb2)
                    
                    print(f"  {regions[i]['name']} <-> {regions[j]['name']}: {similarity:.4f}")
    
    # Print summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"PDF: {args.pdf_path}")
    print(f"Pages processed: {results['num_pages']}")
    print(f"Model: {args.model}")
    print(f"DPI: {args.dpi}")
    if results['pages']:
        print(f"Embedding dimension: {results['pages'][0]['regions'][0]['embedding_dim']}")
        total_regions = sum(len(p['regions']) for p in results['pages'])
        print(f"Total regions extracted: {total_regions}")
    if not args.no_save:
        print(f"Embeddings saved to: {args.output}")
    print("="*60)
    print("\n✓ Processing complete!")


# Example usage
if __name__ == "__main__":
    main()