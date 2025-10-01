#!/usr/bin/env python3
"""
PDF Document Embedding Extractor using DinoV2

This script processes PDF documents by:
1. Converting each page to an image
2. Applying DinoV2 embedding extraction
3. Generating PCA visualizations for each page
4. Saving results in the 'results' folder

Dependencies:
- torch>=2.8.0
- torchvision>=0.23.0
- einops==0.7.0
- PyMuPDF
- Pillow
"""

import os
import sys
import argparse
import fitz  # PyMuPDF
import torch
from PIL import Image
from einops import rearrange
from torchvision.transforms import Normalize
from torchvision.transforms.functional import resize, to_tensor
from torchvision.utils import save_image
from torchvision.io.image import read_image, ImageReadMode
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DinoV2EmbeddingExtractor:
    """DinoV2 Embedding Extractor for document pages"""
    
    def __init__(self, device='auto'):
        """Initialize the DinoV2 model and preprocessing
        
        Args:
            device: Device to run the model on ('auto', 'cpu', 'cuda')
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Load DinoV2 model
        logger.info("Loading DinoV2 model...")
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.dinov2.to(self.device)
        self.dinov2.eval()
        
        # ImageNet normalization parameters
        self.imagenet_mean = (0.485, 0.456, 0.406)
        self.imagenet_std = (0.229, 0.224, 0.225)
        self.norm = Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        
        # Standard image size for DinoV2
        self.image_size = (672, 672)
        
    def minmax_norm(self, x):
        """Min-max normalization"""
        return (x - x.min(0).values) / (x.max(0).values - x.min(0).values)
    
    def extract_embeddings(self, image_tensor):
        """Extract DinoV2 embeddings and generate PCA visualization
        
        Args:
            image_tensor: RGB image tensor of shape (C, H, W)
            
        Returns:
            PCA visualization tensor of shape (C, H, W)
        """
        with torch.no_grad():
            # Ensure proper shape and device
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            image_tensor = image_tensor.to(self.device)
            
            # Resize to standard size
            H, W = self.image_size
            image_resized = resize(image_tensor, (H, W))
            
            # Normalize
            image_norm = self.norm(image_resized / 255.0)
            
            # Extract features
            features = self.dinov2.forward_features(image_norm)
            E_patch = features["x_norm_patchtokens"]
            
            # Reshape for PCA
            E_patch_norm = rearrange(E_patch, "B L E -> (B L) E")
            
            # First PCA to separate foreground/background
            _, _, V = torch.pca_lowrank(E_patch_norm)
            E_pca_1 = torch.matmul(E_patch_norm, V[:, :1])
            E_pca_1_norm = self.minmax_norm(E_pca_1)
            
            # Create foreground mask
            M_fg = E_pca_1_norm.squeeze() > 0.5
            
            if M_fg.sum() > 0:  # If we have foreground pixels
                # Second PCA for foreground pixels
                _, _, V_fg = torch.pca_lowrank(E_patch_norm[M_fg])
                E_pca_3_fg = torch.matmul(E_patch_norm[M_fg], V_fg[:, :3])
                E_pca_3_fg = self.minmax_norm(E_pca_3_fg)
                
                # Create visualization
                B, L, _ = E_patch.shape
                Z = B * L
                I_draw = torch.zeros(Z, 3, device=self.device)
                I_draw[M_fg] = E_pca_3_fg
                
                I_draw = rearrange(I_draw, "(B L) C -> B L C", B=B)
                I_draw = rearrange(I_draw, "B (h w) C -> B h w C", h=H//14, w=W//14)
                
                # Get first image and convert to channel-first format
                pca_image = I_draw[0]
                pca_image = rearrange(pca_image, "H W C -> C H W")
                
                # Resize back to original size
                pca_image = resize(pca_image, (H, W))
            else:
                # If no foreground detected, return zeros
                pca_image = torch.zeros(3, H, W, device=self.device)
                
        return pca_image.cpu()

def pdf_to_images(pdf_path, output_dir):
    """Convert PDF pages to images
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save page images
        
    Returns:
        List of image file paths
    """
    logger.info(f"Converting PDF to images: {pdf_path}")
    
    # Open PDF
    doc = fitz.open(pdf_path)
    image_paths = []
    
    # Create pages directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Convert to image (300 DPI for good quality)
        mat = fitz.Matrix(300/72, 300/72)  # 300 DPI scale
        pix = page.get_pixmap(matrix=mat)
        
        # Save as PNG
        image_path = os.path.join(output_dir, f"page_{page_num + 1:03d}.png")
        pix.save(image_path)
        image_paths.append(image_path)
        
        logger.info(f"Saved page {page_num + 1} as {image_path}")
    
    doc.close()
    logger.info(f"Converted {len(image_paths)} pages to images")
    
    return image_paths

def process_document(pdf_path, pages_dir="pages", results_dir="results", device='auto'):
    """Process a PDF document and extract embeddings for each page
    
    Args:
        pdf_path: Path to the PDF file
        pages_dir: Directory to save page images
        results_dir: Directory to save embedding visualizations
        device: Device to run the model on
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Initialize extractor
    extractor = DinoV2EmbeddingExtractor(device=device)
    
    # Convert PDF to images
    image_paths = pdf_to_images(pdf_path, pages_dir)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Process each page
    for i, image_path in enumerate(image_paths):
        logger.info(f"Processing page {i + 1}/{len(image_paths)}: {image_path}")
        
        try:
            # Load image
            image_tensor = read_image(image_path, ImageReadMode.RGB)
            
            # Extract embeddings and generate visualization
            pca_image = extractor.extract_embeddings(image_tensor)
            
            # Save result
            result_path = os.path.join(results_dir, f"page_{i + 1:03d}_embedding.png")
            save_image(pca_image, result_path)
            
            logger.info(f"Saved embedding visualization: {result_path}")
            
        except Exception as e:
            logger.error(f"Error processing page {i + 1}: {str(e)}")
            continue
    
    logger.info("Document processing completed!")

def main():
    parser = argparse.ArgumentParser(description="Extract DinoV2 embeddings from PDF documents")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--pages-dir", default="pages", help="Directory to save page images (default: pages)")
    parser.add_argument("--results-dir", default="results", help="Directory to save results (default: results)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], 
                       help="Device to run the model on (default: auto)")
    
    args = parser.parse_args()
    
    try:
        process_document(args.pdf_path, args.pages_dir, args.results_dir, args.device)
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()