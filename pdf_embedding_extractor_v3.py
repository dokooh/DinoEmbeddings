#!/usr/bin/env python3
"""
PDF Document Embedding Extractor using DinoV3

This script processes PDF documents by:
1. Converting each page to an image
2. Applying DinoV3 embedding extraction from Hugging Face Transformers
3. Generating PCA visualizations for each page
4. Saving results in the 'results' folder

Dependencies:
- torch>=2.8.0
- torchvision>=0.23.0
- transformers>=4.40.0
- einops==0.7.0
- PyMuPDF
- Pillow
"""

import os
import sys
import argparse
import io  # Added missing import
import fitz  # PyMuPDF
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms.functional import resize
from torchvision.utils import save_image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DinoV3EmbeddingExtractor:
    """DinoV3 Embedding Extractor for document pages using Hugging Face Transformers"""
    
    def __init__(self, model_name="facebook/dinov3-vits16-pretrain-lvd1689m", device='auto'):
        """Initialize the DinoV3 model and processor
        
        Args:
            model_name: DinoV3 model name from Hugging Face
            device: Device to run the model on ('auto', 'cpu', 'cuda')
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Load DinoV3 model and processor with fallback to DinoV2
        try:
            logger.info(f"Attempting to load DinoV3 model: {model_name}")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.dinov3 = AutoModel.from_pretrained(model_name)
            self.dinov3.to(self.device)
            self.dinov3.eval()
            logger.info(f"✅ DinoV3 model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️  DinoV3 model access restricted: {str(e)[:100]}...")
            logger.info("🔄 Falling back to DinoV2 with DinoV3-compatible API...")
            fallback_model = "facebook/dinov2-small"
            self.processor = AutoImageProcessor.from_pretrained(fallback_model)
            self.dinov3 = AutoModel.from_pretrained(fallback_model)
            self.dinov3.to(self.device)
            self.dinov3.eval()
            logger.info(f"✅ DinoV2 model loaded as fallback")
        
        self.model_name = model_name
        self.patch_size = self.dinov3.config.patch_size
        self.num_register_tokens = getattr(self.dinov3.config, 'num_register_tokens', 0)
        
        logger.info(f"Model loaded - Patch size: {self.patch_size}, Register tokens: {self.num_register_tokens}")
        
    def minmax_norm(self, x):
        """Min-max normalization"""
        return (x - x.min(0).values) / (x.max(0).values - x.min(0).values)
    
    def extract_embeddings(self, image_pil):
        """Extract DinoV3 embeddings and generate PCA visualization
        
        Args:
            image_pil: PIL Image
            
        Returns:
            PCA visualization tensor of shape (C, H, W)
        """
        with torch.no_grad():
            # Process image with the processor
            inputs = self.processor(images=image_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            outputs = self.dinov3(**inputs)
            
            # Get patch embeddings (excluding CLS token and register tokens)
            last_hidden_states = outputs.last_hidden_state
            batch_size, seq_len, hidden_size = last_hidden_states.shape
            
            # Skip CLS token (first) and register tokens (next num_register_tokens)
            E_patch = last_hidden_states[:, 1 + self.num_register_tokens:, :]
            
            # Calculate patch grid dimensions
            if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                img_height, img_width = inputs.pixel_values.shape[-2:]
            elif 'pixel_values' in inputs:
                img_height, img_width = inputs['pixel_values'].shape[-2:]
            else:
                # Default to common input size for DinoV2
                img_height, img_width = 224, 224
            num_patches_height = img_height // self.patch_size
            num_patches_width = img_width // self.patch_size
            
            logger.debug(f"Image size: {img_height}x{img_width}, Patches: {num_patches_height}x{num_patches_width}")
            
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
                I_draw = rearrange(I_draw, "B (h w) C -> B h w C", h=num_patches_height, w=num_patches_width)
                
                # Get first image and convert to channel-first format
                pca_image = I_draw[0]
                pca_image = rearrange(pca_image, "H W C -> C H W")
                
                # Resize to standard size for visualization
                pca_image = resize(pca_image, (img_height, img_width))
            else:
                # If no foreground detected, return zeros
                pca_image = torch.zeros(3, img_height, img_width, device=self.device)
                
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

def process_document(pdf_path, pages_dir="pages", results_dir="results", 
                    model_name="dinov2_vitb14", device='auto'):
    """Process a PDF document and extract DinoV2 embeddings for each page
    
    Args:
        pdf_path: Path to the PDF file
        pages_dir: Directory to save page images
        results_dir: Directory to save embedding visualizations
        model_name: DinoV2 model name (dinov2_vitb14, dinov2_vits14, dinov2_vitl14)
        device: Device to run the model on
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Initialize extractor
    extractor = DinoV3EmbeddingExtractor(model_name=model_name, device=device)
    
    # Convert PDF to images
    image_paths = pdf_to_images(pdf_path, pages_dir)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Process each page
    for i, image_path in enumerate(image_paths):
        logger.info(f"Processing page {i + 1}/{len(image_paths)}: {image_path}")
        
        try:
            # Load image as PIL Image
            image_pil = Image.open(image_path).convert('RGB')
            
            # Extract embeddings and generate visualization
            pca_image = extractor.extract_embeddings(image_pil)
            
            # Save result
            result_path = os.path.join(results_dir, f"page_{i + 1:03d}_dinov2_embedding.png")
            save_image(pca_image, result_path)
            
            logger.info(f"Saved DinoV2 embedding visualization: {result_path}")
            
        except Exception as e:
            logger.error(f"Error processing page {i + 1}: {str(e)}")
            continue
    
    logger.info("Document processing with DinoV2 completed!")

def main():
    parser = argparse.ArgumentParser(description="Extract DinoV2 embeddings from PDF documents (DinoV3 alternative)")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--pages-dir", default="pages", help="Directory to save page images (default: pages)")
    parser.add_argument("--results-dir", default="results", help="Directory to save results (default: results)")
    parser.add_argument("--model", default="dinov2_vitb14", 
                       help="DinoV2 model name (dinov2_vitb14, dinov2_vits14, dinov2_vitl14)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], 
                       help="Device to run the model on (default: auto)")
    
    args = parser.parse_args()
    
    try:
        process_document(args.pdf_path, args.pages_dir, args.results_dir, args.model, args.device)
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()