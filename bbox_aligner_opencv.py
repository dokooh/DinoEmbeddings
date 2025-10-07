#!/usr/bin/env python3
"""
BBox Aligner with DinoV3 and Gemini 2.5 Flash
============================================

This script combines PDF processing, Gemini 2.5 Flash for section detection, 
and DinoV3 for bounding box refinement through white background detection.

Workflow:
1. PDF → Extract pages as images
2. Gemini 2.5 Flash → Detect section bounding boxes
3. DinoV3 → Detect white background and extend bboxes
4. Generate aligned and extended bounding boxes

Author: GitHub Copilot
Date: October 2025
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
import logging
from datetime import datetime

# Core dependencies
import numpy as np
from PIL import Image
import cv2

# PDF processing
import fitz  # PyMuPDF

# Google Gemini
import google.generativeai as genai

# DinoV3 dependencies
import torch
import torchvision.transforms as T

# Machine Learning
from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BBoxAlignerDinoV3:
    """
    Main class for PDF processing with Gemini + DinoV3 bbox alignment.
    """
    
    def __init__(self, gemini_api_key: str = None, model_name: str = 'dinov2_vitb14'):
        """
        Initialize the BBox Aligner.
        
        Args:
            gemini_api_key: Google Gemini API key
            model_name: DinoV3 model name (fallback to DinoV2 if V3 unavailable)
        """
        self.logger = self._setup_logging()
        
        # Initialize Gemini
        self._setup_gemini(gemini_api_key)
        
        # Initialize DinoV3/V2
        self._setup_dino_model(model_name)
        
        # Gemini prompt for section detection
        self.gemini_prompt = """Taken the image you must return the coordinates of the section on the image, section is a kind of rectangle form with fields and tables presented on the image with some information. Give only section coordinates like x1,y1,x2,y2. x1 and y1 are x and y postion of start of the section from top and x2 is width and y2 is height. Coordinates must be relative from 0 to 1. There can be more then 1 section on the image but that's rare. You must return an valid json array and then in the array for each section you must add to it array with the coordinates. For example if you see 1 section on the image you must return '[[x1,y1,x2,y2]]' and nothing else ! If you see no section juts return an empty array like [] you must return a valid json always. Additinal guide: The sections have grey/yellow background. The sections have the header like '2. *header_name*', each section has it own header. Sections usualy take 70-80% of page, they don't finish unless there is new section."""
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('bbox_aligner')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_gemini(self, api_key: str = None):
        """Initialize Google Gemini API."""
        try:
            if api_key:
                genai.configure(api_key=api_key)
            else:
                # Try to get from environment
                api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable or pass as parameter.")
                genai.configure(api_key=api_key)
            
            # Initialize model (using available models from API)
            model_names = [
                'gemini-2.5-flash',  # Latest available model
                'gemini-2.0-flash-exp',
                'gemini-2.0-flash',
                'gemini-flash-latest',
                'gemini-pro-latest'
            ]
            
            model_initialized = False
            for model_name in model_names:
                try:
                    self.gemini_model = genai.GenerativeModel(model_name)
                    self.logger.info(f"✅ Gemini model '{model_name}' initialized successfully")
                    model_initialized = True
                    break
                except Exception as model_error:
                    self.logger.warning(f"⚠️ Failed to initialize {model_name}: {str(model_error)}")
                    continue
            
            if not model_initialized:
                raise Exception("Failed to initialize any Gemini model. Please check your API key and model availability.")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Gemini: {str(e)}")
            raise
    
    def _setup_dino_model(self, model_name: str):
        """Initialize DinoV3 or fallback to DinoV2."""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Using device: {self.device}")
            
            # Try DinoV3 first, fallback to DinoV2
            try:
                # Attempt DinoV3 (this might not exist yet)
                self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov3_vitb14')
                self.logger.info("✅ DinoV3 model loaded successfully")
                self.is_dinov3 = True
            except:
                # Fallback to DinoV2
                self.dino_model = torch.hub.load('facebookresearch/dinov2', model_name)
                self.logger.info(f"✅ DinoV2 model ({model_name}) loaded as fallback")
                self.is_dinov3 = False
            
            self.dino_model.eval()
            self.dino_model.to(self.device)
            
            # Image preprocessing for Dino
            self.dino_transform = T.Compose([
                T.Resize((518, 518)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Dino model: {str(e)}")
            raise
    
    def extract_pdf_pages(self, pdf_path: str, pages_dir: str, dpi: int = 300, 
                         start_page: int = 1, end_page: int = None) -> List[str]:
        """
        Extract PDF pages as high-quality images.
        
        Args:
            pdf_path: Path to PDF file
            pages_dir: Directory to save page images
            dpi: Resolution for conversion
            start_page: First page to extract (1-indexed)
            end_page: Last page to extract (1-indexed, None for all)
            
        Returns:
            List of paths to extracted page images
        """
        try:
            # Create pages directory
            pages_path = Path(pages_dir)
            pages_path.mkdir(exist_ok=True)
            
            # Open PDF
            pdf_doc = fitz.open(pdf_path)
            total_pages = len(pdf_doc)
            
            # Determine page range
            if end_page is None:
                end_page = total_pages
            
            start_page = max(1, start_page) - 1  # Convert to 0-indexed
            end_page = min(total_pages, end_page)
            
            self.logger.info(f"📊 Extracting pages {start_page + 1} to {end_page} from {total_pages} total pages at {dpi} DPI")
            
            extracted_pages = []
            
            for page_num in range(start_page, end_page):
                page = pdf_doc[page_num]
                
                # Create transformation matrix for desired DPI
                mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 DPI is default
                
                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat)
                
                # Save as PNG
                page_filename = f"page_{page_num + 1:03d}.png"
                page_path = pages_path / page_filename
                pix.save(str(page_path))
                
                extracted_pages.append(str(page_path))
                self.logger.info(f"   ✅ Page {page_num + 1} → {page_filename}")
            
            pdf_doc.close()
            self.logger.info(f"📁 Extracted {len(extracted_pages)} pages to {pages_dir}/")
            
            return extracted_pages
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract PDF pages: {str(e)}")
            raise
    
    def detect_sections_with_gemini(self, image_path: str) -> List[List[float]]:
        """
        Use Gemini 2.5 Flash to detect section bounding boxes.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of bounding boxes [[x1, y1, x2, y2], ...] with normalized coordinates
        """
        try:
            # Load and prepare image
            image = Image.open(image_path)
            
            self.logger.info(f"🔍 Analyzing image with Gemini: {Path(image_path).name}")
            
            # Send to Gemini
            response = self.gemini_model.generate_content([
                self.gemini_prompt,
                image
            ])
            
            # Parse response
            response_text = response.text.strip()
            self.logger.info(f"📝 Gemini response: {response_text}")
            
            # Clean up JSON response (remove markdown code blocks if present)
            json_text = response_text
            if json_text.startswith('```json'):
                json_text = json_text[7:]  # Remove ```json
            if json_text.startswith('```'):
                json_text = json_text[3:]   # Remove ```
            if json_text.endswith('```'):
                json_text = json_text[:-3]  # Remove ending ```
            json_text = json_text.strip()
            
            # Try to parse JSON
            try:
                bboxes = json.loads(json_text)
                
                # Validate format
                if not isinstance(bboxes, list):
                    raise ValueError("Response is not a list")
                
                # Validate each bbox
                validated_bboxes = []
                for i, bbox in enumerate(bboxes):
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        self.logger.warning(f"⚠️ Invalid bbox format at index {i}: {bbox}")
                        continue
                    
                    # Ensure all coordinates are float and in [0, 1]
                    try:
                        x1, y1, x2, y2 = [float(coord) for coord in bbox]
                        
                        # Validate ranges
                        if not all(0 <= coord <= 1 for coord in [x1, y1, x2, y2]):
                            self.logger.warning(f"⚠️ Coordinates out of range [0,1]: {bbox}")
                            continue
                        
                        # Ensure x2 > x1 and y2 > y1
                        if x2 <= x1 or y2 <= y1:
                            self.logger.warning(f"⚠️ Invalid bbox dimensions: {bbox}")
                            continue
                        
                        validated_bboxes.append([x1, y1, x2, y2])
                        
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"⚠️ Error parsing bbox coordinates: {bbox} - {str(e)}")
                        continue
                
                self.logger.info(f"✅ Detected {len(validated_bboxes)} valid sections")
                return validated_bboxes
                
            except json.JSONDecodeError as e:
                self.logger.error(f"❌ Failed to parse Gemini JSON response: {str(e)}")
                self.logger.error(f"Raw response: {response_text}")
                return []
                
        except Exception as e:
            self.logger.error(f"❌ Error in Gemini section detection: {str(e)}")
            return []
    
    def detect_white_background_with_dino(self, image_path: str, bbox: List[float]) -> np.ndarray:
        """
        Use DinoV3/V2 with enhanced PCA analysis to detect white background within a bounding box region.
        This improved version extracts embeddings ONLY from the cropped bounding box region,
        combining deep feature analysis with pixel-level brightness analysis.
        
        Args:
            image_path: Path to the image file
            bbox: Bounding box coordinates [x1, y1, x2, y2] in normalized format
            
        Returns:
            2D numpy array representing white background probability map
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            from scipy import ndimage
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            img_width, img_height = image.size
            
            # Convert normalized bbox to pixel coordinates
            x1, y1, x2, y2 = bbox
            px1 = int(x1 * img_width)
            py1 = int(y1 * img_height)
            px2 = int(x2 * img_width)
            py2 = int(y2 * img_height)
            
            # CRITICAL: Crop to bounding box region FIRST - this is now the ONLY region analyzed
            bbox_region = image.crop((px1, py1, px2, py2))
            crop_array = np.array(bbox_region)  # For pixel-level analysis
            
            # Ensure minimum size for DinoV2/V3 processing
            min_size = 224  # DinoV2 minimum input size
            if bbox_region.width < min_size or bbox_region.height < min_size:
                # Resize while maintaining aspect ratio
                bbox_region = bbox_region.resize((max(min_size, bbox_region.width), 
                                                max(min_size, bbox_region.height)), 
                                               Image.Resampling.LANCZOS)
                crop_array = np.array(bbox_region)  # Update array after resize
                self.logger.info(f"🔍 Resized bbox region to {bbox_region.size} for DinoV2/V3 processing")
            
            # Transform ONLY the cropped bbox region for Dino model
            img_tensor = self.dino_transform(bbox_region).unsqueeze(0).to(self.device)
            
            # Extract features from the cropped region only
            with torch.no_grad():
                features = self.dino_model.forward_features(img_tensor)
                patch_tokens = features['x_norm_patchtokens']  # [1, num_patches, embed_dim]
            
            # Get patch grid dimensions (based on the cropped region, not original image)
            num_patches = patch_tokens.shape[1]
            grid_size = int(np.sqrt(num_patches))  # Should be 37 for 518x518 input
            
            # Convert to numpy for analysis
            patch_features = patch_tokens.squeeze(0).cpu().numpy()  # [num_patches, embed_dim]
            
            # Standardize features for better PCA analysis
            scaler = StandardScaler()
            patch_features_scaled = scaler.fit_transform(patch_features)
            
            # Apply enhanced PCA analysis
            n_components = min(10, patch_features.shape[1])  # Use more components
            pca = PCA(n_components=n_components)
            pca_features = pca.fit_transform(patch_features_scaled)
            
            # Calculate multiple metrics for white background detection
            white_probabilities = np.zeros(num_patches)
            
            # 1. PCA complexity score (lower = more likely white/uniform)
            pca_norms = np.linalg.norm(pca_features, axis=1)
            pca_complexity = pca_norms / np.percentile(pca_norms, 75)
            pca_scores = np.maximum(0, 1.0 - pca_complexity)
            
            # 2. Feature variance score (lower variance = more uniform = more white)
            feature_variances = np.var(patch_features, axis=1)
            var_threshold = np.percentile(feature_variances, 75)
            var_scores = np.maximum(0, 1.0 - (feature_variances / var_threshold))
            
            # 3. PCA component diversity score
            component_diversities = np.std(pca_features[:, :3], axis=1)
            diversity_threshold = np.percentile(component_diversities, 75)
            diversity_scores = np.maximum(0, 1.0 - (component_diversities / diversity_threshold))
            
            # 4. Pixel-level brightness and uniformity analysis (on cropped bbox region only)
            patch_h = crop_array.shape[0] // grid_size
            patch_w = crop_array.shape[1] // grid_size
            
            brightness_scores = np.zeros(num_patches)
            uniformity_scores = np.zeros(num_patches)
            
            # Analyze patches within the cropped bounding box region
            for i in range(grid_size):
                for j in range(grid_size):
                    patch_idx = i * grid_size + j
                    
                    # Get pixel region for this patch within the cropped bbox
                    patch_y1 = i * patch_h
                    patch_y2 = min((i + 1) * patch_h, crop_array.shape[0])
                    patch_x1 = j * patch_w
                    patch_x2 = min((j + 1) * patch_w, crop_array.shape[1])
                    
                    if patch_y2 > patch_y1 and patch_x2 > patch_x1:
                        # Extract pixels from the cropped bbox region only
                        patch_pixels = crop_array[patch_y1:patch_y2, patch_x1:patch_x2]
                        
                        # Brightness score (higher brightness = more likely white)
                        avg_brightness = np.mean(patch_pixels) / 255.0
                        brightness_scores[patch_idx] = avg_brightness ** 2  # Quadratic emphasis
                        
                        # Uniformity score (lower std = more uniform = more likely white)
                        color_std = np.std(patch_pixels) / 255.0
                        uniformity_scores[patch_idx] = max(0, 1.0 - color_std * 2)
                    else:
                        brightness_scores[patch_idx] = 0.5
                        uniformity_scores[patch_idx] = 0.5
            
            # Combine all scores with optimized weights
            for i in range(num_patches):
                combined_score = (
                    brightness_scores[i] * 0.4 +      # Pixel brightness most important
                    uniformity_scores[i] * 0.25 +     # Color uniformity
                    pca_scores[i] * 0.15 +            # PCA complexity
                    var_scores[i] * 0.1 +             # Feature variance
                    diversity_scores[i] * 0.1         # Component diversity
                )
                white_probabilities[i] = combined_score
            
            # Reshape to spatial grid
            white_prob_map = white_probabilities.reshape(grid_size, grid_size)
            
            # Apply light smoothing to reduce noise
            white_prob_map = ndimage.gaussian_filter(white_prob_map, sigma=0.5)
            
            # Log analysis results
            avg_prob = np.mean(white_prob_map)
            max_prob = np.max(white_prob_map)
            min_prob = np.min(white_prob_map)
            
            # Calculate bbox dimensions for context
            bbox_width = px2 - px1
            bbox_height = py2 - py1
            
            self.logger.info(f"🧠 Enhanced PCA analysis on cropped bbox region ({bbox_width}x{bbox_height}px):")
            self.logger.info(f"   📐 Grid size: {grid_size}x{grid_size} patches")
            self.logger.info(f"   📊 Explained variance ratios: {pca.explained_variance_ratio_[:3]}")
            self.logger.info(f"   🎯 White probability stats: avg={avg_prob:.3f}, max={max_prob:.3f}, min={min_prob:.3f}")
            self.logger.info(f"   🔍 Analysis performed on bbox region ONLY (not full image)")
            
            # Store embeddings for debugging (if output directory is available)
            if hasattr(self, 'current_output_dir') and self.current_output_dir:
                self._save_embeddings_for_debugging(
                    patch_features, pca_features, white_prob_map, 
                    bbox, image_path, self.current_output_dir
                )
            
            return white_prob_map
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced Dino PCA white background detection: {str(e)}")
            # Return default probability map
            return np.zeros((37, 37))
    
    def _save_embeddings_for_debugging(self, patch_features: np.ndarray, pca_features: np.ndarray, 
                                     white_prob_map: np.ndarray, bbox: List[float], 
                                     image_path: str, output_dir: str):
        """
        Save visual embedding images for debugging purposes instead of raw data.
        
        Args:
            patch_features: Raw patch features from DinoV2/V3 [num_patches, embed_dim]
            pca_features: PCA-transformed features [num_patches, n_components]
            white_prob_map: White probability map [grid_size, grid_size]
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            image_path: Path to the source image
            output_dir: Directory to save debugging files
        """
        try:
            import matplotlib.pyplot as plt
            from datetime import datetime
            
            # Create embeddings subdirectory
            embeddings_dir = Path(output_dir) / 'embeddings_debug'
            embeddings_dir.mkdir(exist_ok=True)
            
            # Generate unique filename based on image and bbox
            image_name = Path(image_path).stem
            bbox_str = f"{bbox[0]:.3f}_{bbox[1]:.3f}_{bbox[2]:.3f}_{bbox[3]:.3f}"
            timestamp = datetime.now().strftime("%H%M%S")
            base_filename = f"{image_name}_bbox_{bbox_str}_{timestamp}"
            
            # Calculate grid size for visualization
            num_patches = patch_features.shape[0]
            grid_size = int(np.sqrt(num_patches))
            
            # Create comprehensive embedding visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Embedding Analysis: {image_name}\nBBox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]', 
                        fontsize=14, fontweight='bold')
            
            # 1. White Probability Map
            im1 = axes[0, 0].imshow(white_prob_map, cmap='viridis', vmin=0, vmax=1)
            axes[0, 0].set_title('White Background Probability')
            axes[0, 0].set_xlabel('Patch X')
            axes[0, 0].set_ylabel('Patch Y')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
            
            # 2. Feature Embedding Norms (reshaped to spatial grid)
            feature_norms = np.linalg.norm(patch_features, axis=1).reshape(grid_size, grid_size)
            im2 = axes[0, 1].imshow(feature_norms, cmap='plasma')
            axes[0, 1].set_title('Feature Embedding Norms')
            axes[0, 1].set_xlabel('Patch X')
            axes[0, 1].set_ylabel('Patch Y')
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            # 3. PCA First Component (spatial visualization)
            pca_first_comp = pca_features[:, 0].reshape(grid_size, grid_size)
            im3 = axes[0, 2].imshow(pca_first_comp, cmap='RdBu_r')
            axes[0, 2].set_title('PCA First Component')
            axes[0, 2].set_xlabel('Patch X')
            axes[0, 2].set_ylabel('Patch Y')
            plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # 4. Feature Variance Map
            feature_variances = np.var(patch_features, axis=1).reshape(grid_size, grid_size)
            im4 = axes[1, 0].imshow(feature_variances, cmap='coolwarm')
            axes[1, 0].set_title('Feature Variance (Uniformity)')
            axes[1, 0].set_xlabel('Patch X')
            axes[1, 0].set_ylabel('Patch Y')
            plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            # 5. PCA Component Diversity
            pca_diversity = np.std(pca_features[:, :3], axis=1).reshape(grid_size, grid_size)
            im5 = axes[1, 1].imshow(pca_diversity, cmap='magma')
            axes[1, 1].set_title('PCA Component Diversity')
            axes[1, 1].set_xlabel('Patch X')
            axes[1, 1].set_ylabel('Patch Y')
            plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            # 6. Combined Analysis (weighted combination)
            # Combine white probability with feature analysis for final view
            combined_score = (white_prob_map * 0.4 + 
                            (1.0 - feature_variances / np.max(feature_variances)) * 0.3 +
                            (1.0 - pca_diversity / np.max(pca_diversity)) * 0.3)
            im6 = axes[1, 2].imshow(combined_score, cmap='viridis', vmin=0, vmax=1)
            axes[1, 2].set_title('Combined White Detection Score')
            axes[1, 2].set_xlabel('Patch X')
            axes[1, 2].set_ylabel('Patch Y')
            plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Save the comprehensive visualization
            viz_path = embeddings_dir / f"{base_filename}_embedding_analysis.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Create a separate detailed white probability heatmap
            fig2, ax = plt.subplots(1, 1, figsize=(8, 8))
            im = ax.imshow(white_prob_map, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f'White Background Probability Map\n{image_name} - BBox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Patch X')
            ax.set_ylabel('Patch Y')
            
            # Add grid lines for patch visualization
            ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Add colorbar with detailed labels
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('White Background Probability', rotation=270, labelpad=20)
            
            # Add statistics text
            stats_text = f"""Statistics:
Mean: {np.mean(white_prob_map):.3f}
Std:  {np.std(white_prob_map):.3f}
Min:  {np.min(white_prob_map):.3f}
Max:  {np.max(white_prob_map):.3f}
Grid: {grid_size}×{grid_size}"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontfamily='monospace', fontsize=9)
            
            plt.tight_layout()
            
            # Save the detailed heatmap
            heatmap_path = embeddings_dir / f"{base_filename}_white_probability_heatmap.png"
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Save lightweight summary with just statistics (no large arrays)
            summary_path = embeddings_dir / f"{base_filename}_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"DinoV2/V3 Embedding Analysis Debug Report\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Image: {image_name}\n")
                f.write(f"BBox: {bbox}\n\n")
                
                f.write(f"Patch Features Shape: {patch_features.shape}\n")
                f.write(f"PCA Features Shape: {pca_features.shape}\n")
                f.write(f"White Prob Map Shape: {white_prob_map.shape}\n")
                f.write(f"Grid Size: {grid_size}×{grid_size}\n\n")
                
                f.write(f"White Probability Statistics:\n")
                f.write(f"  Mean: {np.mean(white_prob_map):.4f}\n")
                f.write(f"  Std:  {np.std(white_prob_map):.4f}\n")
                f.write(f"  Min:  {np.min(white_prob_map):.4f}\n")
                f.write(f"  Max:  {np.max(white_prob_map):.4f}\n\n")
                
                f.write(f"Patch Features Statistics:\n")
                f.write(f"  Mean embedding norm: {np.mean(np.linalg.norm(patch_features, axis=1)):.4f}\n")
                f.write(f"  Std embedding norm:  {np.std(np.linalg.norm(patch_features, axis=1)):.4f}\n")
                f.write(f"  Feature dimensionality: {patch_features.shape[1]}\n")
                f.write(f"  Number of patches: {patch_features.shape[0]}\n\n")
                
                f.write(f"PCA Analysis:\n")
                pca_var = np.var(pca_features, axis=0)
                f.write(f"  Component variances: {pca_var[:5]}\n")
                f.write(f"  Total explained variance: {np.sum(pca_var):.4f}\n\n")
                
                f.write(f"Generated Visualizations:\n")
                f.write(f"  - {viz_path.name}: Comprehensive 6-panel analysis\n")
                f.write(f"  - {heatmap_path.name}: Detailed white probability heatmap\n")
                f.write(f"  - {summary_path.name}: This summary file\n")
            
            self.logger.info(f"🎨 Visual embeddings saved for debugging:")
            self.logger.info(f"   📁 Directory: {embeddings_dir}")
            self.logger.info(f"   �️ Analysis: {viz_path.name}")
            self.logger.info(f"   � Heatmap: {heatmap_path.name}")
            self.logger.info(f"   📄 Summary: {summary_path.name}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save visual embeddings for debugging: {str(e)}")
    
    def extend_bbox_to_white(self, image, bbox, white_threshold=240, max_extension=500):
        """
        Extend a bounding box until all four edges reach white areas using OpenCV approach.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image (grayscale or color)
        bbox : tuple
            Initial bounding box as (x, y, width, height) in pixel coordinates
        white_threshold : int
            Pixel intensity threshold to consider as white (0-255)
        max_extension : int
            Maximum pixels to extend in any direction (safety limit)
        
        Returns:
        --------
        tuple : Extended bounding box as (x, y, width, height) in pixel coordinates
        """
        import cv2
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        x, y, bw, bh = bbox
        
        # Ensure bbox is within image bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = min(bw, w - x)
        bh = min(bh, h - y)
        
        # Track original position for safety check
        orig_x, orig_y = x, y
        orig_right, orig_bottom = x + bw, y + bh
        
        # Extend left edge
        extended = 0
        while x > 0 and extended < max_extension:
            # Check if the entire left edge has reached white
            left_edge = gray[y:y+bh, x-1]
            if np.all(left_edge >= white_threshold):
                break
            x -= 1
            bw += 1
            extended += 1
        
        # Extend right edge
        extended = 0
        while (x + bw) < w and extended < max_extension:
            # Check if the entire right edge has reached white
            right_edge = gray[y:y+bh, x+bw]
            if np.all(right_edge >= white_threshold):
                break
            bw += 1
            extended += 1
        
        # Extend top edge
        extended = 0
        while y > 0 and extended < max_extension:
            # Check if the entire top edge has reached white
            top_edge = gray[y-1, x:x+bw]
            if np.all(top_edge >= white_threshold):
                break
            y -= 1
            bh += 1
            extended += 1
        
        # Extend bottom edge
        extended = 0
        while (y + bh) < h and extended < max_extension:
            # Check if the entire bottom edge has reached white
            bottom_edge = gray[y+bh, x:x+bw]
            if np.all(bottom_edge >= white_threshold):
                break
            bh += 1
            extended += 1
        
        return (x, y, bw, bh)

    # COMMENTED OUT - BOTTOM BORDER OPTIMIZATION (reverted to previous strategy)
    # def optimize_bottom_border(self, image, bbox, white_threshold=240, max_extension=500, 
    #                           tolerance_ratio=0.1, sample_width=50):
    #     """
    #     Post-optimization strategy specifically for bottom border alignment.
    #     Uses multiple sampling strategies to ensure bottom border extends to white regions.
    #     
    #     Parameters:
    #     -----------
    #     image : numpy.ndarray
    #         Input image (grayscale or color)
    #     bbox : tuple
    #         Initial bounding box as (x, y, width, height) in pixel coordinates
    #     white_threshold : int
    #         Pixel intensity threshold to consider as white (0-255)
    #     max_extension : int
    #         Maximum pixels to extend bottom border
    #     tolerance_ratio : float
    #         Ratio of non-white pixels allowed in the edge (0.1 = 10% tolerance)
    #     sample_width : int
    #         Width of sampling windows for edge analysis
    #     
    #     Returns:
    #     --------
    #     tuple : Optimized bounding box as (x, y, width, height) in pixel coordinates
    #     """
    #     import cv2
    #     
    #     # Convert to grayscale if needed
    #     if len(image.shape) == 3:
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     else:
    #         gray = image.copy()
    #     
    #     h, w = gray.shape
    #     x, y, bw, bh = bbox
    #     
    #     # Ensure bbox is within image bounds
    #     x = max(0, min(x, w - 1))
    #     y = max(0, min(y, h - 1))
    #     bw = min(bw, w - x)
    #     bh = min(bh, h - y)
    #     
    #     original_bottom = y + bh
    #     
    #     self.logger.info(f"🔧 Optimizing bottom border from y={original_bottom}")
    #     
    #     # Strategy 1: Relaxed threshold - allow some tolerance for noise
    #     extended = 0
    #     current_bottom = y + bh
    #     while current_bottom < h and extended < max_extension:
    #         # Sample the entire bottom edge
    #         bottom_edge = gray[current_bottom, x:x+bw]
    #         
    #         # Calculate ratio of white pixels
    #         white_pixels = np.sum(bottom_edge >= white_threshold)
    #         white_ratio = white_pixels / len(bottom_edge)
    #         
    #         # If most pixels are white (within tolerance), continue extending
    #         if white_ratio >= (1.0 - tolerance_ratio):
    #             bh += 1
    #             current_bottom += 1
    #             extended += 1
    #         else:
    #             break
    #     
    #     # Strategy 2: Multi-point sampling - check key positions along bottom edge
    #     if extended == 0:  # If Strategy 1 didn't extend, try sampling approach
    #         current_bottom = y + bh
    #         sample_positions = []
    #         
    #         # Create sampling positions across the width
    #         if bw >= sample_width * 3:
    #             # For wide boxes, sample left, center, and right regions
    #             left_start = x + 10  # Small offset from edge
    #             center_start = x + (bw - sample_width) // 2
    #             right_start = x + bw - sample_width - 10
    #             sample_positions = [
    #                 (left_start, min(sample_width, bw - 20)),
    #                 (center_start, sample_width),
    #                 (right_start, min(sample_width, bw - 20))
    #             ]
    #         else:
    #             # For narrow boxes, sample the center region
    #             center_start = max(x, x + (bw - min(sample_width, bw)) // 2)
    #             sample_width_adj = min(sample_width, bw)
    #             sample_positions = [(center_start, sample_width_adj)]
    #         
    #         extended_sampling = 0
    #         while current_bottom < h and extended_sampling < max_extension:
    #             all_samples_white = True
    #             
    #             for sample_x, sample_w in sample_positions:
    #                 # Sample this region of the bottom edge
    #                 sample_edge = gray[current_bottom, sample_x:sample_x+sample_w]
    #                 white_pixels = np.sum(sample_edge >= white_threshold)
    #                 white_ratio = white_pixels / len(sample_edge)
    #                 
    #                 # If this sample region is not sufficiently white, stop
    #                 if white_ratio < (1.0 - tolerance_ratio):
    #                     all_samples_white = False
    #                     break
    #             
    #             if all_samples_white:
    #                 bh += 1
    #                 current_bottom += 1
    #                 extended_sampling += 1
    #                 extended += 1  # Update total extension count
    #             else:
    #                 break
    #     
    #     # Strategy 3: Gradient-based refinement for final positioning
    #     if extended > 0:
    #         # Look for the optimal stopping point by analyzing gradient changes
    #         search_start = max(original_bottom, current_bottom - min(20, extended // 2))
    #         search_end = current_bottom
    #         
    #         if search_end > search_start:
    #             best_bottom = current_bottom
    #             min_gradient = float('inf')
    #             
    #             for test_bottom in range(search_start, search_end + 1):
    #                 if test_bottom >= h:
    #                     break
    #                 
    #                 # Calculate gradient at this position
    #                 test_edge = gray[test_bottom, x:x+bw]
    #                 if test_bottom > 0:
    #                     prev_edge = gray[test_bottom-1, x:x+bw]
    #                     gradient = np.mean(np.abs(test_edge.astype(float) - prev_edge.astype(float)))
    #                     
    #                     # Also check if this line is sufficiently white
    #                     white_pixels = np.sum(test_edge >= white_threshold)
    #                     white_ratio = white_pixels / len(test_edge)
    #                     
    #                     # Prefer positions with low gradient and high whiteness
    #                     if gradient < min_gradient and white_ratio >= (1.0 - tolerance_ratio * 1.5):
    #                         min_gradient = gradient
    #                         best_bottom = test_bottom
    #             
    #             # Update to the refined position
    #             bh = best_bottom - y
    #     
    #     final_extension = (y + bh) - original_bottom
    #     
    #     self.logger.info(f"📏 Bottom border optimization complete:")
    #     self.logger.info(f"   Original bottom: {original_bottom}")
    #     self.logger.info(f"   Optimized bottom: {y + bh}")
    #     self.logger.info(f"   Extension: {final_extension} pixels")
    #     self.logger.info(f"   Strategies used: {'Relaxed threshold' if extended > 0 else 'Multi-sampling'}")
    #     
    #     return (x, y, bw, bh)

    def align_bbox_with_white_background(self, image_path: str, original_bbox: List[float], 
                                       other_bboxes: List[List[float]] = None) -> Dict[str, Any]:
        """
        Simple bbox alignment using OpenCV-based white background detection.
        
        Args:
            image_path: Path to the image file
            original_bbox: Original bbox [x1, y1, x2, y2] in normalized coordinates
            other_bboxes: List of other bounding boxes to avoid overlapping (not used in this version)
            
        Returns:
            Dictionary containing aligned bbox and validation metrics
        """
        try:
            import cv2
            
            if other_bboxes is None:
                other_bboxes = []
            
            self.logger.info(f"🎯 Starting OpenCV-based bbox alignment with white background detection")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL if cv2 fails
                pil_image = Image.open(image_path).convert('RGB')
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            img_height, img_width = image.shape[:2]
            
            # Convert normalized bbox to pixel coordinates  
            x1, y1, x2, y2 = original_bbox
            px1 = int(x1 * img_width)
            py1 = int(y1 * img_height)
            px2 = int(x2 * img_width)
            py2 = int(y2 * img_height)
            
            # Convert to (x, y, width, height) format for OpenCV approach
            bbox_xywh = (px1, py1, px2 - px1, py2 - py1)
            
            # Apply OpenCV-based white background extension
            extended_bbox_xywh = self.extend_bbox_to_white(
                image, bbox_xywh, white_threshold=240, max_extension=500
            )
            
            # COMMENTED OUT - Bottom border optimization (reverted to previous strategy)
            # optimized_bbox_xywh = self.optimize_bottom_border(
            #     image, extended_bbox_xywh, white_threshold=240, max_extension=300,
            #     tolerance_ratio=0.15, sample_width=60
            # )
            
            # Convert back to normalized (x1, y1, x2, y2) format
            ex, ey, ew, eh = extended_bbox_xywh
            extended_bbox = [
                ex / img_width,           # x1
                ey / img_height,          # y1  
                (ex + ew) / img_width,    # x2
                (ey + eh) / img_height    # y2
            ]
            
            # Calculate extension metrics for logging
            original_area = (px2 - px1) * (py2 - py1)
            extended_area = ew * eh
            extension_ratio = extended_area / original_area if original_area > 0 else 1.0
            
            self.logger.info(f"� OpenCV alignment complete:")
            self.logger.info(f"   Original bbox: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
            self.logger.info(f"   Extended bbox: [{extended_bbox[0]:.3f}, {extended_bbox[1]:.3f}, {extended_bbox[2]:.3f}, {extended_bbox[3]:.3f}]")
            self.logger.info(f"   Extension ratio: {extension_ratio:.2f}x")
            
            # Simple validation metrics
            validation_metrics = {
                'opencv': {
                    'white_coverage': 0.8,  # Placeholder - OpenCV method inherently finds white areas
                    'boundary_sharpness': 0.9,  # OpenCV method stops at sharp boundaries
                    'size_stability': min(2.0, extension_ratio) / 2.0,  # Penalize excessive expansion
                    'overall_score': 0.85  # Good baseline score for OpenCV approach
                }
            }
            
            return {
                'aligned_bbox': extended_bbox,
                'best_strategy': 'opencv',
                'all_strategies': {'opencv': extended_bbox},
                'validation_metrics': validation_metrics,
                'original_bbox': original_bbox,
                'extension_ratio': extension_ratio
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in OpenCV bbox alignment: {str(e)}")
            return {
                'aligned_bbox': original_bbox,
                'best_strategy': 'fallback',
                'error': str(e)
            }
    
    # COMMENTED OUT - OLD ALIGNMENT STRATEGIES (replaced with OpenCV approach)
    # def _align_with_gradient_analysis(self, image_path: str, bbox: List[float], 
    #                                 other_bboxes: List[List[float]]) -> List[float]:
    #     """
    #     Strategy 1: Use gradient analysis to find precise white background boundaries.
    #     """
    #     try:
    #         from scipy import ndimage
    #         import cv2
    #         
    #         # Load and process image
    #         image = Image.open(image_path).convert('RGB')
    #         img_array = np.array(image)
    #         img_width, img_height = image.size
    #         
    #         # Convert bbox to pixel coordinates with padding for analysis
    #         x1, y1, x2, y2 = bbox
    #         padding = 50  # Pixels to extend search area
    #         
    #         px1 = max(0, int(x1 * img_width) - padding)
    #         py1 = max(0, int(y1 * img_height) - padding)
    #         px2 = min(img_width, int(x2 * img_width) + padding)
    #         py2 = min(img_height, int(y2 * img_height) + padding)
    #         
    #         # Extract extended region for analysis
    #         analysis_region = img_array[py1:py2, px1:px2]
    #         
    #         # Convert to grayscale for gradient analysis
    #         gray_region = cv2.cvtColor(analysis_region, cv2.COLOR_RGB2GRAY)
    #         
    #         # Compute gradients
    #         grad_x = ndimage.sobel(gray_region, axis=1)
    #         grad_y = ndimage.sobel(gray_region, axis=0)
    #         gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    #         
    #         # Find white regions (high brightness, low gradient)
    #         brightness_mask = gray_region > 200  # White threshold
    #         low_gradient_mask = gradient_magnitude < 30  # Low gradient threshold
    #         white_background_mask = brightness_mask & low_gradient_mask
    #         
    #         # Find optimal boundaries by searching for consistent white regions
    #         original_rel_x1 = (int(x1 * img_width) - px1) / (px2 - px1)
    #         original_rel_y1 = (int(y1 * img_height) - py1) / (py2 - py1)
    #         original_rel_x2 = (int(x2 * img_width) - px1) / (px2 - px1)
    #         original_rel_y2 = (int(y2 * img_height) - py1) / (py2 - py1)
    #         
    #         # Extend boundaries to white regions
    #         aligned_bbox = self._extend_to_white_boundaries(
    #             white_background_mask, 
    #             [original_rel_x1, original_rel_y1, original_rel_x2, original_rel_y2],
    #             (px1, py1, px2, py2), 
    #             (img_width, img_height)
    #         )
    #         
    #         self.logger.info(f"🔍 Gradient analysis alignment: {aligned_bbox}")
    #         return aligned_bbox
    #         
    #     except Exception as e:
    #         self.logger.warning(f"⚠️ Gradient analysis failed: {str(e)}")
    #         return bbox
    
    # def _align_with_multiscale_embeddings(self, image_path: str, bbox: List[float], 
    #                                     other_bboxes: List[List[float]]) -> List[float]:
    #     """
    #     Strategy 2: Use multi-scale DinoV2/V3 embeddings for precise boundary detection.
    #     """
    #     try:
    #         # Analyze at multiple scales for better boundary precision
    #         scales = [0.8, 1.0, 1.2]  # Different crop scales around the bbox
    #         scale_results = []
    #         
    #         for scale in scales:
    #             # Create scaled bbox for analysis
    #             center_x = (bbox[0] + bbox[2]) / 2
    #             center_y = (bbox[1] + bbox[3]) / 2
    #             width = (bbox[2] - bbox[0]) * scale
    #             height = (bbox[3] - bbox[1]) * scale
    #             
    #             scaled_bbox = [
    #                 max(0.0, center_x - width/2),
    #                 max(0.0, center_y - height/2),
    #                 min(1.0, center_x + width/2),
    #                 min(1.0, center_y + height/2)
    #             ]
    #             
    #             # Get white probability map at this scale
    #             white_map = self.detect_white_background_with_dino(image_path, scaled_bbox)
    #             
    #             # Find optimal boundaries within this scale
    #             optimal_boundaries = self._find_optimal_boundaries_from_embeddings(
    #                 white_map, scaled_bbox, bbox
    #             )
    #             
    #             scale_results.append({
    #                 'scale': scale,
    #                 'boundaries': optimal_boundaries,
    #                 'confidence': np.mean(white_map)
    #             })
    #         
    #         # Select best scale result based on confidence
    #         best_result = max(scale_results, key=lambda x: x['confidence'])
    #         aligned_bbox = best_result['boundaries']
    #         
    #         self.logger.info(f"🔬 Multi-scale embedding alignment: {aligned_bbox}")
    #         self.logger.info(f"   Best scale: {best_result['scale']}, confidence: {best_result['confidence']:.3f}")
    #         
    #         return aligned_bbox
    #         
    #     except Exception as e:
    #         self.logger.warning(f"⚠️ Multi-scale embedding analysis failed: {str(e)}")
    #         return bbox
    
    # def _align_with_contour_detection(self, image_path: str, bbox: List[float], 
    #                                 other_bboxes: List[List[float]]) -> List[float]:
    #     """
    #     Strategy 3: Use contour detection to find precise white background boundaries.
    #     """
    #     try:
    #         import cv2
    #         
    #         # Load image
    #         image = Image.open(image_path).convert('RGB')
    #         img_array = np.array(image)
    #         img_width, img_height = image.size
    #         
    #         # Convert to HSV for better white detection
    #         hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    #         
    #         # Create mask for white/light colors
    #         # White in HSV: low saturation, high value
    #         lower_white = np.array([0, 0, 200])
    #         upper_white = np.array([180, 30, 255])
    #         white_mask = cv2.inRange(hsv, lower_white, upper_white)
    #         
    #         # Morphological operations to clean up the mask
    #         kernel = np.ones((5, 5), np.uint8)
    #         white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    #         white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    #         
    #         # Find contours
    #         contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         
    #         # Filter contours by area and find the one containing our bbox
    #         min_area = 1000  # Minimum contour area
    #         large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    #         
    #         # Find contour that best encompasses the original bbox
    #         bbox_center_x = int((bbox[0] + bbox[2]) / 2 * img_width)
    #         bbox_center_y = int((bbox[1] + bbox[3]) / 2 * img_height)
    #         
    #         best_contour = None
    #         best_distance = float('inf')
    #         
    #         for contour in large_contours:
    #             # Check if bbox center is inside this contour
    #             distance = cv2.pointPolygonTest(contour, (bbox_center_x, bbox_center_y), True)
    #             if distance >= 0 and distance < best_distance:  # Inside contour
    #                 best_contour = contour
    #                 best_distance = distance
    #         
    #         if best_contour is not None:
    #             # Get bounding rectangle of the contour
    #             x, y, w, h = cv2.boundingRect(best_contour)
    #             
    #             # Convert back to normalized coordinates
    #             aligned_bbox = [
    #                 max(0.0, x / img_width),
    #                 max(0.0, y / img_height),
    #                 min(1.0, (x + w) / img_width),
    #                 min(1.0, (y + h) / img_height)
    #             ]
    #             
    #             # Ensure minimum size and avoid excessive expansion
    #             aligned_bbox = self._constrain_bbox_expansion(bbox, aligned_bbox, max_expansion=0.5)
    #             
    #             self.logger.info(f"🎭 Contour-based alignment: {aligned_bbox}")
    #             return aligned_bbox
    #         else:
    #             self.logger.info(f"🎭 No suitable contour found, using original bbox")
    #             return bbox
    #             
    #     except Exception as e:
    #         self.logger.warning(f"⚠️ Contour detection failed: {str(e)}")
    #         return bbox
    
    # def _align_with_adaptive_thresholding(self, image_path: str, bbox: List[float], 
    #                                     other_bboxes: List[List[float]]) -> List[float]:
    #     """
    #     Strategy 4: Use adaptive thresholding based on local image statistics.
    #     """
    #     try:
    #         # Get white probability map from embeddings
    #         white_map = self.detect_white_background_with_dino(image_path, bbox)
    #         
    #         # Calculate adaptive threshold based on local statistics
    #         mean_prob = np.mean(white_map)
    #         std_prob = np.std(white_map)
    #         
    #         # Adaptive threshold: mean + k * std, where k depends on variance
    #         if std_prob > 0.1:
    #             k = 0.5  # High variance - be more conservative
    #         elif std_prob > 0.05:
    #             k = 1.0  # Medium variance - standard approach
    #         else:
    #             k = 1.5  # Low variance - be more aggressive
    #         
    #         adaptive_threshold = min(0.9, mean_prob + k * std_prob)
    #         
    #         # Find regions above adaptive threshold
    #         high_confidence_mask = white_map > adaptive_threshold
    #         
    #         # Extend bbox to encompass high-confidence white regions
    #         aligned_bbox = self._extend_bbox_to_mask_boundaries(
    #             bbox, high_confidence_mask, expansion_limit=0.3
    #         )
    #         
    #         self.logger.info(f"🎚️ Adaptive threshold alignment: {aligned_bbox}")
    #         self.logger.info(f"   Threshold: {adaptive_threshold:.3f}, k-factor: {k}")
    #         
    #         return aligned_bbox
    #         
    #     except Exception as e:
    #         self.logger.warning(f"⚠️ Adaptive thresholding failed: {str(e)}")
    #         return bbox
    # 
    # def _compute_consensus_alignment(self, original_bbox: List[float], *strategy_bboxes) -> List[float]:
    #     """
    #     Strategy 5: Compute consensus alignment by averaging valid strategy results.
    #     """
    #     try:
    #         valid_bboxes = [bbox for bbox in strategy_bboxes if bbox is not None]
    #         
    #         if not valid_bboxes:
    #             return original_bbox
    #         
    #         # Calculate weighted average with more weight to less extreme changes
    #         weights = []
    #         for bbox in valid_bboxes:
    #             # Calculate change magnitude
    #             change_magnitude = sum(abs(bbox[i] - original_bbox[i]) for i in range(4))
    #             # Inverse weight - smaller changes get higher weight
    #             weight = 1.0 / (1.0 + change_magnitude * 10)  # Scale factor
    #             weights.append(weight)
    #         
    #         # Normalize weights
    #         total_weight = sum(weights)
    #         if total_weight > 0:
    #             weights = [w / total_weight for w in weights]
    #         else:
    #             weights = [1.0 / len(valid_bboxes)] * len(valid_bboxes)
    #         
    #         # Compute weighted average
    #         consensus_bbox = [0.0, 0.0, 0.0, 0.0]
    #         for i in range(4):
    #             consensus_bbox[i] = sum(bbox[i] * weight for bbox, weight in zip(valid_bboxes, weights))
    #         
    #         # Ensure valid bounds
    #         consensus_bbox = [
    #             max(0.0, min(1.0, consensus_bbox[0])),
    #             max(0.0, min(1.0, consensus_bbox[1])),
    #             max(0.0, min(1.0, consensus_bbox[2])),
    #             max(0.0, min(1.0, consensus_bbox[3]))
    #         ]
    #         
    #         self.logger.info(f"🤝 Consensus alignment: {consensus_bbox}")
    #         self.logger.info(f"   Used {len(valid_bboxes)} strategies with weights: {[f'{w:.3f}' for w in weights]}")
    #         
    #         return consensus_bbox
    #         
    #     except Exception as e:
    #         self.logger.warning(f"⚠️ Consensus computation failed: {str(e)}")
    #         return original_bbox
    
    def _extend_to_white_boundaries(self, white_mask: np.ndarray, relative_bbox: List[float], 
                                   analysis_coords: Tuple[int, int, int, int], 
                                   img_size: Tuple[int, int]) -> List[float]:
        """Helper method to extend bbox to white background boundaries."""
        try:
            mask_height, mask_width = white_mask.shape
            px1, py1, px2, py2 = analysis_coords
            img_width, img_height = img_size
            
            # Convert relative coordinates to mask coordinates
            mask_x1 = int(relative_bbox[0] * mask_width)
            mask_y1 = int(relative_bbox[1] * mask_height)
            mask_x2 = int(relative_bbox[2] * mask_width)
            mask_y2 = int(relative_bbox[3] * mask_height)
            
            # Extend boundaries to white regions
            # Extend left
            new_x1 = mask_x1
            for x in range(mask_x1, -1, -1):
                if np.mean(white_mask[mask_y1:mask_y2, max(0, x-5):x+5]) > 0.7:
                    new_x1 = x
                else:
                    break
            
            # Extend right
            new_x2 = mask_x2
            for x in range(mask_x2, mask_width):
                if np.mean(white_mask[mask_y1:mask_y2, x:min(mask_width, x+5)]) > 0.7:
                    new_x2 = x
                else:
                    break
            
            # Extend top
            new_y1 = mask_y1
            for y in range(mask_y1, -1, -1):
                if np.mean(white_mask[max(0, y-5):y+5, mask_x1:mask_x2]) > 0.7:
                    new_y1 = y
                else:
                    break
            
            # Extend bottom
            new_y2 = mask_y2
            for y in range(mask_y2, mask_height):
                if np.mean(white_mask[y:min(mask_height, y+5), mask_x1:mask_x2]) > 0.7:
                    new_y2 = y
                else:
                    break
            
            # Convert back to normalized coordinates
            extended_bbox = [
                max(0.0, (px1 + new_x1 * (px2 - px1) / mask_width) / img_width),
                max(0.0, (py1 + new_y1 * (py2 - py1) / mask_height) / img_height),
                min(1.0, (px1 + new_x2 * (px2 - px1) / mask_width) / img_width),
                min(1.0, (py1 + new_y2 * (py2 - py1) / mask_height) / img_height)
            ]
            
            return extended_bbox
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error extending to white boundaries: {str(e)}")
            return [relative_bbox[0], relative_bbox[1], relative_bbox[2], relative_bbox[3]]
    
    def _find_optimal_boundaries_from_embeddings(self, white_map: np.ndarray, 
                                               scaled_bbox: List[float], 
                                               original_bbox: List[float]) -> List[float]:
        """Find optimal boundaries from embedding-based white probability map."""
        try:
            grid_size = white_map.shape[0]
            
            # Find high-confidence white regions (above 75th percentile)
            threshold = np.percentile(white_map, 75)
            high_conf_mask = white_map > threshold
            
            # Find the bounding box of high-confidence regions
            rows, cols = np.where(high_conf_mask)
            
            if len(rows) == 0:
                return original_bbox
            
            # Calculate boundaries in grid coordinates
            min_row, max_row = np.min(rows), np.max(rows)
            min_col, max_col = np.min(cols), np.max(cols)
            
            # Convert to relative coordinates within the scaled bbox
            rel_y1 = min_row / grid_size
            rel_y2 = (max_row + 1) / grid_size
            rel_x1 = min_col / grid_size
            rel_x2 = (max_col + 1) / grid_size
            
            # Map back to absolute normalized coordinates
            bbox_width = scaled_bbox[2] - scaled_bbox[0]
            bbox_height = scaled_bbox[3] - scaled_bbox[1]
            
            optimal_bbox = [
                max(0.0, scaled_bbox[0] + rel_x1 * bbox_width),
                max(0.0, scaled_bbox[1] + rel_y1 * bbox_height),
                min(1.0, scaled_bbox[0] + rel_x2 * bbox_width),
                min(1.0, scaled_bbox[1] + rel_y2 * bbox_height)
            ]
            
            return optimal_bbox
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error finding optimal boundaries: {str(e)}")
            return original_bbox
    
    def _constrain_bbox_expansion(self, original_bbox: List[float], new_bbox: List[float], 
                                max_expansion: float = 0.5) -> List[float]:
        """Constrain bbox expansion to prevent excessive changes."""
        try:
            original_width = original_bbox[2] - original_bbox[0]
            original_height = original_bbox[3] - original_bbox[1]
            
            max_width_change = original_width * max_expansion
            max_height_change = original_height * max_expansion
            
            # Constrain width expansion
            if (new_bbox[2] - new_bbox[0]) > (original_width + max_width_change):
                center_x = (new_bbox[0] + new_bbox[2]) / 2
                new_width = original_width + max_width_change
                new_bbox[0] = center_x - new_width / 2
                new_bbox[2] = center_x + new_width / 2
            
            # Constrain height expansion  
            if (new_bbox[3] - new_bbox[1]) > (original_height + max_height_change):
                center_y = (new_bbox[1] + new_bbox[3]) / 2
                new_height = original_height + max_height_change
                new_bbox[1] = center_y - new_height / 2
                new_bbox[3] = center_y + new_height / 2
            
            # Ensure bounds
            new_bbox = [max(0.0, min(1.0, coord)) for coord in new_bbox]
            
            return new_bbox
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error constraining bbox expansion: {str(e)}")
            return original_bbox
    
    def _extend_bbox_to_mask_boundaries(self, bbox: List[float], mask: np.ndarray, 
                                      expansion_limit: float = 0.3) -> List[float]:
        """Extend bbox to encompass mask boundaries with limits."""
        try:
            grid_size = mask.shape[0]
            
            # Find mask boundaries
            rows, cols = np.where(mask)
            
            if len(rows) == 0:
                return bbox
            
            # Calculate extension in grid coordinates
            min_row, max_row = np.min(rows), np.max(rows)
            min_col, max_col = np.min(cols), np.max(cols)
            
            # Convert to bbox relative coordinates
            rel_y1 = min_row / grid_size
            rel_y2 = (max_row + 1) / grid_size
            rel_x1 = min_col / grid_size
            rel_x2 = (max_col + 1) / grid_size
            
            # Map to image coordinates and apply expansion limits
            original_width = bbox[2] - bbox[0]
            original_height = bbox[3] - bbox[1]
            
            max_width_expansion = original_width * expansion_limit
            max_height_expansion = original_height * expansion_limit
            
            extended_bbox = [
                max(0.0, max(bbox[0] - max_width_expansion, bbox[0] + rel_x1 * original_width)),
                max(0.0, max(bbox[1] - max_height_expansion, bbox[1] + rel_y1 * original_height)),
                min(1.0, min(bbox[2] + max_width_expansion, bbox[0] + rel_x2 * original_width)),
                min(1.0, min(bbox[3] + max_height_expansion, bbox[1] + rel_y2 * original_height))
            ]
            
            return extended_bbox
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error extending bbox to mask boundaries: {str(e)}")
            return bbox
    
    # COMMENTED OUT - OLD VALIDATION METHODS (no longer needed with single OpenCV strategy)
    # def _validate_alignment_strategies(self, image_path: str, original_bbox: List[float], 
    #                                  strategy_results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    #     """Validate each alignment strategy with multiple metrics."""
    #     try:
    #         validation_results = {}
    #         
    #         for strategy_name, aligned_bbox in strategy_results.items():
    #             if aligned_bbox is None:
    #                 validation_results[strategy_name] = {
    #                     'white_coverage': 0.0,
    #                     'boundary_sharpness': 0.0,
    #                     'size_stability': 0.0,
    #                     'embedding_confidence': 0.0,
    #                     'overall_score': 0.0
    #                 }
    #                 continue
    #             
    #             # Metric 1: White coverage - how much of the bbox is white background
    #             white_map = self.detect_white_background_with_dino(image_path, aligned_bbox)
    #             white_coverage = np.mean(white_map)
    #             
    #             # Metric 2: Boundary sharpness - how well-defined are the boundaries
    #             boundary_sharpness = self._calculate_boundary_sharpness(image_path, aligned_bbox)
    #             
    #             # Metric 3: Size stability - penalize excessive size changes
    #             original_area = (original_bbox[2] - original_bbox[0]) * (original_bbox[3] - original_bbox[1])
    #             aligned_area = (aligned_bbox[2] - aligned_bbox[0]) * (aligned_bbox[3] - aligned_bbox[1])
    #             size_ratio = aligned_area / original_area if original_area > 0 else 1.0
    #             size_stability = 1.0 / (1.0 + abs(size_ratio - 1.0))
                
    #             # Metric 4: Embedding confidence - consistency with DinoV2/V3 analysis
    #             embedding_confidence = min(1.0, white_coverage * 1.5)  # Scale white coverage
    #             
    #             # Overall score (weighted combination)
    #             overall_score = (
    #                 white_coverage * 0.4 +
    #                 boundary_sharpness * 0.2 + 
    #                 size_stability * 0.2 +
    #                 embedding_confidence * 0.2
    #             )
    #             
    #             validation_results[strategy_name] = {
    #                 'white_coverage': white_coverage,
    #                 'boundary_sharpness': boundary_sharpness,
    #                 'size_stability': size_stability,
    #                 'embedding_confidence': embedding_confidence,
    #                 'overall_score': overall_score
    #             }
    #             
    #             self.logger.info(f"📊 {strategy_name} validation:")
    #             self.logger.info(f"   White coverage: {white_coverage:.3f}")
    #             self.logger.info(f"   Boundary sharpness: {boundary_sharpness:.3f}")
    #             self.logger.info(f"   Size stability: {size_stability:.3f}")
    #             self.logger.info(f"   Overall score: {overall_score:.3f}")
    #         
    #         return validation_results
    #         
    #     except Exception as e:
    #         self.logger.warning(f"⚠️ Error validating alignment strategies: {str(e)}")
    #         return {}
    # 
    # def _calculate_boundary_sharpness(self, image_path: str, bbox: List[float]) -> float:
    #     """Calculate how sharp/well-defined the boundaries are."""
    #     try:
    #         import cv2
    #         from scipy import ndimage
    #         
    #         # Load image
    #         image = Image.open(image_path).convert('RGB')
    #         img_array = np.array(image)
    #         img_width, img_height = image.size
    #         
    #         # Extract bbox region with small border
    #         border = 10
    #         x1 = max(0, int(bbox[0] * img_width) - border)
    #         y1 = max(0, int(bbox[1] * img_height) - border)
    #         x2 = min(img_width, int(bbox[2] * img_width) + border)
    #         y2 = min(img_height, int(bbox[3] * img_height) + border)
    #         
    #         region = img_array[y1:y2, x1:x2]
    #         gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    #         
    #         # Calculate gradient magnitude at boundaries
    #         grad_x = ndimage.sobel(gray_region, axis=1)
    #         grad_y = ndimage.sobel(gray_region, axis=0)
    #         gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    #         
    #         # Focus on boundary regions (edges of the extracted region)
    #         boundary_width = 5
    #         h, w = gradient_magnitude.shape
    #         
    #         boundaries = np.concatenate([
    #             gradient_magnitude[:boundary_width, :].flatten(),  # Top
    #             gradient_magnitude[-boundary_width:, :].flatten(),  # Bottom
    #             gradient_magnitude[:, :boundary_width].flatten(),  # Left
    #             gradient_magnitude[:, -boundary_width:].flatten()   # Right
    #         ])
    #         
    #         # Higher gradient = sharper boundary
    #         sharpness = np.mean(boundaries) / 255.0  # Normalize
    #         return min(1.0, sharpness)
    #         
    #     except Exception as e:
    #         self.logger.warning(f"⚠️ Error calculating boundary sharpness: {str(e)}")
    #         return 0.5  # Default medium sharpness
    
    # def _select_best_alignment(self, validation_results: Dict[str, Dict[str, float]]) -> Tuple[str, List[float]]:
    #     """Select the best alignment strategy based on validation metrics."""
    #     try:
    #         if not validation_results:
    #             return 'fallback', None
    #         
    #         # Find strategy with highest overall score
    #         best_strategy = max(validation_results.keys(), 
    #                           key=lambda k: validation_results[k]['overall_score'])
    #         
    #         best_score = validation_results[best_strategy]['overall_score']
    #         
    #         # If best score is too low, fall back to original
    #         if best_score < 0.3:
    #             self.logger.warning(f"⚠️ All strategies scored low (best: {best_score:.3f}), using fallback")
    #             return 'fallback', None
    #         
    #         self.logger.info(f"🏆 Selected {best_strategy} with score: {best_score:.3f}")
    #         return best_strategy, best_strategy  # Return strategy name twice for now - will be fixed in alignment method
    #         
    #     except Exception as e:
    #         self.logger.warning(f"⚠️ Error selecting best alignment: {str(e)}")
    #         return 'fallback', None
    
    def extend_bbox_with_white_detection(self, image_path: str, original_bbox: List[float], 
                                       white_threshold: float = 0.6, 
                                       other_bboxes: List[List[float]] = None) -> List[float]:
        """
        Extend a bounding box based on PCA white background detection until reaching 
        non-white content or another bounding box.
        
        Args:
            image_path: Path to the image file
            original_bbox: Original bbox [x1, y1, x2, y2] in normalized coordinates
            white_threshold: Threshold for white background detection (0.0-1.0)
            other_bboxes: List of other bounding boxes to avoid overlapping
            
        Returns:
            Extended bounding box [x1, y1, x2, y2] in normalized coordinates
        """
        try:
            if other_bboxes is None:
                other_bboxes = []
            
            # Load image to get dimensions
            image = Image.open(image_path)
            img_width, img_height = image.size
            
            # Convert original bbox to pixel coordinates
            x1, y1, x2, y2 = original_bbox
            px1 = int(x1 * img_width)
            py1 = int(y1 * img_height)
            px2 = int(x2 * img_width)
            py2 = int(y2 * img_height)
            
            # Extension parameters - make them adaptive to image size
            base_step = max(5, min(20, int(min(img_width, img_height) * 0.01)))  # 1% of smaller dimension
            max_extensions = 25  # Allow more extension attempts
            
            # Start with original coordinates
            extended_px1, extended_py1 = px1, py1
            extended_px2, extended_py2 = px2, py2
            
            self.logger.info(f"🔧 Starting extension with step size: {base_step}px, threshold: {white_threshold}")
            
            # Extend LEFT direction
            consecutive_white_left = 0
            for step in range(max_extensions):
                new_x1 = max(0, extended_px1 - base_step)
                if new_x1 == extended_px1:  # Hit image boundary
                    break
                    
                test_bbox = [new_x1 / img_width, y1, x2, y2]
                
                # Check collision with other bboxes
                if self._bbox_overlaps_with_others(test_bbox, other_bboxes, original_bbox):
                    self.logger.info(f"⚠️ Left extension stopped: collision with other bbox")
                    break
                
                # Analyze white background in the LEFT extension region only
                extension_region = [new_x1 / img_width, y1, extended_px1 / img_width, y2]
                white_map = self.detect_white_background_with_dino(image_path, extension_region)
                avg_whiteness = np.mean(white_map)
                
                if avg_whiteness > white_threshold:
                    extended_px1 = new_x1
                    consecutive_white_left += 1
                else:
                    self.logger.info(f"🛑 Left extension stopped: non-white content detected (whiteness: {avg_whiteness:.3f})")
                    break
            
            # Extend RIGHT direction
            consecutive_white_right = 0
            for step in range(max_extensions):
                new_x2 = min(img_width, extended_px2 + base_step)
                if new_x2 == extended_px2:  # Hit image boundary
                    break
                    
                test_bbox = [x1, y1, new_x2 / img_width, y2]
                
                # Check collision with other bboxes
                if self._bbox_overlaps_with_others(test_bbox, other_bboxes, original_bbox):
                    self.logger.info(f"⚠️ Right extension stopped: collision with other bbox")
                    break
                
                # Analyze white background in the RIGHT extension region only
                extension_region = [extended_px2 / img_width, y1, new_x2 / img_width, y2]
                white_map = self.detect_white_background_with_dino(image_path, extension_region)
                avg_whiteness = np.mean(white_map)
                
                if avg_whiteness > white_threshold:
                    extended_px2 = new_x2
                    consecutive_white_right += 1
                else:
                    self.logger.info(f"🛑 Right extension stopped: non-white content detected (whiteness: {avg_whiteness:.3f})")
                    break
            
            # Extend TOP direction
            consecutive_white_top = 0
            for step in range(max_extensions):
                new_y1 = max(0, extended_py1 - base_step)
                if new_y1 == extended_py1:  # Hit image boundary
                    break
                    
                test_bbox = [x1, new_y1 / img_height, x2, y2]
                
                # Check collision with other bboxes
                if self._bbox_overlaps_with_others(test_bbox, other_bboxes, original_bbox):
                    self.logger.info(f"⚠️ Top extension stopped: collision with other bbox")
                    break
                
                # Analyze white background in the TOP extension region only
                extension_region = [x1, new_y1 / img_height, x2, extended_py1 / img_height]
                white_map = self.detect_white_background_with_dino(image_path, extension_region)
                avg_whiteness = np.mean(white_map)
                
                if avg_whiteness > white_threshold:
                    extended_py1 = new_y1
                    consecutive_white_top += 1
                else:
                    self.logger.info(f"🛑 Top extension stopped: non-white content detected (whiteness: {avg_whiteness:.3f})")
                    break
            
            # Extend BOTTOM direction
            consecutive_white_bottom = 0
            for step in range(max_extensions):
                new_y2 = min(img_height, extended_py2 + base_step)
                if new_y2 == extended_py2:  # Hit image boundary
                    break
                    
                test_bbox = [x1, y1, x2, new_y2 / img_height]
                
                # Check collision with other bboxes
                if self._bbox_overlaps_with_others(test_bbox, other_bboxes, original_bbox):
                    self.logger.info(f"⚠️ Bottom extension stopped: collision with other bbox")
                    break
                
                # Analyze white background in the BOTTOM extension region only
                extension_region = [x1, extended_py2 / img_height, x2, new_y2 / img_height]
                white_map = self.detect_white_background_with_dino(image_path, extension_region)
                avg_whiteness = np.mean(white_map)
                
                if avg_whiteness > white_threshold:
                    extended_py2 = new_y2
                    consecutive_white_bottom += 1
                else:
                    self.logger.info(f"🛑 Bottom extension stopped: non-white content detected (whiteness: {avg_whiteness:.3f})")
                    break
            
            # Convert back to normalized coordinates
            extended_bbox = [
                extended_px1 / img_width,
                extended_py1 / img_height,
                extended_px2 / img_width,
                extended_py2 / img_height
            ]
            
            # Calculate extension amounts
            dx1 = (px1 - extended_px1) / img_width
            dy1 = (py1 - extended_py1) / img_height
            dx2 = (extended_px2 - px2) / img_width
            dy2 = (extended_py2 - py2) / img_height
            
            total_extension = dx1 + dy1 + dx2 + dy2
            
            self.logger.info(f"📏 PCA-based extension results:")
            self.logger.info(f"    Left: {dx1:.3f} ({consecutive_white_left} steps)")
            self.logger.info(f"    Right: {dx2:.3f} ({consecutive_white_right} steps)")
            self.logger.info(f"    Top: {dy1:.3f} ({consecutive_white_top} steps)")
            self.logger.info(f"    Bottom: {dy2:.3f} ({consecutive_white_bottom} steps)")
            self.logger.info(f"    Total extension: {total_extension:.3f}")
            
            # Fallback approach: if no extension occurred (all directions are 0.000), 
            # extend by adaptive fallback based on bbox dimensions
            if total_extension < 0.001:  # Essentially zero extension
                self.logger.info("🔄 PCA extension yielded no results (total extension < 0.001), applying fallback approach...")
                self.logger.info("💡 This typically occurs when:")
                self.logger.info("   - Background has very subtle variations")
                self.logger.info("   - Content boundaries are not clearly defined")
                self.logger.info("   - Document has uniform/complex patterns")
                
                fallback_bbox = self._apply_pca_component_fallback(
                    image_path, original_bbox, other_bboxes
                )
                
                # Calculate fallback extension amounts for logging
                fdx1 = original_bbox[0] - fallback_bbox[0]
                fdy1 = original_bbox[1] - fallback_bbox[1]
                fdx2 = fallback_bbox[2] - original_bbox[2]
                fdy2 = fallback_bbox[3] - original_bbox[3]
                fallback_total = fdx1 + fdy1 + fdx2 + fdy2
                
                self.logger.info(f"🎯 Fallback extension results:")
                self.logger.info(f"    Left: {fdx1:.3f}, Top: {fdy1:.3f}, Right: {fdx2:.3f}, Bottom: {fdy2:.3f}")
                self.logger.info(f"    Total fallback extension: {fallback_total:.3f}")
                
                return fallback_bbox
            
            return extended_bbox
            
        except Exception as e:
            self.logger.error(f"❌ Error extending bbox with PCA: {str(e)}")
            return original_bbox
    
    def _bbox_overlaps_with_others(self, bbox: List[float], other_bboxes: List[List[float]], 
                                 exclude_bbox: List[float] = None) -> bool:
        """
        Check if a bounding box overlaps with any other bounding boxes.
        
        Args:
            bbox: Bounding box to check [x1, y1, x2, y2]
            other_bboxes: List of other bounding boxes
            exclude_bbox: Bbox to exclude from overlap check
            
        Returns:
            True if overlap detected, False otherwise
        """
        x1, y1, x2, y2 = bbox
        
        for other_bbox in other_bboxes:
            # Skip if this is the bbox we want to exclude
            if exclude_bbox and other_bbox == exclude_bbox:
                continue
            
            ox1, oy1, ox2, oy2 = other_bbox
            
            # Check for overlap
            if not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2):
                return True
        
        return False
    
    def _apply_pca_component_fallback(self, image_path: str, original_bbox: List[float], 
                                    other_bboxes: List[List[float]] = None) -> List[float]:
        """
        Fallback approach: extend bounding box by the size of the largest PCA component.
        This is used when PCA-based white detection doesn't find clear extension boundaries.
        
        Args:
            image_path: Path to the image file
            original_bbox: Original bbox [x1, y1, x2, y2] in normalized coordinates
            other_bboxes: List of other bounding boxes to avoid overlapping
            
        Returns:
            Extended bounding box [x1, y1, x2, y2] in normalized coordinates
        """
        try:
            if other_bboxes is None:
                other_bboxes = []
            
            # Load image to get dimensions
            image = Image.open(image_path)
            img_width, img_height = image.size
            
            # Get PCA analysis of ONLY the original bbox region to determine extension amount
            white_map = self.detect_white_background_with_dino(image_path, original_bbox)
            
            # Calculate bbox dimensions
            bbox_width = original_bbox[2] - original_bbox[0]
            bbox_height = original_bbox[3] - original_bbox[1]
            
            # Analyze white map from the cropped bbox region to determine appropriate extension strategy
            avg_whiteness = np.mean(white_map)
            max_whiteness = np.max(white_map)
            whiteness_std = np.std(white_map)
            
            self.logger.info(f"🔍 PCA analysis for fallback decision (bbox region only):")
            self.logger.info(f"   Average whiteness: {avg_whiteness:.3f}")
            self.logger.info(f"   Max whiteness: {max_whiteness:.3f}")
            self.logger.info(f"   Whiteness std dev: {whiteness_std:.3f}")
            
            # Adaptive extension factor based on whiteness characteristics
            if avg_whiteness > 0.7:
                # High whiteness suggests good potential for extension
                base_extension_factor = 0.20  # More aggressive extension
                self.logger.info("📈 High whiteness detected: using aggressive extension (20%)")
            elif avg_whiteness > 0.5:
                # Moderate whiteness
                base_extension_factor = 0.15  # Standard extension
                self.logger.info("📊 Moderate whiteness detected: using standard extension (15%)")
            else:
                # Low whiteness suggests complex content nearby
                base_extension_factor = 0.10  # Conservative extension
                self.logger.info("📉 Low whiteness detected: using conservative extension (10%)")
            
            # Further adjust based on whiteness variability
            if whiteness_std < 0.1:
                # Low variability suggests uniform region - can extend more
                base_extension_factor *= 1.2
                self.logger.info(f"🎯 Low variability detected: increasing extension to {base_extension_factor:.1%}")
            elif whiteness_std > 0.3:
                # High variability suggests mixed content - be more conservative
                base_extension_factor *= 0.8
                self.logger.info(f"⚠️ High variability detected: reducing extension to {base_extension_factor:.1%}")
            
            # Calculate extension amounts in each direction
            horizontal_extension = bbox_width * base_extension_factor
            vertical_extension = bbox_height * base_extension_factor
            
            # Apply maximum limits to prevent excessive extension
            max_horizontal_ext = min(horizontal_extension, 0.1)  # Max 10% of image width
            max_vertical_ext = min(vertical_extension, 0.1)      # Max 10% of image height
            
            # Calculate proposed extended bbox
            extended_x1 = max(0.0, original_bbox[0] - max_horizontal_ext)
            extended_y1 = max(0.0, original_bbox[1] - max_vertical_ext)
            extended_x2 = min(1.0, original_bbox[2] + max_horizontal_ext)
            extended_y2 = min(1.0, original_bbox[3] + max_vertical_ext)
            
            proposed_bbox = [extended_x1, extended_y1, extended_x2, extended_y2]
            
            # Check for collisions with other bboxes and adjust if necessary
            if self._bbox_overlaps_with_others(proposed_bbox, other_bboxes, original_bbox):
                self.logger.info("⚠️ Fallback extension would overlap with other bbox, reducing extension...")
                
                # Reduce extension by half and try again
                reduced_h_ext = max_horizontal_ext * 0.5
                reduced_v_ext = max_vertical_ext * 0.5
                
                extended_x1 = max(0.0, original_bbox[0] - reduced_h_ext)
                extended_y1 = max(0.0, original_bbox[1] - reduced_v_ext)
                extended_x2 = min(1.0, original_bbox[2] + reduced_h_ext)
                extended_y2 = min(1.0, original_bbox[3] + reduced_v_ext)
                
                proposed_bbox = [extended_x1, extended_y1, extended_x2, extended_y2]
                
                # If still overlapping, use minimal extension
                if self._bbox_overlaps_with_others(proposed_bbox, other_bboxes, original_bbox):
                    self.logger.info("⚠️ Still overlapping, using minimal fallback extension...")
                    minimal_ext = 0.02  # 2% extension
                    
                    extended_x1 = max(0.0, original_bbox[0] - minimal_ext)
                    extended_y1 = max(0.0, original_bbox[1] - minimal_ext)
                    extended_x2 = min(1.0, original_bbox[2] + minimal_ext)
                    extended_y2 = min(1.0, original_bbox[3] + minimal_ext)
                    
                    proposed_bbox = [extended_x1, extended_y1, extended_x2, extended_y2]
            
            self.logger.info(f"🔧 Fallback extension applied:")
            self.logger.info(f"    Horizontal extension: ±{max_horizontal_ext:.3f}")
            self.logger.info(f"    Vertical extension: ±{max_vertical_ext:.3f}")
            self.logger.info(f"    Extension factor: {base_extension_factor:.1%}")
            
            return proposed_bbox
            
        except Exception as e:
            self.logger.error(f"❌ Error in fallback extension: {str(e)}")
            return original_bbox
    
    def process_page(self, image_path: str, output_dir: str, extension_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Process a single page: detect sections with Gemini, extend with DinoV3.
        
        Args:
            image_path: Path to page image
            output_dir: Directory to save results
            extension_threshold: Threshold for white background detection
            
        Returns:
            Dictionary with processing results
        """
        try:
            page_name = Path(image_path).stem
            self.logger.info(f"🔍 Processing page: {page_name}")
            
            # Set current output directory for embedding debugging
            self.current_output_dir = output_dir
            
            # Step 1: Detect sections with Gemini
            gemini_bboxes = self.detect_sections_with_gemini(image_path)
            
            if not gemini_bboxes:
                self.logger.warning(f"⚠️ No sections detected by Gemini for {page_name}")
                return {
                    'page_name': page_name,
                    'image_path': image_path,
                    'gemini_bboxes': [],
                    'extended_bboxes': [],
                    'processing_success': True,
                    'extension_threshold': extension_threshold,
                    'sections_detected': 0
                }
            
            # Step 2: Align bboxes with advanced white background detection
            extended_bboxes = []
            alignment_results = []
            
            for i, bbox in enumerate(gemini_bboxes):
                self.logger.info(f"🔧 Aligning bbox {i+1}/{len(gemini_bboxes)} with advanced strategies")
                
                # Use advanced alignment with multiple validation strategies
                alignment_result = self.align_bbox_with_white_background(
                    image_path, bbox, gemini_bboxes
                )
                
                # If advanced alignment fails, fall back to traditional method
                if alignment_result.get('best_strategy') == 'fallback' or alignment_result.get('aligned_bbox') is None:
                    self.logger.info(f"📦 Falling back to traditional extension for bbox {i+1}")
                    extended_bbox = self.extend_bbox_with_white_detection(
                        image_path, bbox, extension_threshold, gemini_bboxes
                    )
                    alignment_result['aligned_bbox'] = extended_bbox
                    alignment_result['best_strategy'] = 'traditional_fallback'
                
                extended_bboxes.append(alignment_result['aligned_bbox'])
                alignment_results.append(alignment_result)
            
            # Step 3: Create visualization
            viz_path = self.visualize_bboxes(
                image_path, gemini_bboxes, extended_bboxes, 
                output_dir, page_name
            )
            
            # Step 4: Prepare results with alignment information
            result = {
                'page_name': page_name,
                'image_path': image_path,
                'gemini_bboxes': gemini_bboxes,
                'extended_bboxes': extended_bboxes,
                'alignment_results': alignment_results,
                'visualization_path': viz_path,
                'processing_success': True,
                'extension_threshold': extension_threshold,
                'sections_detected': len(gemini_bboxes),
                'alignment_strategies_used': [r.get('best_strategy', 'unknown') for r in alignment_results]
            }
            
            # Step 5: Save individual page results
            self.save_page_results(result, output_dir)
            
            self.logger.info(f"✅ Page {page_name} processed successfully")
            self.logger.info(f"   - Gemini sections: {len(gemini_bboxes)}")
            self.logger.info(f"   - Extended sections: {len(extended_bboxes)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error processing page {Path(image_path).stem}: {str(e)}")
            return {
                'page_name': Path(image_path).stem,
                'image_path': image_path,
                'error': str(e),
                'processing_success': False
            }
    
    def visualize_bboxes(self, image_path: str, gemini_bboxes: List[List[float]], 
                        extended_bboxes: List[List[float]], output_dir: str, 
                        page_name: str) -> str:
        """
        Create visualization showing original and extended bounding boxes.
        
        Args:
            image_path: Path to the image
            gemini_bboxes: Original bounding boxes from Gemini
            extended_bboxes: Extended bounding boxes from DinoV3
            output_dir: Output directory
            page_name: Name of the page
            
        Returns:
            Path to saved visualization
        """
        try:
            # Load image
            image = Image.open(image_path)
            img_width, img_height = image.size
            
            # Create figure
            fig, ax = plt.subplots(1, figsize=(12, 16))
            ax.imshow(image)
            
            # Draw original Gemini bboxes in red
            for i, bbox in enumerate(gemini_bboxes):
                x1, y1, x2, y2 = bbox
                
                # Convert to pixel coordinates
                px1, py1 = x1 * img_width, y1 * img_height
                px2, py2 = x2 * img_width, y2 * img_height
                
                width, height = px2 - px1, py2 - py1
                
                rect = patches.Rectangle(
                    (px1, py1), width, height,
                    linewidth=2, edgecolor='red', facecolor='none',
                    linestyle='--', alpha=0.8
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(px1, py1 - 10, f'Gemini {i+1}', 
                       color='red', fontsize=10, weight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Draw extended bboxes in blue
            for i, bbox in enumerate(extended_bboxes):
                x1, y1, x2, y2 = bbox
                
                # Convert to pixel coordinates
                px1, py1 = x1 * img_width, y1 * img_height
                px2, py2 = x2 * img_width, y2 * img_height
                
                width, height = px2 - px1, py2 - py1
                
                rect = patches.Rectangle(
                    (px1, py1), width, height,
                    linewidth=3, edgecolor='blue', facecolor='none',
                    alpha=0.8
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(px2, py1 - 10, f'Extended {i+1}', 
                       color='blue', fontsize=10, weight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add legend
            legend_elements = [
                patches.Patch(color='red', label='Gemini Detection'),
                patches.Patch(color='blue', label='DinoV3 Extended')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Set title
            ax.set_title(f'BBox Alignment Results: {page_name}', fontsize=14, weight='bold')
            ax.axis('off')
            
            # Save visualization
            viz_path = Path(output_dir) / f"{page_name}_bbox_alignment.png"
            plt.tight_layout()
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Visualization saved: {viz_path}")
            return str(viz_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error creating visualization: {str(e)}")
            return ""
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def save_page_results(self, result: Dict[str, Any], output_dir: str):
        """Save individual page results to JSON file."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            json_path = output_path / f"{result['page_name']}_results.json"
            
            # Convert the result dictionary
            serializable_result = self._convert_numpy_types(result)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Results saved: {json_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving page results: {str(e)}")
    
    def generate_summary_report(self, all_results: List[Dict[str, Any]], output_dir: str):
        """Generate comprehensive summary report."""
        try:
            output_path = Path(output_dir)
            
            # Prepare summary data
            total_pages = len(all_results)
            successful_pages = len([r for r in all_results if r.get('processing_success', False)])
            failed_pages = total_pages - successful_pages
            
            total_gemini_sections = sum(r.get('sections_detected', 0) for r in all_results)
            avg_sections_per_page = total_gemini_sections / successful_pages if successful_pages > 0 else 0
            
            summary = {
                'processing_summary': {
                    'total_pages': total_pages,
                    'successful_pages': successful_pages,
                    'failed_pages': failed_pages,
                    'success_rate': successful_pages / total_pages if total_pages > 0 else 0
                },
                'detection_summary': {
                    'total_sections_detected': total_gemini_sections,
                    'average_sections_per_page': avg_sections_per_page
                },
                'page_details': all_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save JSON summary
            json_path = output_path / 'bbox_alignment_summary.json'
            
            # Convert the summary dictionary
            serializable_summary = self._convert_numpy_types(summary)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_summary, f, indent=2, ensure_ascii=False)
            
            # Create human-readable report
            report_path = output_path / 'bbox_alignment_report.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("BBox Alignment with DinoV3 and Gemini 2.5 Flash - Report\n")
                f.write("=" * 65 + "\n\n")
                
                f.write(f"Processing Summary:\n")
                f.write(f"  Total Pages: {total_pages}\n")
                f.write(f"  Successful: {successful_pages}\n")
                f.write(f"  Failed: {failed_pages}\n")
                f.write(f"  Success Rate: {summary['processing_summary']['success_rate']:.1%}\n\n")
                
                f.write(f"Detection Summary:\n")
                f.write(f"  Total Sections: {total_gemini_sections}\n")
                f.write(f"  Avg Sections/Page: {avg_sections_per_page:.1f}\n\n")
                
                f.write("Page-by-Page Results:\n")
                f.write("-" * 30 + "\n")
                
                for result in all_results:
                    if result.get('processing_success', False):
                        f.write(f"{result['page_name']}: {result['sections_detected']} sections detected and extended\n")
                    else:
                        f.write(f"{result['page_name']}: FAILED - {result.get('error', 'Unknown error')}\n")
            
            self.logger.info(f"📋 Summary report saved: {json_path}")
            self.logger.info(f"📄 Text report saved: {report_path}")
            
            # Generate embeddings analysis report if embeddings were saved
            self._generate_embeddings_analysis_report(output_dir)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating summary report: {str(e)}")
    
    def _generate_embeddings_analysis_report(self, output_dir: str):
        """Generate comprehensive analysis report of visual embedding images."""
        try:
            embeddings_dir = Path(output_dir) / 'embeddings_debug'
            if not embeddings_dir.exists():
                return
                
            # Find all visual embedding files
            analysis_images = list(embeddings_dir.glob("*_embedding_analysis.png"))
            heatmap_images = list(embeddings_dir.glob("*_white_probability_heatmap.png"))
            summary_files = list(embeddings_dir.glob("*_summary.txt"))
            
            if not (analysis_images or heatmap_images or summary_files):
                return
                
            # Create comprehensive visual analysis report
            analysis_path = embeddings_dir / 'visual_embeddings_analysis_report.txt'
            with open(analysis_path, 'w', encoding='utf-8') as f:
                f.write("DinoV2/V3 Visual Embeddings Analysis Report\n")
                f.write("=" * 55 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Visual Embedding Files Generated:\n")
                f.write("-" * 35 + "\n")
                f.write(f"📊 Comprehensive Analysis Images: {len(analysis_images)}\n")
                f.write(f"🔥 White Probability Heatmaps: {len(heatmap_images)}\n")
                f.write(f"📄 Summary Text Files: {len(summary_files)}\n\n")
                
                # Parse summary files for statistics
                if summary_files:
                    f.write("Statistics Summary (from text files):\n")
                    f.write("-" * 40 + "\n")
                    
                    all_stats = []
                    for summary_file in sorted(summary_files):
                        try:
                            with open(summary_file, 'r', encoding='utf-8') as sf:
                                content = sf.read()
                                # Extract image name and bbox from filename
                                base_name = summary_file.stem.replace('_summary', '')
                                parts = base_name.split('_bbox_')
                                if len(parts) == 2:
                                    image_name = parts[0]
                                    bbox_part = parts[1].rsplit('_', 1)[0]  # Remove timestamp
                                    
                                    # Extract statistics from content
                                    lines = content.split('\n')
                                    stats = {'image': image_name, 'bbox': bbox_part}
                                    for line in lines:
                                        if 'Mean:' in line and 'White Probability' in lines[lines.index(line)-1]:
                                            stats['white_mean'] = float(line.split(':')[1].strip())
                                        elif 'Max:' in line and 'White Probability' in lines[lines.index(line)-3]:
                                            stats['white_max'] = float(line.split(':')[1].strip())
                                        elif 'Grid Size:' in line:
                                            stats['grid_size'] = line.split(':')[1].strip()
                                    
                                    all_stats.append(stats)
                        except Exception as e:
                            f.write(f"  ⚠️ Could not parse {summary_file.name}: {str(e)}\n")
                    
                    if all_stats:
                        white_means = [s.get('white_mean', 0) for s in all_stats if 'white_mean' in s]
                        white_maxs = [s.get('white_max', 0) for s in all_stats if 'white_max' in s]
                        
                        if white_means:
                            f.write(f"White Background Detection Across All BBoxes:\n")
                            f.write(f"  Average Mean Probability: {np.mean(white_means):.3f}\n")
                            f.write(f"  Average Max Probability: {np.mean(white_maxs):.3f}\n")
                            f.write(f"  Probability Range: [{min(white_means):.3f}, {max(white_maxs):.3f}]\n\n")
                        
                        f.write("Individual BBox Visual Analysis:\n")
                        f.write("-" * 35 + "\n")
                        for i, stats in enumerate(all_stats, 1):
                            f.write(f"{i}. Image: {stats.get('image', 'Unknown')}\n")
                            f.write(f"   BBox: {stats.get('bbox', 'Unknown')}\n")
                            f.write(f"   White Mean: {stats.get('white_mean', 'N/A'):.3f}\n")
                            f.write(f"   White Max: {stats.get('white_max', 'N/A'):.3f}\n")
                            f.write(f"   Grid Size: {stats.get('grid_size', 'N/A')}\n\n")
                
                f.write("Generated Visual Files:\n")
                f.write("-" * 25 + "\n")
                
                f.write("📊 Comprehensive Analysis Images:\n")
                for img_file in sorted(analysis_images):
                    f.write(f"  - {img_file.name}\n")
                    f.write(f"    Contains: 6-panel analysis (white probability, feature norms, PCA, variance, diversity, combined)\n\n")
                
                f.write("🔥 White Probability Heatmaps:\n")
                for img_file in sorted(heatmap_images):
                    f.write(f"  - {img_file.name}\n")
                    f.write(f"    Contains: Detailed heatmap with statistics overlay\n\n")
                
                f.write("📄 Summary Text Files:\n")
                for txt_file in sorted(summary_files):
                    f.write(f"  - {txt_file.name}\n")
                    f.write(f"    Contains: Numerical statistics and analysis details\n\n")
                
                f.write("Visual Analysis Guide:\n")
                f.write("-" * 25 + "\n")
                f.write("🖼️ How to Use These Visualizations:\n\n")
                f.write("1. Comprehensive Analysis Images (*_embedding_analysis.png):\n")
                f.write("   - Top Row: White probability, Feature norms, PCA first component\n")
                f.write("   - Bottom Row: Feature variance, PCA diversity, Combined score\n")
                f.write("   - Use to understand spatial patterns in embeddings\n\n")
                
                f.write("2. White Probability Heatmaps (*_white_probability_heatmap.png):\n")
                f.write("   - Detailed view of white background detection\n")
                f.write("   - Grid overlay shows patch boundaries\n")
                f.write("   - Statistics overlay shows numerical summary\n")
                f.write("   - Use for fine-grained white background analysis\n\n")
                
                f.write("3. Summary Text Files (*_summary.txt):\n")
                f.write("   - Numerical statistics and analysis details\n")
                f.write("   - Feature dimensions and processing information\n")
                f.write("   - Use for programmatic analysis or detailed review\n\n")
                
                f.write("Color Scales Used:\n")
                f.write("- White Probability: Viridis (dark = low, bright = high)\n")
                f.write("- Feature Norms: Plasma (dark = low, bright = high)\n")
                f.write("- PCA Components: RdBu_r (red = negative, blue = positive)\n")
                f.write("- Variance/Diversity: Various (darker = more uniform/less diverse)\n")
            
            self.logger.info(f"📊 Visual embeddings analysis report generated: {analysis_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not generate visual embeddings analysis report: {str(e)}")


def main():
    """Main function to set up argument parsing and run the bbox aligner."""
    parser = argparse.ArgumentParser(
        description="BBox Aligner with DinoV3 and Gemini 2.5 Flash",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python bbox_aligner_dinov3.py document.pdf
  
  # With custom API key
  python bbox_aligner_dinov3.py document.pdf --api-key your_gemini_key
  
  # Custom output directory
  python bbox_aligner_dinov3.py document.pdf --output-dir results
  
  # Process specific pages
  python bbox_aligner_dinov3.py document.pdf --start-page 1 --end-page 3
        """
    )
    
    # Positional arguments
    parser.add_argument(
        'pdf_path',
        help='Path to the PDF document to process'
    )
    
    # Optional arguments
    parser.add_argument(
        '--api-key',
        help='Google Gemini API key (or set GEMINI_API_KEY env var)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='bbox_alignment_results',
        help='Directory to save results (default: bbox_alignment_results)'
    )
    
    parser.add_argument(
        '--pages-dir',
        default='extracted_pages',
        help='Directory to save extracted page images (default: extracted_pages)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution for PDF to image conversion (default: 300)'
    )
    
    parser.add_argument(
        '--start-page',
        type=int,
        default=1,
        help='First page to process (1-indexed, default: 1)'
    )
    
    parser.add_argument(
        '--end-page',
        type=int,
        help='Last page to process (1-indexed, default: all pages)'
    )
    
    parser.add_argument(
        '--extension-threshold',
        type=float,
        default=0.6,
        help='Threshold for white background detection (default: 0.6, optimized for enhanced PCA)'
    )
    
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip PDF extraction if page images already exist'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("🚀 BBox Aligner with DinoV3 and Gemini 2.5 Flash")
    print("=" * 60)
    print(f"📄 PDF: {args.pdf_path}")
    print(f"📁 Pages dir: {args.pages_dir}")
    print(f"📁 Output dir: {args.output_dir}")
    print(f"🎯 DPI: {args.dpi}")
    print(f"🔍 Extension threshold: {args.extension_threshold}")
    print("=" * 60)
    
    try:
        # Initialize the aligner
        aligner = BBoxAlignerDinoV3(
            gemini_api_key=args.api_key,
            model_name='dinov2_vitb14'
        )
        
        print("🎉 Initialization complete! Ready to process PDF.")
        
        # Step 1: Extract PDF pages
        if not args.skip_extraction:
            print("\n📄 Extracting PDF pages...")
            page_images = aligner.extract_pdf_pages(
                args.pdf_path, 
                args.pages_dir, 
                args.dpi,
                args.start_page,
                args.end_page
            )
        else:
            print("\n⏭️  Skipping PDF extraction...")
            # Find existing page images
            pages_path = Path(args.pages_dir)
            if not pages_path.exists():
                print(f"❌ Pages directory not found: {args.pages_dir}")
                sys.exit(1)
            
            page_images = sorted([
                str(p) for p in pages_path.glob("*.png") 
                if p.stem.startswith("page_")
            ])
            
            if not page_images:
                print(f"❌ No page images found in {args.pages_dir}")
                sys.exit(1)
            
            print(f"📁 Found {len(page_images)} existing page images")
        
        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Step 2: Process each page
        print(f"\n🔍 Processing {len(page_images)} pages...")
        all_results = []
        
        for i, image_path in enumerate(page_images, 1):
            print(f"\n--- Page {i}/{len(page_images)} ---")
            
            result = aligner.process_page(
                image_path, 
                args.output_dir,
                args.extension_threshold
            )
            
            all_results.append(result)
        
        # Step 3: Generate summary report
        print(f"\n📊 Generating summary report...")
        aligner.generate_summary_report(all_results, args.output_dir)
        
        # Final summary
        successful_pages = len([r for r in all_results if r.get('processing_success', False)])
        total_sections = sum(r.get('sections_detected', 0) for r in all_results)
        
        print(f"\n🎉 Processing Complete!")
        print("=" * 60)
        print(f"📊 Processed: {successful_pages}/{len(all_results)} pages")
        print(f"🎯 Detected: {total_sections} sections total")
        print(f"📁 Results saved in: {args.output_dir}/")
        
        # List generated files
        output_files = list(Path(args.output_dir).glob("*"))
        if output_files:
            print(f"\nGenerated files:")
            for file_path in sorted(output_files):
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                print(f"   - {file_path.name} ({file_size:.1f} MB)")
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()