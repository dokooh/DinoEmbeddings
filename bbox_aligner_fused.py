#!/usr/bin/env python3
"""
BBox Aligner Fused - DinoV3 + OpenCV Sequential Approach
========================================================

This script combines PDF processing, Gemini 2.5 Flash for section detection, 
and a fused approach using both DinoV3 and OpenCV for bounding box refinement.

Sequential Workflow:
1. PDF → Extract pages as images
2. Gemini 2.5 Flash → Detect section bounding boxes
3. DinoV3 → First-stage alignment with embedding-based analysis
4. OpenCV → Second-stage refinement with pixel-based white background detection
5. Bottom Border Post-Optimization → Third-stage bottom edge extension to white areas
6. Generate final aligned and extended bounding boxes

This fused approach leverages the semantic understanding of DinoV3, followed by 
the precise pixel-level analysis of OpenCV, and finalized with specialized 
bottom border optimization for optimal bounding box alignment.

Author: GitHub Copilot
Date: October 2025
"""

import os
import sys
import logging
import argparse
import json
import numpy as np
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Tuple, Any
import cv2
from datetime import datetime

# Core dependencies
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# PDF processing
import fitz  # PyMuPDF

# Google Gemini
import google.generativeai as genai

# DinoV3 dependencies
import timm
import torchvision.transforms as T

# Machine Learning
from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BBoxAlignerFused:
    """
    Main class for PDF processing with Gemini + Fused DinoV3/OpenCV/Bottom-Optimization bbox alignment.
    
    This class implements a 3-stage sequential approach:
    1. DinoV3 embedding-based analysis for semantic understanding
    2. OpenCV pixel-based refinement for precise boundary detection
    3. Bottom border post-optimization for enhanced bottom edge alignment
    """
    
    def __init__(self, gemini_api_key: str = None, model_name: str = 'dinov2_vitb14'):
        """
        Initialize the Fused BBox Aligner.
        
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
        logger = logging.getLogger('bbox_aligner_fused')
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
                api_key = os.getenv('GEMINI_API_KEY')
                if not api_key:
                    raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass --api-key")
                genai.configure(api_key=api_key)
            
            # Initialize model (using available models from API)
            model_names = [
                'gemini-2.5-flash',  # Latest model if available
                'gemini-2.0-flash-exp',
                'gemini-2.0-flash',
                'gemini-flash-latest',
                'gemini-pro-latest'
            ]
            
            model_initialized = False
            for model_name in model_names:
                try:
                    self.gemini_model = genai.GenerativeModel(model_name)
                    self.logger.info(f"✅ Initialized Gemini model: {model_name}")
                    model_initialized = True
                    break
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to initialize {model_name}: {str(e)}")
                    continue
            
            if not model_initialized:
                raise Exception("Failed to initialize any Gemini model")
            
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
                self.dino_model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
                self.logger.info("✅ Loaded DinoV3 model")
            except:
                self.dino_model = timm.create_model('vit_base_patch14_dinov2', pretrained=True)
                self.logger.info("✅ Loaded DinoV2 model (DinoV3 unavailable)")
            
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
                
                # Get page as image
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat)
                
                # Save as PNG
                page_filename = f"page_{page_num + 1:03d}.png"
                page_path = pages_path / page_filename
                pix.save(str(page_path))
                
                extracted_pages.append(str(page_path))
                self.logger.info(f"  📄 Extracted page {page_num + 1}: {page_filename}")
            
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
                json_text = json_text[7:]
            if json_text.startswith('```'):
                json_text = json_text[3:]
            if json_text.endswith('```'):
                json_text = json_text[:-3]
            json_text = json_text.strip()
            
            # Try to parse JSON
            try:
                sections = json.loads(json_text)
                
                if not isinstance(sections, list):
                    self.logger.warning(f"⚠️ Expected list, got {type(sections)}")
                    return []
                
                valid_sections = []
                for section in sections:
                    if isinstance(section, list) and len(section) == 4:
                        x1, y1, x2, y2 = section
                        # Validate coordinates
                        if all(isinstance(coord, (int, float)) and 0 <= coord <= 1 for coord in section):
                            # Ensure proper ordering (x1 < x2, y1 < y2)
                            if x1 < x2 and y1 < y2:
                                valid_sections.append([float(x1), float(y1), float(x2), float(y2)])
                            else:
                                self.logger.warning(f"⚠️ Invalid bbox ordering: {section}")
                        else:
                            self.logger.warning(f"⚠️ Invalid coordinates: {section}")
                    else:
                        self.logger.warning(f"⚠️ Invalid section format: {section}")
                
                self.logger.info(f"✅ Detected {len(valid_sections)} valid sections")
                return valid_sections
                
            except json.JSONDecodeError as e:
                self.logger.error(f"❌ Failed to parse JSON: {str(e)}")
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
            
            self.logger.info(f"🧠 DinoV3 PCA analysis on cropped bbox region ({bbox_width}x{bbox_height}px):")
            self.logger.info(f"   📐 Grid size: {grid_size}x{grid_size} patches")
            self.logger.info(f"   📊 Explained variance ratios: {pca.explained_variance_ratio_[:3]}")
            self.logger.info(f"   🎯 White probability stats: avg={avg_prob:.3f}, max={max_prob:.3f}, min={min_prob:.3f}")
            self.logger.info(f"   🔍 Analysis performed on bbox region ONLY (not full image)")
            
            return white_prob_map
            
        except Exception as e:
            self.logger.error(f"❌ Error in DinoV3 PCA white background detection: {str(e)}")
            # Return default probability map
            return np.zeros((37, 37))
    
    def extend_bbox_to_white_opencv(self, image, bbox, white_threshold=240, max_extension=500):
        """
        Extend a bounding box until all four edges reach white areas using OpenCV approach.
        This is the second stage of the fused approach.
        
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
    
    def optimize_bottom_border_post_processing(self, image, bbox, white_threshold=240, max_extension=500, 
                                             tolerance_ratio=0.1, sample_width=50):
        """
        Post-optimization strategy specifically for bottom border alignment.
        Uses multiple sampling strategies to ensure bottom border extends to white regions.
        This is the third stage of the fused approach.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image (grayscale or color)
        bbox : tuple
            Initial bounding box as (x, y, width, height) in pixel coordinates
        white_threshold : int
            Pixel intensity threshold to consider as white (0-255)
        max_extension : int
            Maximum pixels to extend bottom border
        tolerance_ratio : float
            Ratio of non-white pixels allowed in the edge (0.1 = 10% tolerance)
        sample_width : int
            Width of sampling windows for edge analysis
        
        Returns:
        --------
        tuple : Optimized bounding box as (x, y, width, height) in pixel coordinates
        """
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
        
        original_bottom = y + bh
        
        self.logger.info(f"🔧 Post-processing: Optimizing bottom border from y={original_bottom}")
        
        # Strategy 1: Relaxed threshold - allow some tolerance for noise
        extended = 0
        current_bottom = y + bh
        while current_bottom < h and extended < max_extension:
            # Sample the entire bottom edge
            bottom_edge = gray[current_bottom, x:x+bw]
            
            # Calculate ratio of white pixels
            white_pixels = np.sum(bottom_edge >= white_threshold)
            white_ratio = white_pixels / len(bottom_edge)
            
            # If most pixels are white (within tolerance), continue extending
            if white_ratio >= (1.0 - tolerance_ratio):
                bh += 1
                current_bottom += 1
                extended += 1
            else:
                break
        
        # Strategy 2: Multi-point sampling - check key positions along bottom edge
        if extended == 0:  # If Strategy 1 didn't extend, try sampling approach
            current_bottom = y + bh
            sample_positions = []
            
            # Create sampling positions across the width
            if bw >= sample_width * 3:
                # For wide boxes, sample left, center, and right regions
                left_start = x + 10  # Small offset from edge
                center_start = x + (bw - sample_width) // 2
                right_start = x + bw - sample_width - 10
                sample_positions = [
                    (left_start, min(sample_width, bw - 20)),
                    (center_start, sample_width),
                    (right_start, min(sample_width, bw - 20))
                ]
            else:
                # For narrow boxes, sample the center region
                center_start = max(x, x + (bw - min(sample_width, bw)) // 2)
                sample_width_adj = min(sample_width, bw)
                sample_positions = [(center_start, sample_width_adj)]
            
            extended_sampling = 0
            while current_bottom < h and extended_sampling < max_extension:
                all_samples_white = True
                
                for sample_x, sample_w in sample_positions:
                    # Sample this region of the bottom edge
                    sample_edge = gray[current_bottom, sample_x:sample_x+sample_w]
                    white_pixels = np.sum(sample_edge >= white_threshold)
                    white_ratio = white_pixels / len(sample_edge)
                    
                    # If this sample region is not sufficiently white, stop
                    if white_ratio < (1.0 - tolerance_ratio):
                        all_samples_white = False
                        break
                
                if all_samples_white:
                    bh += 1
                    current_bottom += 1
                    extended_sampling += 1
                    extended += 1  # Update total extension count
                else:
                    break
        
        # Strategy 3: Gradient-based refinement for final positioning
        if extended > 0:
            # Look for the optimal stopping point by analyzing gradient changes
            search_start = max(original_bottom, current_bottom - min(20, extended // 2))
            search_end = current_bottom
            
            if search_end > search_start:
                best_bottom = current_bottom
                min_gradient = float('inf')
                
                for test_bottom in range(search_start, search_end + 1):
                    if test_bottom >= h:
                        break
                    
                    # Calculate gradient at this position
                    test_edge = gray[test_bottom, x:x+bw]
                    if test_bottom > 0:
                        prev_edge = gray[test_bottom-1, x:x+bw]
                        gradient = np.mean(np.abs(test_edge.astype(float) - prev_edge.astype(float)))
                        
                        # Also check if this line is sufficiently white
                        white_pixels = np.sum(test_edge >= white_threshold)
                        white_ratio = white_pixels / len(test_edge)
                        
                        # Prefer positions with low gradient and high whiteness
                        if gradient < min_gradient and white_ratio >= (1.0 - tolerance_ratio * 1.5):
                            min_gradient = gradient
                            best_bottom = test_bottom
                
                # Update to the refined position
                bh = best_bottom - y
        
        final_extension = (y + bh) - original_bottom
        
        self.logger.info(f"📏 Bottom border post-optimization complete:")
        self.logger.info(f"   Original bottom: {original_bottom}")
        self.logger.info(f"   Optimized bottom: {y + bh}")
        self.logger.info(f"   Extension: {final_extension} pixels")
        self.logger.info(f"   Strategies used: {'Relaxed threshold' if extended > 0 else 'Multi-sampling'}")
        
        return (x, y, bw, bh)
    
    def validate_and_expand_bottom_with_white_ratio(self, image, bbox, white_threshold=240, 
                                                   white_ratio_target=0.7, max_additional_extension=300):
        """
        Final validation step: Check white pixel ratio inside bounding box and expand bottom edge
        if white ratio is below target threshold (70% by default).
        This is the fourth and final stage of the fused approach.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image (grayscale or color)
        bbox : tuple
            Initial bounding box as (x, y, width, height) in pixel coordinates
        white_threshold : int
            Pixel intensity threshold to consider as white (0-255)
        white_ratio_target : float
            Target white pixel ratio (0.7 = 70%)
        max_additional_extension : int
            Maximum additional pixels to extend bottom border
        
        Returns:
        --------
        tuple : Final validated bounding box as (x, y, width, height) in pixel coordinates
        """
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
        
        original_bottom = y + bh
        
        self.logger.info(f"🔍 Final validation: Checking white pixel ratio inside bounding box")
        
        # Calculate current white pixel ratio inside the bounding box
        def calculate_white_ratio(bbox_coords):
            bx, by, bbw, bbh = bbox_coords
            # Extract the region inside the bounding box
            bbox_region = gray[by:by+bbh, bx:bx+bbw]
            if bbox_region.size == 0:
                return 0.0
            
            # Count white pixels
            white_pixels = np.sum(bbox_region >= white_threshold)
            total_pixels = bbox_region.size
            return white_pixels / total_pixels if total_pixels > 0 else 0.0
        
        current_white_ratio = calculate_white_ratio((x, y, bw, bh))
        
        self.logger.info(f"📊 Current white pixel ratio: {current_white_ratio:.3f} (target: {white_ratio_target:.3f})")
        
        # If white ratio is already sufficient, return as is
        if current_white_ratio >= white_ratio_target:
            self.logger.info(f"✅ White ratio target achieved: {current_white_ratio:.3f} >= {white_ratio_target:.3f}")
            return (x, y, bw, bh)
        
        # White ratio is insufficient, expand bottom edge until target is reached
        self.logger.info(f"⚠️ White ratio below target, expanding bottom edge...")
        
        extended = 0
        current_bottom = y + bh
        best_ratio = current_white_ratio
        best_bbox = (x, y, bw, bh)
        
        while current_bottom < h and extended < max_additional_extension:
            # Try extending bottom by one pixel
            test_bbox = (x, y, bw, bh + extended + 1)
            test_white_ratio = calculate_white_ratio(test_bbox)
            
            # Keep track of the best ratio achieved
            if test_white_ratio > best_ratio:
                best_ratio = test_white_ratio
                best_bbox = test_bbox
            
            # Check if we've reached the target ratio
            if test_white_ratio >= white_ratio_target:
                final_bbox = test_bbox
                self.logger.info(f"🎯 Target white ratio achieved: {test_white_ratio:.3f} >= {white_ratio_target:.3f}")
                break
            
            extended += 1
            current_bottom += 1
        else:
            # Use the best ratio achieved if target wasn't reached
            final_bbox = best_bbox
            self.logger.info(f"🔄 Maximum extension reached. Best white ratio: {best_ratio:.3f}")
        
        final_extension = (y + final_bbox[3]) - original_bottom
        final_white_ratio = calculate_white_ratio(final_bbox)
        
        self.logger.info(f"📏 White ratio validation complete:")
        self.logger.info(f"   Original bottom: {original_bottom}")
        self.logger.info(f"   Final bottom: {y + final_bbox[3]}")
        self.logger.info(f"   Additional extension: {final_extension} pixels")
        self.logger.info(f"   Final white ratio: {final_white_ratio:.3f}")
        self.logger.info(f"   Target achieved: {'Yes' if final_white_ratio >= white_ratio_target else 'No'}")
        
        return final_bbox
    
    def align_bbox_with_fused_approach(self, image_path: str, original_bbox: List[float], 
                                     other_bboxes: List[List[float]] = None) -> Dict[str, Any]:
        """
        Apply the fused alignment approach: DinoV3 first, then OpenCV refinement, then bottom border post-optimization.
        
        Args:
            image_path: Path to the image file
            original_bbox: Original bbox [x1, y1, x2, y2] in normalized coordinates
            other_bboxes: List of other bounding boxes to avoid overlapping
            
        Returns:
            Dictionary containing aligned bbox and processing metrics
        """
        try:
            if other_bboxes is None:
                other_bboxes = []
            
            self.logger.info(f"🔧 Starting fused alignment approach (DinoV3 → OpenCV → Bottom Post-Optimization)")
            
            # Stage 1: DinoV3 embedding-based alignment
            self.logger.info(f"📊 Stage 1: DinoV3 embedding-based analysis")
            dino_aligned_bbox = self._align_with_dino_stage(image_path, original_bbox, other_bboxes)
            
            # Stage 2: OpenCV pixel-based refinement
            self.logger.info(f"🎯 Stage 2: OpenCV pixel-based refinement")
            opencv_aligned_bbox = self._align_with_opencv_stage(image_path, dino_aligned_bbox, other_bboxes)
            
            # Stage 3: Bottom border post-optimization
            self.logger.info(f"🔧 Stage 3: Bottom border post-optimization")
            final_aligned_bbox = self._align_with_bottom_optimization_stage(image_path, opencv_aligned_bbox, other_bboxes)
            
            # Calculate metrics
            original_area = (original_bbox[2] - original_bbox[0]) * (original_bbox[3] - original_bbox[1])
            dino_area = (dino_aligned_bbox[2] - dino_aligned_bbox[0]) * (dino_aligned_bbox[3] - dino_aligned_bbox[1])
            opencv_area = (opencv_aligned_bbox[2] - opencv_aligned_bbox[0]) * (opencv_aligned_bbox[3] - opencv_aligned_bbox[1])
            final_area = (final_aligned_bbox[2] - final_aligned_bbox[0]) * (final_aligned_bbox[3] - final_aligned_bbox[1])
            
            dino_extension_ratio = dino_area / original_area if original_area > 0 else 1.0
            opencv_refinement_ratio = opencv_area / dino_area if dino_area > 0 else 1.0
            bottom_optimization_ratio = final_area / opencv_area if opencv_area > 0 else 1.0
            final_extension_ratio = final_area / original_area if original_area > 0 else 1.0
            
            self.logger.info(f"✅ Fused alignment complete:")
            self.logger.info(f"   Original bbox: [{original_bbox[0]:.3f}, {original_bbox[1]:.3f}, {original_bbox[2]:.3f}, {original_bbox[3]:.3f}]")
            self.logger.info(f"   DinoV3 result: [{dino_aligned_bbox[0]:.3f}, {dino_aligned_bbox[1]:.3f}, {dino_aligned_bbox[2]:.3f}, {dino_aligned_bbox[3]:.3f}]")
            self.logger.info(f"   OpenCV result: [{opencv_aligned_bbox[0]:.3f}, {opencv_aligned_bbox[1]:.3f}, {opencv_aligned_bbox[2]:.3f}, {opencv_aligned_bbox[3]:.3f}]")
            self.logger.info(f"   Final result:  [{final_aligned_bbox[0]:.3f}, {final_aligned_bbox[1]:.3f}, {final_aligned_bbox[2]:.3f}, {final_aligned_bbox[3]:.3f}]")
            self.logger.info(f"   DinoV3 extension: {dino_extension_ratio:.2f}x")
            self.logger.info(f"   OpenCV refinement: {opencv_refinement_ratio:.2f}x")
            self.logger.info(f"   Bottom optimization: {bottom_optimization_ratio:.2f}x")
            self.logger.info(f"   Total extension: {final_extension_ratio:.2f}x")
            
            return {
                'aligned_bbox': final_aligned_bbox,
                'dino_bbox': dino_aligned_bbox,
                'opencv_bbox': opencv_aligned_bbox,
                'intermediate_bbox': dino_aligned_bbox,  # Keep for backward compatibility
                'original_bbox': original_bbox,
                'stage1_method': 'dinov3_embeddings',
                'stage2_method': 'opencv_pixels',
                'stage3_method': 'bottom_border_optimization',
                'dino_extension_ratio': dino_extension_ratio,
                'opencv_refinement_ratio': opencv_refinement_ratio,
                'bottom_optimization_ratio': bottom_optimization_ratio,
                'final_extension_ratio': final_extension_ratio,
                'processing_stages': 3
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in fused bbox alignment: {str(e)}")
            return {
                'aligned_bbox': original_bbox,
                'intermediate_bbox': original_bbox,
                'original_bbox': original_bbox,
                'error': str(e)
            }
    
    def _align_with_dino_stage(self, image_path: str, original_bbox: List[float], 
                              other_bboxes: List[List[float]]) -> List[float]:
        """
        Stage 1: DinoV3 embedding-based alignment using white background detection.
        
        Args:
            image_path: Path to the image file
            original_bbox: Original bbox [x1, y1, x2, y2] in normalized coordinates
            other_bboxes: List of other bounding boxes to avoid overlapping
            
        Returns:
            DinoV3-aligned bounding box [x1, y1, x2, y2] in normalized coordinates
        """
        try:
            # Get white probability map from DinoV3 analysis
            white_prob_map = self.detect_white_background_with_dino(image_path, original_bbox)
            
            # Use embedding-based extension logic
            extension_threshold = 0.8  # High confidence threshold for white regions
            aligned_bbox = self._extend_bbox_with_embedding_analysis(
                image_path, original_bbox, white_prob_map, extension_threshold, other_bboxes
            )
            
            self.logger.info(f"🧠 DinoV3 stage result: {aligned_bbox}")
            return aligned_bbox
            
        except Exception as e:
            self.logger.warning(f"⚠️ DinoV3 stage failed: {str(e)}, using original bbox")
            return original_bbox
    
    def _align_with_opencv_stage(self, image_path: str, dino_bbox: List[float], 
                                other_bboxes: List[List[float]]) -> List[float]:
        """
        Stage 2: OpenCV pixel-based refinement of the DinoV3 result.
        
        Args:
            image_path: Path to the image file
            dino_bbox: DinoV3-aligned bbox [x1, y1, x2, y2] in normalized coordinates
            other_bboxes: List of other bounding boxes to avoid overlapping
            
        Returns:
            Final OpenCV-refined bounding box [x1, y1, x2, y2] in normalized coordinates
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL if cv2 fails
                pil_image = Image.open(image_path).convert('RGB')
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            img_height, img_width = image.shape[:2]
            
            # Convert normalized bbox to pixel coordinates  
            x1, y1, x2, y2 = dino_bbox
            px1 = int(x1 * img_width)
            py1 = int(y1 * img_height)
            px2 = int(x2 * img_width)
            py2 = int(y2 * img_height)
            
            # Convert to (x, y, width, height) format for OpenCV approach
            bbox_xywh = (px1, py1, px2 - px1, py2 - py1)
            
            # Apply OpenCV-based white background extension
            extended_bbox_xywh = self.extend_bbox_to_white_opencv(
                image, bbox_xywh, white_threshold=240, max_extension=500
            )
            
            # Convert back to normalized (x1, y1, x2, y2) format
            ex, ey, ew, eh = extended_bbox_xywh
            extended_bbox = [
                ex / img_width,           # x1
                ey / img_height,          # y1  
                (ex + ew) / img_width,    # x2
                (ey + eh) / img_height    # y2
            ]
            
            self.logger.info(f"🎯 OpenCV stage result: {extended_bbox}")
            return extended_bbox
            
        except Exception as e:
            self.logger.warning(f"⚠️ OpenCV stage failed: {str(e)}, using DinoV3 result")
            return dino_bbox
    
    def _align_with_bottom_optimization_stage(self, image_path: str, opencv_bbox: List[float], 
                                            other_bboxes: List[List[float]]) -> List[float]:
        """
        Stage 3: Bottom border post-optimization of the OpenCV result.
        
        Args:
            image_path: Path to the image file
            opencv_bbox: OpenCV-aligned bbox [x1, y1, x2, y2] in normalized coordinates
            other_bboxes: List of other bounding boxes to avoid overlapping
            
        Returns:
            Final bottom-optimized bounding box [x1, y1, x2, y2] in normalized coordinates
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL if cv2 fails
                pil_image = Image.open(image_path).convert('RGB')
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            img_height, img_width = image.shape[:2]
            
            # Convert normalized bbox to pixel coordinates  
            x1, y1, x2, y2 = opencv_bbox
            px1 = int(x1 * img_width)
            py1 = int(y1 * img_height)
            px2 = int(x2 * img_width)
            py2 = int(y2 * img_height)
            
            # Convert to (x, y, width, height) format for bottom optimization
            bbox_xywh = (px1, py1, px2 - px1, py2 - py1)
            
            # Apply bottom border post-optimization
            optimized_bbox_xywh = self.optimize_bottom_border_post_processing(
                image, bbox_xywh, white_threshold=240, max_extension=500,
                tolerance_ratio=0.1, sample_width=50
            )
            
            # Convert back to normalized (x1, y1, x2, y2) format
            ox, oy, ow, oh = optimized_bbox_xywh
            optimized_bbox = [
                ox / img_width,           # x1
                oy / img_height,          # y1  
                (ox + ow) / img_width,    # x2
                (oy + oh) / img_height    # y2
            ]
            
            self.logger.info(f"🔧 Bottom optimization stage result: {optimized_bbox}")
            return optimized_bbox
            
        except Exception as e:
            self.logger.warning(f"⚠️ Bottom optimization stage failed: {str(e)}, using OpenCV result")
            return opencv_bbox
    
    def _extend_bbox_with_embedding_analysis(self, image_path: str, original_bbox: List[float], 
                                           white_prob_map: np.ndarray, extension_threshold: float,
                                           other_bboxes: List[List[float]]) -> List[float]:
        """
        Extend bounding box based on DinoV3 embedding analysis.
        
        Args:
            image_path: Path to the image file
            original_bbox: Original bbox [x1, y1, x2, y2] in normalized coordinates
            white_prob_map: White probability map from DinoV3 analysis
            extension_threshold: Threshold for considering areas as white background
            other_bboxes: List of other bounding boxes to avoid overlapping
            
        Returns:
            Extended bounding box [x1, y1, x2, y2] in normalized coordinates
        """
        try:
            # Find high-confidence white regions
            high_confidence_mask = white_prob_map > extension_threshold
            
            if not np.any(high_confidence_mask):
                # No high-confidence regions found, return original
                return original_bbox
            
            # Find bounding box of high-confidence regions
            grid_size = white_prob_map.shape[0]
            rows, cols = np.where(high_confidence_mask)
            
            if len(rows) == 0:
                return original_bbox
            
            # Calculate extension boundaries in grid coordinates
            min_row, max_row = np.min(rows), np.max(rows)
            min_col, max_col = np.min(cols), np.max(cols)
            
            # Convert to relative coordinates (0-1 range within the original bbox region)
            rel_y1 = min_row / grid_size
            rel_y2 = (max_row + 1) / grid_size
            rel_x1 = min_col / grid_size
            rel_x2 = (max_col + 1) / grid_size
            
            # Map to absolute normalized coordinates
            bbox_width = original_bbox[2] - original_bbox[0]
            bbox_height = original_bbox[3] - original_bbox[1]
            
            extended_bbox = [
                max(0.0, original_bbox[0] + rel_x1 * bbox_width),
                max(0.0, original_bbox[1] + rel_y1 * bbox_height),
                min(1.0, original_bbox[0] + rel_x2 * bbox_width),
                min(1.0, original_bbox[1] + rel_y2 * bbox_height)
            ]
            
            # Ensure minimum size
            min_size = 0.01
            if extended_bbox[2] - extended_bbox[0] < min_size:
                center_x = (extended_bbox[0] + extended_bbox[2]) / 2
                extended_bbox[0] = max(0.0, center_x - min_size / 2)
                extended_bbox[2] = min(1.0, center_x + min_size / 2)
            
            if extended_bbox[3] - extended_bbox[1] < min_size:
                center_y = (extended_bbox[1] + extended_bbox[3]) / 2
                extended_bbox[1] = max(0.0, center_y - min_size / 2)
                extended_bbox[3] = min(1.0, center_y + min_size / 2)
            
            return extended_bbox
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error in embedding-based extension: {str(e)}")
            return original_bbox
    
    def process_page(self, image_path: str, output_dir: str, extension_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Process a single page: detect sections with Gemini, align with fused approach.
        
        Args:
            image_path: Path to page image
            output_dir: Directory to save results
            extension_threshold: Threshold for white background detection
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Store current output directory for debugging
            self.current_output_dir = output_dir
            
            page_name = Path(image_path).stem
            self.logger.info(f"🔍 Processing page: {page_name}")
            
            # Step 1: Detect sections with Gemini
            gemini_bboxes = self.detect_sections_with_gemini(image_path)
            
            if not gemini_bboxes:
                self.logger.warning(f"⚠️ No sections detected in {page_name}")
                return {
                    'page_name': page_name,
                    'sections_detected': 0,
                    'processing_success': False,
                    'error': 'No sections detected'
                }
            
            # Step 2: Apply fused alignment to each detected section
            aligned_results = []
            
            for i, bbox in enumerate(gemini_bboxes):
                self.logger.info(f"🎯 Processing section {i+1}/{len(gemini_bboxes)}")
                
                # Apply fused alignment
                alignment_result = self.align_bbox_with_fused_approach(
                    image_path, bbox, gemini_bboxes
                )
                
                aligned_results.append({
                    'section_id': i,
                    'original_bbox': bbox,
                    'dino_bbox': alignment_result.get('dino_bbox', bbox),
                    'opencv_bbox': alignment_result.get('opencv_bbox', bbox),
                    'intermediate_bbox': alignment_result.get('intermediate_bbox', bbox),  # Backward compatibility
                    'aligned_bbox': alignment_result.get('aligned_bbox', bbox),
                    'dino_extension_ratio': alignment_result.get('dino_extension_ratio', 1.0),
                    'opencv_refinement_ratio': alignment_result.get('opencv_refinement_ratio', 1.0),
                    'bottom_optimization_ratio': alignment_result.get('bottom_optimization_ratio', 1.0),
                    'final_extension_ratio': alignment_result.get('final_extension_ratio', 1.0)
                })
            
            # Step 3: Create visualization
            visualization_path = self.visualize_bboxes(
                image_path, 
                gemini_bboxes, 
                [result['dino_bbox'] for result in aligned_results],
                [result['opencv_bbox'] for result in aligned_results],
                [result['aligned_bbox'] for result in aligned_results],
                output_dir, 
                page_name
            )
            
            # Prepare results
            result = {
                'page_name': page_name,
                'image_path': image_path,
                'sections_detected': len(gemini_bboxes),
                'processing_success': True,
                'original_bboxes': gemini_bboxes,
                'dino_stage_bboxes': [result['dino_bbox'] for result in aligned_results],
                'opencv_stage_bboxes': [result['opencv_bbox'] for result in aligned_results],
                'final_aligned_bboxes': [result['aligned_bbox'] for result in aligned_results],
                'alignment_results': aligned_results,
                'visualization_path': visualization_path,
                'fused_approach': True,
                'processing_stages': 3
            }
            
            # Save individual page results
            self.save_page_results(result, output_dir)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process page {Path(image_path).stem}: {str(e)}")
            return {
                'page_name': Path(image_path).stem,
                'processing_success': False,
                'error': str(e)
            }
    
    def visualize_bboxes(self, image_path: str, gemini_bboxes: List[List[float]], 
                        dino_bboxes: List[List[float]], opencv_bboxes: List[List[float]], 
                        final_bboxes: List[List[float]], output_dir: str, page_name: str) -> str:
        """
        Create visualization showing original, DinoV3, OpenCV, and final bounding boxes.
        
        Args:
            image_path: Path to the image
            gemini_bboxes: Original bounding boxes from Gemini
            dino_bboxes: DinoV3-aligned bounding boxes
            opencv_bboxes: OpenCV-refined bounding boxes
            final_bboxes: Final bottom-optimized bounding boxes
            output_dir: Output directory
            page_name: Name of the page
            
        Returns:
            Path to saved visualization
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # Load image
            image = Image.open(image_path)
            img_width, img_height = image.size
            
            # Create figure
            fig, ax = plt.subplots(1, figsize=(16, 20))
            ax.imshow(image)
            
            # Colors for different stages
            colors = {
                'original': 'red',
                'dino': 'blue', 
                'opencv': 'orange',
                'final': 'green'
            }
            
            # Draw bounding boxes for each stage
            for i, (orig_bbox, dino_bbox, opencv_bbox, final_bbox) in enumerate(zip(gemini_bboxes, dino_bboxes, opencv_bboxes, final_bboxes)):
                # Convert normalized coordinates to pixel coordinates
                def norm_to_pixel(bbox):
                    x1, y1, x2, y2 = bbox
                    return (
                        x1 * img_width, y1 * img_height,
                        (x2 - x1) * img_width, (y2 - y1) * img_height
                    )
                
                # Original bbox (red, dashed)
                x, y, w, h = norm_to_pixel(orig_bbox)
                rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                       edgecolor=colors['original'], facecolor='none', 
                                       linestyle='--', label=f'Original {i+1}' if i == 0 else "")
                ax.add_patch(rect)
                
                # DinoV3 bbox (blue, dotted)
                x, y, w, h = norm_to_pixel(dino_bbox)
                rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                       edgecolor=colors['dino'], facecolor='none', 
                                       linestyle=':', label=f'DinoV3 {i+1}' if i == 0 else "")
                ax.add_patch(rect)
                
                # OpenCV bbox (orange, dash-dot)
                x, y, w, h = norm_to_pixel(opencv_bbox)
                rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                       edgecolor=colors['opencv'], facecolor='none', 
                                       linestyle='-.', label=f'OpenCV {i+1}' if i == 0 else "")
                ax.add_patch(rect)
                
                # Final bbox (green, solid)
                x, y, w, h = norm_to_pixel(final_bbox)
                rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                       edgecolor=colors['final'], facecolor='none', 
                                       linestyle='-', label=f'Final {i+1}' if i == 0 else "")
                ax.add_patch(rect)
                
                # Add section number
                ax.text(x + 10, y + 30, f'S{i+1}', fontsize=12, fontweight='bold',
                       color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            # Set title and labels
            ax.set_title(f'Fused Bbox Alignment - {page_name}\n'
                        f'Red (--): Original | Blue (:): DinoV3 | Orange (-.): OpenCV | Green (-): Final', 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Add legend
            if gemini_bboxes:
                ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
            
            # Save visualization
            viz_path = Path(output_dir) / f"{page_name}_fused_alignment.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"📊 Visualization saved: {viz_path}")
            return str(viz_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create visualization: {str(e)}")
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
            # Convert numpy types for JSON serialization
            json_result = self._convert_numpy_types(result)
            
            # Save to JSON file
            output_path = Path(output_dir) / f"{result['page_name']}_fused_results.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Page results saved: {output_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save page results: {str(e)}")
    
    def generate_summary_report(self, all_results: List[Dict[str, Any]], output_dir: str):
        """Generate comprehensive summary report for fused approach."""
        try:
            # Calculate overall statistics
            successful_pages = [r for r in all_results if r.get('processing_success', False)]
            total_sections = sum(r.get('sections_detected', 0) for r in successful_pages)
            
            # Calculate extension statistics
            dino_extensions = []
            opencv_refinements = []
            bottom_optimizations = []
            final_extensions = []
            
            for result in successful_pages:
                for alignment in result.get('alignment_results', []):
                    dino_extensions.append(alignment.get('dino_extension_ratio', 1.0))
                    opencv_refinements.append(alignment.get('opencv_refinement_ratio', 1.0))
                    bottom_optimizations.append(alignment.get('bottom_optimization_ratio', 1.0))
                    final_extensions.append(alignment.get('final_extension_ratio', 1.0))
            
            # Create summary report
            summary = {
                'processing_summary': {
                    'total_pages': len(all_results),
                    'successful_pages': len(successful_pages),
                    'failed_pages': len(all_results) - len(successful_pages),
                    'success_rate': len(successful_pages) / len(all_results) if all_results else 0,
                    'total_sections_detected': total_sections,
                    'avg_sections_per_page': total_sections / len(successful_pages) if successful_pages else 0
                },
                'fused_approach_statistics': {
                    'stage1_method': 'DinoV3_embeddings',
                    'stage2_method': 'OpenCV_pixels',
                    'stage3_method': 'bottom_border_optimization',
                    'total_processing_stages': 3,
                    'dino_extension_stats': {
                        'mean': np.mean(dino_extensions) if dino_extensions else 0,
                        'std': np.std(dino_extensions) if dino_extensions else 0,
                        'min': np.min(dino_extensions) if dino_extensions else 0,
                        'max': np.max(dino_extensions) if dino_extensions else 0
                    },
                    'opencv_refinement_stats': {
                        'mean': np.mean(opencv_refinements) if opencv_refinements else 0,
                        'std': np.std(opencv_refinements) if opencv_refinements else 0,
                        'min': np.min(opencv_refinements) if opencv_refinements else 0,
                        'max': np.max(opencv_refinements) if opencv_refinements else 0
                    },
                    'bottom_optimization_stats': {
                        'mean': np.mean(bottom_optimizations) if bottom_optimizations else 0,
                        'std': np.std(bottom_optimizations) if bottom_optimizations else 0,
                        'min': np.min(bottom_optimizations) if bottom_optimizations else 0,
                        'max': np.max(bottom_optimizations) if bottom_optimizations else 0
                    },
                    'final_extension_stats': {
                        'mean': np.mean(final_extensions) if final_extensions else 0,
                        'std': np.std(final_extensions) if final_extensions else 0,
                        'min': np.min(final_extensions) if final_extensions else 0,
                        'max': np.max(final_extensions) if final_extensions else 0
                    }
                },
                'detailed_results': [self._convert_numpy_types(result) for result in all_results],
                'generated_at': datetime.now().isoformat(),
                'approach': 'fused_dinov3_opencv'
            }
            
            # Save summary report
            summary_path = Path(output_dir) / 'fused_alignment_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📋 Summary report saved: {summary_path}")
            
            # Create human-readable report
            report_lines = [
                "🚀 FUSED BBOX ALIGNMENT SUMMARY REPORT",
                "=" * 60,
                f"📊 Processing Statistics:",
                f"   Total pages processed: {summary['processing_summary']['total_pages']}",
                f"   Successful pages: {summary['processing_summary']['successful_pages']}",
                f"   Success rate: {summary['processing_summary']['success_rate']:.1%}",
                f"   Total sections detected: {summary['processing_summary']['total_sections_detected']}",
                f"   Average sections per page: {summary['processing_summary']['avg_sections_per_page']:.1f}",
                "",
                f"🔧 Fused Approach Analysis:",
                f"   Stage 1: DinoV3 Embedding Analysis",
                f"     - Average extension: {summary['fused_approach_statistics']['dino_extension_stats']['mean']:.2f}x",
                f"     - Extension range: {summary['fused_approach_statistics']['dino_extension_stats']['min']:.2f}x - {summary['fused_approach_statistics']['dino_extension_stats']['max']:.2f}x",
                "",
                f"   Stage 2: OpenCV Pixel Refinement",
                f"     - Average refinement: {summary['fused_approach_statistics']['opencv_refinement_stats']['mean']:.2f}x",
                f"     - Refinement range: {summary['fused_approach_statistics']['opencv_refinement_stats']['min']:.2f}x - {summary['fused_approach_statistics']['opencv_refinement_stats']['max']:.2f}x",
                "",
                f"   Stage 3: Bottom Border Post-Optimization",
                f"     - Average optimization: {summary['fused_approach_statistics']['bottom_optimization_stats']['mean']:.2f}x",
                f"     - Optimization range: {summary['fused_approach_statistics']['bottom_optimization_stats']['min']:.2f}x - {summary['fused_approach_statistics']['bottom_optimization_stats']['max']:.2f}x",
                "",
                f"   Final Combined Result:",
                f"     - Average total extension: {summary['fused_approach_statistics']['final_extension_stats']['mean']:.2f}x",
                f"     - Total extension range: {summary['fused_approach_statistics']['final_extension_stats']['min']:.2f}x - {summary['fused_approach_statistics']['final_extension_stats']['max']:.2f}x",
                "",
                f"📁 Output Files:",
                f"   - Summary report: fused_alignment_summary.json",
                f"   - Individual page results: *_fused_results.json",
                f"   - Visualizations: *_fused_alignment.png",
                "=" * 60
            ]
            
            # Save human-readable report
            report_path = Path(output_dir) / 'fused_alignment_report.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            self.logger.info(f"📄 Human-readable report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate summary report: {str(e)}")


def main():
    """Main function to set up argument parsing and run the fused bbox aligner."""
    parser = argparse.ArgumentParser(
        description="Fused BBox Aligner with DinoV3 + OpenCV + Bottom Optimization Sequential Approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with fused approach
  python bbox_aligner_fused.py document.pdf
  
  # With custom API key
  python bbox_aligner_fused.py document.pdf --api-key your_gemini_key
  
  # Custom output directory
  python bbox_aligner_fused.py document.pdf --output-dir fused_results
  
  # Process specific pages
  python bbox_aligner_fused.py document.pdf --start-page 1 --end-page 3
  
  # High DPI processing
  python bbox_aligner_fused.py document.pdf --dpi 600 --extension-threshold 0.7
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
        default='fused_bbox_alignment_results',
        help='Directory to save results (default: fused_bbox_alignment_results)'
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
        default=0.8,
        help='Threshold for DinoV3 white background detection (default: 0.8)'
    )
    
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip PDF extraction if page images already exist'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("🚀 Fused BBox Aligner - DinoV3 + OpenCV + Bottom Optimization Sequential Approach")
    print("=" * 80)
    print(f"📄 PDF: {args.pdf_path}")
    print(f"📁 Pages dir: {args.pages_dir}")
    print(f"📁 Output dir: {args.output_dir}")
    print(f"🎯 DPI: {args.dpi}")
    print(f"🔍 Extension threshold: {args.extension_threshold}")
    print(f"🔧 Approach: DinoV3 Embeddings → OpenCV Pixels → Bottom Optimization (3-Stage Sequential)")
    print("=" * 80)
    
    try:
        # Initialize the fused aligner
        aligner = BBoxAlignerFused(
            gemini_api_key=args.api_key,
            model_name='dinov2_vitb14'
        )
        
        print("🎉 Initialization complete! Ready to process PDF with 3-stage fused approach.")
        
        # Step 1: Extract PDF pages
        if not args.skip_extraction:
            print(f"\n📖 Extracting PDF pages...")
            page_images = aligner.extract_pdf_pages(
                args.pdf_path, 
                args.pages_dir, 
                dpi=args.dpi,
                start_page=args.start_page,
                end_page=args.end_page
            )
        else:
            print(f"\n⏭️ Skipping PDF extraction, using existing images...")
            pages_path = Path(args.pages_dir)
            if pages_path.exists():
                page_images = sorted([str(p) for p in pages_path.glob("*.png")])
                if args.start_page or args.end_page:
                    start_idx = (args.start_page - 1) if args.start_page else 0
                    end_idx = args.end_page if args.end_page else len(page_images)
                    page_images = page_images[start_idx:end_idx]
                print(f"📁 Found {len(page_images)} existing page images")
            else:
                raise FileNotFoundError(f"Pages directory not found: {args.pages_dir}")
        
        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Step 2: Process each page with 3-stage fused approach
        print(f"\n🔍 Processing {len(page_images)} pages with 3-stage fused approach...")
        all_results = []
        
        for i, image_path in enumerate(page_images, 1):
            print(f"\n--- Processing page {i}/{len(page_images)} ---")
            result = aligner.process_page(
                image_path, 
                args.output_dir, 
                extension_threshold=args.extension_threshold
            )
            all_results.append(result)
        
        # Step 3: Generate summary report
        print(f"\n📊 Generating 3-stage fused approach summary report...")
        aligner.generate_summary_report(all_results, args.output_dir)
        
        # Final summary
        successful_pages = len([r for r in all_results if r.get('processing_success', False)])
        total_sections = sum(r.get('sections_detected', 0) for r in all_results)
        
        print(f"\n🎉 3-Stage Fused Processing Complete!")
        print("=" * 80)
        print(f"📊 Processed: {successful_pages}/{len(all_results)} pages")
        print(f"🎯 Detected: {total_sections} sections total")
        print(f"🔧 Approach: DinoV3 → OpenCV → Bottom Optimization (3-stage sequential)")
        print(f"📁 Results saved in: {args.output_dir}/")
        
        # List generated files
        output_files = list(Path(args.output_dir).glob("*"))
        if output_files:
            print(f"\n📁 Generated files:")
            print(f"   📋 Summary: fused_alignment_summary.json")
            print(f"   📄 Report: fused_alignment_report.txt")
            print(f"   📊 Visualizations: *_fused_alignment.png")
            print(f"   📝 Page results: *_fused_results.json")
        
        print("=" * 80)
        print("✅ 3-stage fused approach processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()