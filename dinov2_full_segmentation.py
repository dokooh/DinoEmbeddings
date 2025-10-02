#!/usr/bin/env python3
"""
DINOv2 Full Segmentation Script

This script demonstrates foreground segmentation and object analysis using DINOv2.
It performs PCA-based segmentation to separate foreground objects from background
and generates RGB visualizations of semantic features.

Based on the Jupyter notebook fg_segmentation.ipynb

Dependencies:
- torch
- torchvision 
- matplotlib
- numpy
- opencv-python (cv2)
- scikit-learn
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as tt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import os
import argparse

def load_dinov2_model(model_size='base', force_cpu=False):
    """
    Load DINOv2 model from torch hub with CUDA compatibility handling
    
    Args:
        model_size: 'small', 'base', 'large', or 'giant'
        force_cpu: Force CPU usage even if CUDA is available
    
    Returns:
        DINOv2 model, device
    """
    model_names = {
        'small': 'dinov2_vits14',
        'base': 'dinov2_vitb14', 
        'large': 'dinov2_vitl14',
        'giant': 'dinov2_vitg14'
    }
    
    if model_size not in model_names:
        raise ValueError(f"Model size must be one of: {list(model_names.keys())}")
    
    print(f"Loading DINOv2 {model_size} model...")
    
    # Determine device with CUDA compatibility checks
    device = select_device(force_cpu)
    
    try:
        # Load model
        model = torch.hub.load('facebookresearch/dinov2', model_names[model_size])
        model.eval()
        
        # Try to move to device and test with a dummy tensor
        model = model.to(device)
        
        # Test CUDA compatibility with a small dummy tensor
        if device.type == 'cuda':
            test_tensor = torch.randn(1, 3, 224, 224).to(device)
            try:
                with torch.no_grad():
                    _ = model.forward_features(test_tensor)
                print(f"✅ Model loaded successfully on device: {device}")
            except RuntimeError as e:
                if "CUDA error" in str(e) or "kernel image" in str(e):
                    print(f"⚠️  CUDA compatibility issue detected: {str(e)[:100]}...")
                    print("🔄 Falling back to CPU...")
                    device = torch.device('cpu')
                    model = model.to(device)
                    print(f"✅ Model loaded on CPU device")
                else:
                    raise
        else:
            print(f"✅ Model loaded on device: {device}")
            
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        if device.type == 'cuda':
            print("🔄 Trying CPU fallback...")
            device = torch.device('cpu')
            model = torch.hub.load('facebookresearch/dinov2', model_names[model_size])
            model.eval()
            model = model.to(device)
            print(f"✅ Model loaded on CPU device")
        else:
            raise
    
    return model, device

def select_device(force_cpu=False):
    """
    Select the best available device with compatibility checks
    
    Args:
        force_cpu: Force CPU usage
        
    Returns:
        torch.device: Selected device
    """
    if force_cpu:
        return torch.device('cpu')
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    
    # Check CUDA device properties
    try:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"CUDA devices available: {device_count}")
        print(f"Current device: {current_device} ({device_name})")
        
        # Check compute capability
        major, minor = torch.cuda.get_device_capability(current_device)
        compute_capability = f"{major}.{minor}"
        print(f"Compute capability: {compute_capability}")
        
        # Warn about potential compatibility issues
        if major < 6:  # Very old GPUs
            print(f"⚠️  Warning: Old GPU detected (compute capability {compute_capability})")
            print("   Consider using --force_cpu if you encounter CUDA errors")
        
        return torch.device('cuda')
        
    except Exception as e:
        print(f"⚠️  CUDA device check failed: {str(e)}")
        print("Falling back to CPU")
        return torch.device('cpu')

def load_and_preprocess_images(image_paths, target_size=448):
    """
    Load and preprocess images for DINOv2
    
    Args:
        image_paths: List of image file paths
        target_size: Target image size (square)
    
    Returns:
        images: List of preprocessed numpy arrays
        input_tensor: PyTorch tensor ready for model
    """
    images = []
    
    print(f"Loading {len(image_paths)} images...")
    
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            print(f"Warning: Image not found: {path}")
            continue
            
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: Could not load image: {path}")
            continue
            
        image = cv2.resize(image, (target_size, target_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255
        images.append(image)
        print(f"  Loaded image {i+1}: {path}")
    
    if not images:
        raise ValueError("No valid images were loaded")
    
    # Convert to tensor: [batch, channel, height, width]
    images_arr = np.stack(images)
    input_tensor = torch.Tensor(np.transpose(images_arr, [0, 3, 2, 1]))
    
    # Apply normalization
    transform = tt.Compose([tt.Normalize(mean=0.5, std=0.2)])
    input_tensor = transform(input_tensor)
    
    return images, input_tensor

def extract_patch_features(model, input_tensor, device):
    """
    Extract patch features using DINOv2 with error handling
    
    Args:
        model: DINOv2 model
        input_tensor: Preprocessed image tensor
        device: PyTorch device
    
    Returns:
        patch_tokens: Extracted patch tokens
    """
    print("Extracting features with DINOv2...")
    
    try:
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            result = model.forward_features(input_tensor)
        
        # Get normalized patch tokens (excluding CLS token)
        patch_tokens = result['x_norm_patchtokens'].detach().cpu().numpy()
        
        # Reshape to [batch_size, num_patches, feature_dim]
        batch_size, num_patches, feature_dim = patch_tokens.shape
        patch_tokens = patch_tokens.reshape([batch_size, num_patches, -1])
        
        print(f"Extracted features shape: {patch_tokens.shape}")
        return patch_tokens
        
    except RuntimeError as e:
        if "CUDA error" in str(e) or "kernel image" in str(e):
            print(f"⚠️  CUDA error during feature extraction: {str(e)[:100]}...")
            print("🔄 Moving model to CPU and retrying...")
            
            # Move model to CPU
            model = model.cpu()
            device = torch.device('cpu')
            input_tensor = input_tensor.cpu()
            
            # Retry on CPU
            with torch.no_grad():
                result = model.forward_features(input_tensor)
            
            patch_tokens = result['x_norm_patchtokens'].detach().cpu().numpy()
            batch_size, num_patches, feature_dim = patch_tokens.shape
            patch_tokens = patch_tokens.reshape([batch_size, num_patches, -1])
            
            print(f"✅ Features extracted on CPU, shape: {patch_tokens.shape}")
            return patch_tokens
        else:
            raise

def perform_foreground_segmentation(patch_tokens, threshold=0.6):
    """
    Perform foreground segmentation using PCA
    
    Args:
        patch_tokens: Patch features from DINOv2
        threshold: Threshold for foreground/background separation
    
    Returns:
        masks: Binary masks for each image
        fg_pca: Fitted PCA model
        image_norm_patches: Normalized PCA features
    """
    print("Performing foreground segmentation...")
    
    batch_size, num_patches, feature_dim = patch_tokens.shape
    
    # Reshape all patches for PCA
    all_patches = patch_tokens.reshape([-1, feature_dim])
    
    # Fit PCA with 1 component for foreground/background separation
    fg_pca = PCA(n_components=1)
    reduced_patches = fg_pca.fit_transform(all_patches)
    
    # Scale features to (0,1)
    norm_patches = minmax_scale(reduced_patches)
    
    # Reshape back to original image structure
    image_norm_patches = norm_patches.reshape([batch_size, num_patches])
    
    # Create binary masks using threshold
    masks = []
    for i in range(batch_size):
        mask = (image_norm_patches[i, :] > threshold)
        masks.append(mask)
        print(f"  Image {i+1}: {np.sum(mask)}/{num_patches} patches marked as foreground")
    
    return masks, fg_pca, image_norm_patches

def analyze_foreground_objects(patch_tokens, masks):
    """
    Analyze foreground objects using PCA for RGB visualization
    
    Args:
        patch_tokens: Patch features from DINOv2
        masks: Binary foreground masks
    
    Returns:
        object_pca: Fitted PCA model for objects
        color_patches_list: RGB visualizations for each image
    """
    print("Analyzing foreground objects...")
    
    batch_size, num_patches, feature_dim = patch_tokens.shape
    
    # Extract foreground patches from all images
    fg_patches = []
    for i in range(batch_size):
        fg_patches.append(patch_tokens[i, masks[i], :])
    
    # Combine all foreground patches
    all_fg_patches = np.vstack(fg_patches)
    print(f"Total foreground patches: {all_fg_patches.shape[0]}")
    
    # Fit PCA with 3 components for RGB visualization
    object_pca = PCA(n_components=3)
    reduced_patches = object_pca.fit_transform(all_fg_patches)
    reduced_patches = minmax_scale(reduced_patches)
    
    print(f"PCA explained variance ratio: {object_pca.explained_variance_ratio_}")
    
    # Reshape features back to individual images
    color_patches_list = []
    start_idx = 0
    
    for i in range(batch_size):
        # Create full patch image (background = black)
        patch_image = np.zeros((num_patches, 3), dtype='float32')
        
        # Fill foreground patches with RGB colors
        num_fg_patches = np.sum(masks[i])
        if num_fg_patches > 0:
            patch_image[masks[i], :] = reduced_patches[start_idx:start_idx + num_fg_patches, :]
            start_idx += num_fg_patches
        
        # Reshape to spatial grid (assuming square patches)
        patches_per_side = int(np.sqrt(num_patches))
        color_patches = patch_image.reshape([patches_per_side, patches_per_side, 3])
        color_patches_list.append(color_patches)
    
    return object_pca, color_patches_list

def test_on_new_image(model, device, fg_pca, object_pca, test_image_path, target_size=672):
    """
    Test the trained PCA models on a new image with error handling
    
    Args:
        model: DINOv2 model
        device: PyTorch device
        fg_pca: Trained foreground PCA model
        object_pca: Trained object PCA model
        test_image_path: Path to test image
        target_size: Target size for test image
    
    Returns:
        test_image: Original test image
        result_visualization: RGB visualization result
    """
    print(f"Testing on new image: {test_image_path}")
    
    try:
        # Load and preprocess test image
        test_images, test_tensor = load_and_preprocess_images([test_image_path], target_size)
        test_image = test_images[0]
        
        # Extract features with error handling
        test_patch_tokens = extract_patch_features(model, test_tensor, device)
        test_patch_tokens = test_patch_tokens.reshape([-1, test_patch_tokens.shape[-1]])
        
        # Apply foreground segmentation
        fg_result = fg_pca.transform(test_patch_tokens)
        fg_result = minmax_scale(fg_result)
        fg_mask = (fg_result > 0.5).ravel()
        
        print(f"Test image: {np.sum(fg_mask)}/{len(fg_mask)} patches marked as foreground")
        
        # Apply object analysis
        object_result = object_pca.transform(test_patch_tokens)
        object_result = minmax_scale(object_result)
        
        # Create visualization (only foreground objects)
        only_object = np.zeros_like(object_result)
        only_object[fg_mask, :] = object_result[fg_mask, :]
        
        # Reshape to spatial grid
        patches_per_side = int(np.sqrt(len(test_patch_tokens)))
        result_visualization = only_object.reshape([patches_per_side, patches_per_side, 3])
        
        return test_image, result_visualization
        
    except Exception as e:
        print(f"⚠️  Error during test image processing: {str(e)}")
        raise

def visualize_results(images, masks, image_norm_patches, color_patches_list, save_path=None):
    """
    Visualize segmentation and object analysis results
    
    Args:
        images: Original input images
        masks: Foreground masks
        image_norm_patches: Foreground segmentation features
        color_patches_list: RGB object visualizations
        save_path: Optional path to save the visualization
    """
    batch_size = len(images)
    
    # Create visualization
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # Foreground segmentation overlay
        axes[i, 1].imshow(images[i])
        
        # Create overlay for foreground patches
        patches_per_side = int(np.sqrt(len(masks[i])))
        patch_overlay = image_norm_patches[i].copy()
        patch_overlay[~masks[i]] = 0
        patch_overlay = patch_overlay.reshape([patches_per_side, patches_per_side])
        
        # Resize overlay to match image
        h, w = images[i].shape[:2]
        patch_overlay_resized = cv2.resize(patch_overlay, (w, h))
        
        axes[i, 1].imshow(patch_overlay_resized, alpha=0.5, cmap='hot')
        axes[i, 1].set_title(f'Foreground Segmentation {i+1}')
        axes[i, 1].axis('off')
        
        # Object RGB visualization
        axes[i, 2].imshow(color_patches_list[i])
        axes[i, 2].set_title(f'Object Features {i+1}')
        axes[i, 2].axis('off')
        
        # Combined visualization
        axes[i, 3].imshow(images[i])
        overlay_resized = cv2.resize(color_patches_list[i], (images[i].shape[1], images[i].shape[0]))
        axes[i, 3].imshow(overlay_resized, alpha=0.7)
        axes[i, 3].set_title(f'Combined View {i+1}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def visualize_test_result(test_image, result_visualization, save_path=None):
    """
    Visualize test result
    
    Args:
        test_image: Original test image
        result_visualization: RGB visualization result
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(test_image)
    axes[0].set_title('Test Image')
    axes[0].axis('off')
    
    axes[1].imshow(result_visualization)
    axes[1].set_title('Object Segmentation Result')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Test result saved to: {save_path}")
    
    plt.show()

def main():
    """Main function to run the complete segmentation pipeline"""
    parser = argparse.ArgumentParser(description='DINOv2 Foreground Segmentation')
    parser.add_argument('--training_images', nargs='+', required=True,
                        help='Paths to training images')
    parser.add_argument('--test_image', type=str,
                        help='Path to test image (optional)')
    parser.add_argument('--model_size', choices=['small', 'base', 'large', 'giant'], 
                        default='base', help='DINOv2 model size')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Threshold for foreground segmentation')
    parser.add_argument('--target_size', type=int, default=448,
                        help='Target size for training images')
    parser.add_argument('--test_size', type=int, default=672,
                        help='Target size for test image')
    parser.add_argument('--save_dir', type=str, default='segmentation_results',
                        help='Directory to save results')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    parser.add_argument('--cuda_debug', action='store_true',
                        help='Enable CUDA debugging environment variables')
    
    args = parser.parse_args()
    
    # Set CUDA debugging environment variables if requested
    if args.cuda_debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_USE_CUDA_DSA'] = '1'
        print("🔧 CUDA debugging enabled")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # Load model with CUDA compatibility handling
        model, device = load_dinov2_model(args.model_size, force_cpu=args.force_cpu)
        
        # Load and preprocess training images
        images, input_tensor = load_and_preprocess_images(args.training_images, args.target_size)
        
        # Extract features with error handling
        patch_tokens = extract_patch_features(model, input_tensor, device)
        
        # Perform foreground segmentation
        masks, fg_pca, image_norm_patches = perform_foreground_segmentation(
            patch_tokens, args.threshold)
        
        # Analyze foreground objects
        object_pca, color_patches_list = analyze_foreground_objects(patch_tokens, masks)
        
        # Visualize training results
        training_save_path = os.path.join(args.save_dir, 'training_results.png')
        visualize_results(images, masks, image_norm_patches, color_patches_list, 
                         training_save_path)
        
        # Test on new image if provided
        if args.test_image:
            test_image, result_visualization = test_on_new_image(
                model, device, fg_pca, object_pca, args.test_image, args.test_size)
            
            test_save_path = os.path.join(args.save_dir, 'test_result.png') 
            visualize_test_result(test_image, result_visualization, test_save_path)
        
        print("\n✅ Segmentation pipeline completed successfully!")
        print(f"📁 Results saved in: {args.save_dir}")
        
        if device.type == 'cpu':
            print("\n💡 Tip: If you have a compatible CUDA GPU, you might get better performance")
            print("   Run again without --force_cpu to try GPU acceleration")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        
        # Provide helpful error messages
        if "CUDA" in str(e):
            print("\n🔧 CUDA Troubleshooting:")
            print("   1. Try running with --force_cpu flag")
            print("   2. Update PyTorch to match your CUDA version")
            print("   3. Check GPU compute capability compatibility")
            print("   4. Use --cuda_debug flag for more detailed error info")
        
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())