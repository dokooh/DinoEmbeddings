#!/usr/bin/env python3
"""
CUDA Compatibility Test for DINOv2

This script tests CUDA compatibility for DINOv2 models and provides
debugging information for CUDA-related issues.
"""

import torch
import os
import sys

def test_basic_cuda():
    """Test basic CUDA functionality"""
    print("🔍 Basic CUDA Tests")
    print("-" * 40)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            major, minor = torch.cuda.get_device_capability(i)
            print(f"  Compute capability: {major}.{minor}")
            
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("❌ CUDA not available")
        return False
    
    return True

def test_simple_cuda_operations():
    """Test simple CUDA tensor operations"""
    print("\n🧪 Simple CUDA Operations Test")
    print("-" * 40)
    
    try:
        # Create tensors on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Simple operations
        z = torch.matmul(x, y)
        result = z.sum().item()
        
        print(f"✅ Matrix multiplication successful: {result:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ CUDA operations failed: {str(e)}")
        return False

def test_dinov2_compatibility():
    """Test DINOv2 model compatibility"""
    print("\n🤖 DINOv2 CUDA Compatibility Test")
    print("-" * 40)
    
    try:
        print("Loading DINOv2 small model...")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model.eval()
        print("✅ Model loaded successfully")
        
        # Test CPU first
        print("\nTesting on CPU...")
        test_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            result_cpu = model.forward_features(test_input)
        print("✅ CPU inference successful")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            print("\nTesting on CUDA...")
            
            # Move model to CUDA
            model = model.cuda()
            test_input = test_input.cuda()
            
            with torch.no_grad():
                result_cuda = model.forward_features(test_input)
            print("✅ CUDA inference successful")
            
            # Compare results
            result_cpu_tokens = result_cpu['x_norm_patchtokens']
            result_cuda_tokens = result_cuda['x_norm_patchtokens'].cpu()
            
            diff = torch.abs(result_cpu_tokens - result_cuda_tokens).max().item()
            print(f"✅ CPU vs CUDA max difference: {diff:.6f}")
            
            return True
        else:
            print("⚠️  CUDA not available, skipping CUDA test")
            return True
            
    except RuntimeError as e:
        if "CUDA error" in str(e) or "kernel image" in str(e):
            print(f"❌ CUDA compatibility error: {str(e)}")
            print("\n🔧 Suggested solutions:")
            print("   1. Use CPU mode: python script.py --force_cpu")
            print("   2. Update PyTorch to match your CUDA version")
            print("   3. Check if your GPU is supported")
            return False
        else:
            print(f"❌ Other error: {str(e)}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def get_environment_info():
    """Get environment information"""
    print("\n🌍 Environment Information")
    print("-" * 40)
    
    # Python info
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # CUDA environment variables
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 
                 'CUDA_LAUNCH_BLOCKING', 'TORCH_USE_CUDA_DSA']
    
    print("\nCUDA Environment Variables:")
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")

def main():
    print("DINOv2 CUDA Compatibility Test")
    print("=" * 50)
    
    # Get environment info
    get_environment_info()
    
    # Test basic CUDA
    cuda_basic = test_basic_cuda()
    
    if cuda_basic:
        # Test simple operations
        cuda_ops = test_simple_cuda_operations()
        
        # Test DINOv2 compatibility
        dinov2_compat = test_dinov2_compatibility()
        
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Basic CUDA: {'✅ PASS' if cuda_basic else '❌ FAIL'}")
        print(f"CUDA Operations: {'✅ PASS' if cuda_ops else '❌ FAIL'}")
        print(f"DINOv2 CUDA: {'✅ PASS' if dinov2_compat else '❌ FAIL'}")
        
        if not dinov2_compat:
            print("\n🔧 Recommended solutions:")
            print("1. Run DINOv2 segmentation with --force_cpu flag")
            print("2. Update PyTorch: pip install torch torchvision --upgrade")
            print("3. Check CUDA version compatibility")
            
    else:
        print("\n💡 CUDA not available - use CPU mode")
        print("Run DINOv2 segmentation with --force_cpu flag")

if __name__ == "__main__":
    main()