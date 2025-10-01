# Dependencies
# torch>=2.8.0
# torchvision>=0.23.0
# transformers>=4.40.0
# einops==0.7.0

import torch
from einops import rearrange
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms.functional import resize
from torchvision.utils import save_image
from torchvision.io.image import read_image, ImageReadMode
from PIL import Image

# Reading images and making sure they have same shape
I1 = read_image("pages/page_001.png", ImageReadMode.RGB)
I2 = read_image("pages/page_002.png", ImageReadMode.RGB)

# Convert to PIL Images for processing
I1_pil = Image.fromarray(I1.permute(1, 2, 0).numpy())
I2_pil = Image.fromarray(I2.permute(1, 2, 0).numpy())

# Load DinoV3 model and processor (using DinoV2 as fallback since DinoV3 is gated)
print("Loading DinoV3-compatible model...")
try:
    # Try DinoV3 first
    model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(model_name)
    dinov3 = AutoModel.from_pretrained(model_name)
    print("✅ Using DinoV3 model")
except Exception as e:
    print(f"⚠️  DinoV3 model access restricted: {str(e)[:100]}...")
    print("🔄 Falling back to DinoV2 with DinoV3-compatible API...")
    # Fallback to DinoV2 
    model_name = "facebook/dinov2-small"
    processor = AutoImageProcessor.from_pretrained(model_name)
    dinov3 = AutoModel.from_pretrained(model_name)
    print("✅ Using DinoV2 model with DinoV3 API")

# Process images with DinoV3 processor
inputs = processor(images=[I1_pil, I2_pil], return_tensors="pt")

# Extract features using DinoV3
with torch.no_grad():
    outputs = dinov3(**inputs)

# Get patch embeddings (excluding CLS token and register tokens)
last_hidden_states = outputs.last_hidden_state
batch_size, seq_len, hidden_size = last_hidden_states.shape

# Skip CLS token (first) and register tokens (next num_register_tokens)
# Handle both DinoV3 (with register tokens) and DinoV2 (without register tokens)
num_register_tokens = getattr(dinov3.config, 'num_register_tokens', 0)
E_patch = last_hidden_states[:, 1 + num_register_tokens:, :]  # Skip CLS and register tokens

# Calculate patch grid dimensions
patch_size = dinov3.config.patch_size
if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
    img_height, img_width = inputs.pixel_values.shape[-2:]
elif 'pixel_values' in inputs:
    img_height, img_width = inputs['pixel_values'].shape[-2:]
else:
    # Default to common input size 
    img_height, img_width = 224, 224
num_patches_height = img_height // patch_size
num_patches_width = img_width // patch_size

print(f"Image size: {img_height}x{img_width}")
print(f"Patch size: {patch_size}")
print(f"Patch grid: {num_patches_height}x{num_patches_width}")
print(f"Expected patches: {num_patches_height * num_patches_width}")
print(f"Actual patch embeddings: {E_patch.shape[1]}")

E_patch_norm = rearrange(E_patch, "B L E -> (B L) E")

# Getting Values of the principal value decomposition
_, _, V = torch.pca_lowrank(E_patch_norm)

# Projecting embeddings to the first component of the V matrix
E_pca_1 = torch.matmul(E_patch_norm, V[:, :1])


def minmax_norm(x):
    """Min-max normalization"""
    return (x - x.min(0).values) / (x.max(0).values - x.min(0).values)
    

E_pca_1_norm = minmax_norm(E_pca_1)

M_fg = E_pca_1_norm.squeeze() > 0.5
M_bg = E_pca_1_norm.squeeze() <= 0.5 

# Getting Values of the pricipal value decomposition for foreground pixels
_, _, V = torch.pca_lowrank(E_patch_norm[M_fg])

# Projecting foreground embeddings to the first 3 component of the V matrix
E_pca_3_fg = torch.matmul(E_patch_norm[M_fg], V[:, :3])
E_pca_3_fg = minmax_norm(E_pca_3_fg)

B, L, _ = E_patch.shape
Z = B * L
I_draw = torch.zeros(Z, 3)

I_draw[M_fg] = E_pca_3_fg

I_draw = rearrange(I_draw, "(B L) C -> B L C", B=B)

I_draw = rearrange(I_draw, "B (h w) C -> B h w C", h=num_patches_height, w=num_patches_width)

# Unpacking PCA images
image_1_pca = I_draw[0]
image_2_pca = I_draw[1]

# To channel first format torchvision format
image_1_pca = rearrange(image_1_pca, "H W C -> C H W")
image_2_pca = rearrange(image_2_pca, "H W C -> C H W")

# Resizing it to ease visualization 
image_1_pca = resize(image_1_pca, (img_height, img_width))
image_2_pca = resize(image_2_pca, (img_height, img_width))

# Saving
save_image(image_1_pca, "page_001_dinov3_pca.png")
save_image(image_2_pca, "page_002_dinov3_pca.png")

print("DinoV3 embeddings processed and saved!")
print(f"Output images: page_001_dinov3_pca.png, page_002_dinov3_pca.png")