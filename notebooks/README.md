# Notebook Cell Snippets - Phase 1: Data & Features

This file contains ready-to-paste code cells for your Jupyter notebook. Each section represents a notebook section with markdown cells (shown as quoted text) and code cells (shown as code blocks).

Copy and paste these cells in order into your notebook.

---

## Section 1: Setup and Imports

**Markdown cell:**
```markdown
# Phase 1: RVL-CDIP Data Loading and Feature Extraction

This notebook demonstrates the complete pipeline for loading RVL-CDIP document images and extracting spatial features using ResNet-50 layer4 output.

## 1. Setup and Imports
```

**Code cell 1: Import libraries**
```python
import os
import torch
import torchvision
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path

print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
```

**Code cell 2: Import from src package**
```python
from src.config import Config
from src.data import load_rvl_cdip, get_transform, display_samples, RVL_CDIP_LABELS
from src.features import create_resnet_extractor, extract_features_batch, cache_features, load_cached_features
```

**Code cell 3: Initialize config**
```python
# Create config and set random seeds
config = Config()
config.seed_everything()

print(f"\nConfiguration:")
print(f"  Device: {config.device}")
print(f"  Mode: {config.mode}")
print(f"  Sample size: {config.sample_size}")
print(f"  Batch size: {config.batch_size}")
print(f"  Cache dir: {config.cache_dir}")
```

**Code cell 4: Set MPS fallback for compatibility**
```python
# Enable MPS fallback for operations not yet supported on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

---

## Section 2: Load RVL-CDIP Dataset

**Markdown cell:**
```markdown
## 2. Load RVL-CDIP Dataset

Load the dataset based on config mode:
- **Sample mode**: Small-200 dataset (3,200 images, 200 per class)
- **Full mode**: Complete RVL-CDIP (400k images)
```

**Code cell 5: Load dataset**
```python
# Load dataset (automatically uses config.mode)
data = load_rvl_cdip(config)

print(f"\nDataset splits:")
for split_name, split_data in data.items():
    print(f"  {split_name}: {len(split_data)} samples")

print(f"\nClass labels ({len(RVL_CDIP_LABELS)} classes):")
for i, label in enumerate(RVL_CDIP_LABELS):
    print(f"  {i}: {label}")
```

**Code cell 6: Display class distribution**
```python
# Count class distribution for train split
from collections import Counter

train_labels = [sample["label"] for sample in data["train"]]
label_counts = Counter(train_labels)

print("\nTrain split class distribution:")
for label_idx, count in sorted(label_counts.items()):
    label_name = RVL_CDIP_LABELS[label_idx]
    print(f"  {label_name:25s}: {count:4d} samples")
```

**Code cell 7: Display sample images**
```python
# Display first 10 samples from train split
fig = display_samples(data["train"], n=10)
plt.show()
```

---

## Section 3: Preprocessing Pipeline

**Markdown cell:**
```markdown
## 3. Preprocessing Pipeline

ResNet-50 requires 224x224 RGB images normalized with ImageNet statistics.
Our pipeline:
1. Resize to 224x224
2. Convert grayscale to 3 channels
3. Normalize with ImageNet mean/std
```

**Code cell 8: Get transform pipeline**
```python
# Get preprocessing transform
transform = get_transform()

print("Preprocessing pipeline:")
print(transform)
```

**Code cell 9: Apply transform to single sample**
```python
# Apply transform to a single sample
sample = data["train"][0]
original_image = sample["image"]
transformed_image = transform(original_image)

print(f"Original image size: {original_image.size}")
print(f"Transformed tensor shape: {transformed_image.shape}")
print(f"Transformed tensor range: [{transformed_image.min():.3f}, {transformed_image.max():.3f}]")
```

**Code cell 10: Visualize preprocessed images**
```python
# Visualize original vs preprocessed (denormalized for display)
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor for visualization."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

# Show 5 samples: original and preprocessed side-by-side
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(5):
    sample = data["train"][i]
    original = sample["image"]
    preprocessed = transform(sample["image"])

    # Original (top row)
    axes[0, i].imshow(original, cmap="gray")
    axes[0, i].set_title(f"Original\n{RVL_CDIP_LABELS[sample['label']]}")
    axes[0, i].axis("off")

    # Preprocessed (bottom row) - denormalize and show first channel
    preprocessed_denorm = denormalize(preprocessed)
    axes[1, i].imshow(preprocessed_denorm[0], cmap="gray")
    axes[1, i].set_title("Preprocessed")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()
```

---

## Section 4: ResNet-50 Feature Extraction

**Markdown cell:**
```markdown
## 4. ResNet-50 Feature Extraction

Extract spatial features from ResNet-50 layer4 output (before avgpool).
Output shape: [batch_size, 2048, 7, 7]
- 2048 feature channels
- 7×7 spatial resolution preserves document layout information
```

**Code cell 11: Create feature extractor**
```python
# Create ResNet-50 feature extractor
extractor = create_resnet_extractor(config)
```

**Code cell 12: Extract features for small batch**
```python
# Extract features for 5 sample images
batch_size = 5
batch_samples = data["train"][:batch_size]

# Preprocess images
batch_images = torch.stack([transform(sample["image"]) for sample in batch_samples])
print(f"Batch images shape: {batch_images.shape}")

# Extract features
batch_features = extract_features_batch(extractor, batch_images, config.device)
print(f"Batch features shape: {batch_features.shape}")
print(f"Features per image: {batch_features.shape[1]} channels × {batch_features.shape[2]}×{batch_features.shape[3]} spatial")
```

**Code cell 13: Visualize feature maps**
```python
# Visualize mean activation across channels for one image
sample_idx = 0
features_single = batch_features[sample_idx]  # [2048, 7, 7]

# Compute mean activation across channels
mean_activation = features_single.mean(dim=0)  # [7, 7]

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Original image
axes[0].imshow(batch_samples[sample_idx]["image"], cmap="gray")
axes[0].set_title(f"Original: {RVL_CDIP_LABELS[batch_samples[sample_idx]['label']]}")
axes[0].axis("off")

# Mean feature activation heatmap
im = axes[1].imshow(mean_activation.numpy(), cmap="viridis")
axes[1].set_title("Mean Feature Activation (7×7)")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046)

plt.tight_layout()
plt.show()

print(f"Activation range: [{mean_activation.min():.2f}, {mean_activation.max():.2f}]")
```

---

## Section 5: Feature Caching

**Markdown cell:**
```markdown
## 5. Feature Caching

Cache extracted features to disk for faster experimentation.
- Each image gets one .pt file: {features: [2048, 7, 7], label: int}
- Cached features are device-agnostic (saved as CPU tensors)
```

**Code cell 14: Cache features for all splits**
```python
# Cache features for all available splits
for split_name in data.keys():
    print(f"\nCaching {split_name} split...")
    cache_dir = cache_features(
        data[split_name],
        extractor,
        transform,
        config,
        split_name
    )
    print(f"Cached to: {cache_dir}")
```

**Code cell 15: Load cached features**
```python
# Load cached features for verification
cached_train = load_cached_features(config, "train")
cached_val = load_cached_features(config, "validation")

print(f"\nCached features loaded:")
print(f"  Train: {len(cached_train)} samples")
print(f"  Validation: {len(cached_val)} samples")

# Check first cached sample
first_cached = cached_train[0]
print(f"\nFirst cached sample:")
print(f"  Features shape: {first_cached['features'].shape}")
print(f"  Label: {first_cached['label']} ({RVL_CDIP_LABELS[first_cached['label']]})")
```

**Code cell 16: Verify round-trip integrity**
```python
# Verify cached features match freshly extracted features
print("\nVerifying round-trip integrity...")

# Extract fresh features for first 5 samples
test_samples = data["train"][:5]
test_images = torch.stack([transform(s["image"]) for s in test_samples])
fresh_features = extract_features_batch(extractor, test_images, config.device)

# Compare with cached features
cached_features = torch.stack([cached_train[i]["features"] for i in range(5)])

# Check if they match (within floating point tolerance)
max_diff = (fresh_features - cached_features).abs().max()
print(f"Maximum difference: {max_diff:.2e}")

if max_diff < 1e-5:
    print("✓ Round-trip verification PASSED - cached features match fresh extraction")
else:
    print("✗ Round-trip verification FAILED - features differ")

# Show cache directory structure
cache_root = config.cache_dir
print(f"\nCache directory structure:")
for split_dir in sorted(cache_root.iterdir()):
    if split_dir.is_dir():
        num_files = len(list(split_dir.glob("*.pt")))
        print(f"  {split_dir.name}/: {num_files} .pt files")
```

---

## Next Steps

After running all cells above, you have:
1. Loaded RVL-CDIP dataset
2. Preprocessed images for ResNet-50
3. Extracted spatial features [2048, 7, 7] from layer4
4. Cached all features to disk

**Next phase**: Build graph structure from spatial features and implement GraphSAGE model.
