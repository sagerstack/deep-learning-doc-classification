# Phase 1: Notebook Foundation - Data & Features - Research

**Researched:** 2026-03-25
**Domain:** Jupyter ML experiment notebooks, PyTorch feature extraction, HuggingFace datasets, feature caching
**Confidence:** HIGH

## Summary

Phase 1 establishes the foundation of the project in a Jupyter notebook: loading RVL-CDIP efficiently, extracting ResNet-50 layer4 features preserving 7x7 spatial structure, and caching features to disk for training speed. The standard approach uses HuggingFace datasets with streaming for local development, torchvision's `create_feature_extractor()` for layer4 extraction (2048x7x7 feature maps), and torch.save() for feature caching with .pt format.

Key findings:
- HuggingFace datasets streaming mode enables sampling 100 images locally without downloading 38.8GB
- `torchvision.models.feature_extraction.create_feature_extractor()` is the modern approach (replaces manual forward hooks)
- Feature caching to .pt files is fastest for PyTorch workflows (no conversion overhead vs .npz)
- Notebook structure should follow clear sections: Setup → Data → Preprocessing → Feature Extraction → Caching
- Reference notebook (`reference/RVL-CDIP_ResNet50.ipynb`) shows working patterns: device-agnostic setup, grayscale→RGB conversion, HuggingFace integration

**Primary recommendation:** Use HuggingFace streaming with `.take(100)` for local sampling, `create_feature_extractor()` for layer4 extraction, and cache features to individual .pt files per image for flexibility.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **HuggingFace datasets** | 2.18+ | Load RVL-CDIP with streaming mode | Official RVL-CDIP dataset (`aharley/rvl_cdip`) on HuggingFace Hub. Streaming mode enables sampling without full download. Mini dataset alternative: `dvgodoy/rvl_cdip_mini` (4k images). Context7 benchmark score: 89.4 (high quality docs). |
| **PyTorch** | 2.9.1 | Tensor operations, model loading | Required for torchvision feature extraction. MPS support for Mac GPU acceleration. Already specified in project stack. |
| **torchvision** | 0.25+ | ResNet-50 model, feature extraction | Provides `create_feature_extractor()` for layer4 output extraction. Modern API: `weights=ResNet50_Weights.IMAGENET1K_V2`. |
| **Pillow** | 10.0+ | Image I/O | Required by HuggingFace datasets for image decoding to PIL format. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **tqdm** | 4.66+ | Progress bars for preprocessing loops | Show progress when extracting features for 100+ images. Standard for notebook workflows. |
| **numpy** | 1.26+ | Array operations, shape inspection | Converting tensors for inspection, computing statistics on feature maps. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Streaming | Download full dataset | Streaming: saves 38.8GB disk space, enables local dev with 100 images. Download: faster training on cluster (no streaming overhead). Use streaming for Mac, download for SUTD cluster. |
| create_feature_extractor | Manual forward hooks | create_feature_extractor: official, cleaner API, symbolically traces model. Hooks: more control, but manual cleanup required. Use create_feature_extractor (modern standard). |
| Mini dataset (dvgodoy/rvl_cdip_mini) | Streaming with .take(100) | Mini: fixed 4k subset, faster to load. Streaming: flexible sample size, same code path as full dataset. Use streaming (better for testing scale-up). |

**Installation:**
```bash
poetry add datasets pillow tqdm
# PyTorch, torchvision, numpy already in project stack
```

## Architecture Patterns

### Recommended Notebook Structure

```
notebook.ipynb
├── 1. Setup & Imports           # Libraries, device setup, random seeds
├── 2. Load RVL-CDIP             # HuggingFace datasets, display split sizes
├── 3. Preprocessing Pipeline    # Grayscale→RGB, resize, normalize, show samples
├── 4. ResNet-50 Feature Extract # Layer4 extraction, output 2048x7x7 features
└── 5. Feature Caching           # Save to disk, verify reload works
```

**Rationale:** Each section is independently runnable. Caching separates data engineering from model training (enables reuse).

### Pattern 1: HuggingFace Streaming with Sampling

**What:** Load dataset in streaming mode and take small sample for local development.

**When to use:** Local Mac development with limited disk space. Cannot download full 38.8GB RVL-CDIP.

**Example:**
```python
# Source: HuggingFace datasets Context7 docs
from datasets import load_dataset

# Streaming mode: doesn't download full dataset
ds = load_dataset("aharley/rvl_cdip", streaming=True)

# Sample 100 images from train split
train_sample = ds["train"].take(100)

# Convert to list for indexing (streaming is iterable, not indexable)
train_list = list(train_sample)

print(f"Sampled {len(train_list)} images")
# Output: Sampled 100 images
```

**Cluster workflow:** Use non-streaming mode for full dataset
```python
# On SUTD cluster with sufficient disk
ds = load_dataset("aharley/rvl_cdip", cache_dir="./.hf_cache")
# Downloads full dataset to cache_dir
```

### Pattern 2: ResNet-50 Layer4 Feature Extraction

**What:** Extract features from layer4 of ResNet-50 (before avgpool) to preserve 7x7 spatial structure.

**When to use:** Need spatial feature maps (2048, 7, 7) for graph construction. This is the core of Phase 1.

**Example:**
```python
# Source: torchvision official docs (https://docs.pytorch.org/vision/stable/feature_extraction.html)
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

# Load pretrained ResNet-50
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# Create feature extractor for layer4 output
return_nodes = {'layer4': 'layer4'}  # shortcut for last node in layer4
feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

# Extract features
with torch.no_grad():
    features = feature_extractor(input_tensor)
    layer4_output = features['layer4']  # Shape: [B, 2048, 7, 7]
```

**Alternative (manual approach without create_feature_extractor):**
```python
# From reference notebook pattern
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Remove final layers (avgpool, fc)
feature_model = nn.Sequential(*list(model.children())[:-2])
# Outputs layer4: [B, 2048, 7, 7]

features = feature_model(input_tensor)
```

### Pattern 3: Grayscale to RGB Conversion

**What:** Convert grayscale RVL-CDIP images (1 channel) to 3-channel RGB for ResNet-50 compatibility.

**When to use:** Always. ResNet-50 pretrained on ImageNet expects 3 channels.

**Example:**
```python
# Source: Reference notebook + PITFALLS.md
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Replicate channel 3x
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),  # ImageNet stats
])
```

**Why Grayscale(num_output_channels=3):**
- Replicates single grayscale channel to 3 channels
- Preserves ImageNet pretrained weights (first conv layer expects 3 channels)
- Alternative: modify first conv layer to 1 channel (loses pretrained weights)

### Pattern 4: Feature Caching to Disk

**What:** Save extracted features to disk as .pt files to avoid recomputation during training.

**When to use:** After extracting features for dataset. Enables fast iteration on GNN architecture without re-running CNN.

**Example:**
```python
# Source: PyTorch official docs (Context7)
import torch
from pathlib import Path

# Extract and save features
feature_dir = Path("./cached_features")
feature_dir.mkdir(exist_ok=True)

for idx, image in enumerate(dataset):
    # Extract features
    features = feature_extractor(image)  # [2048, 7, 7]

    # Save to disk
    torch.save(features, feature_dir / f"image_{idx:06d}.pt")

# Load features later
loaded_features = torch.load(feature_dir / "image_000001.pt")
```

**Why .pt format:**
- Native PyTorch format (no conversion overhead)
- Faster than numpy .npz for PyTorch workflows
- `torch.from_numpy()` uses same storage (zero-copy), but saving/loading .npz has overhead
- HDF5 is slower for individual file access (better for single large file)

### Pattern 5: Device-Agnostic Setup

**What:** Code that runs on Mac (MPS), Linux (CUDA), or CPU without modification.

**When to use:** Always. Project targets local Mac dev + SUTD GPU cluster.

**Example:**
```python
# Source: Reference notebook
import torch

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# Move model and data to device
model = model.to(device)
images = images.to(device)
```

### Anti-Patterns to Avoid

- **Loading full dataset without streaming:** Wastes 38.8GB disk space on Mac, unnecessary for local dev.
- **Using deprecated pretrained=True:** Old API. Use `weights=ResNet50_Weights.IMAGENET1K_V2` instead.
- **Extracting features after avgpool:** Loses spatial structure (7x7 → 1x1). Extract from layer4 before avgpool.
- **Caching all features in RAM:** 100 images x 100KB/image = 10MB (fine). 320k images = 32GB (not fine). Cache to disk.
- **Hardcoding device='cuda':** Breaks on Mac. Use device-agnostic pattern.
- **Forgetting .eval() mode:** ResNet has BatchNorm/Dropout. Use `model.eval()` for feature extraction.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Feature extraction from intermediate layers | Manual forward hooks with cleanup | `torchvision.models.feature_extraction.create_feature_extractor()` | Official API since torchvision 0.11. Symbolically traces model, handles cleanup automatically. Forward hooks require manual registration/removal and can leak memory. |
| Dataset sampling from large dataset | Download full dataset, random sample | HuggingFace datasets streaming with `.take(N)` | Streaming mode avoids 38.8GB download. `.take(N)` is efficient for sampling. Custom sampling requires downloading entire dataset first. |
| Grayscale to RGB conversion | Manual tensor replication | `transforms.Grayscale(num_output_channels=3)` | torchvision handles edge cases (already RGB, RGBA, etc.). Manual `tensor.repeat(3, 1, 1)` breaks on non-grayscale images. |
| Feature caching format | Custom HDF5 wrapper | `torch.save()` / `torch.load()` with .pt files | .pt is native PyTorch format, fastest for torch tensors. HDF5 adds dependency and complexity for individual file access. |

**Key insight:** torchvision and HuggingFace datasets have solved these problems. Don't reinvent wheels.

## Common Pitfalls

### Pitfall 1: Grayscale vs RGB Channel Mismatch

**What goes wrong:** Feeding 1-channel grayscale images to ResNet-50 causes shape error in first conv layer.

**Why it happens:** ResNet-50 first conv expects [N, 3, H, W]. RVL-CDIP images are [N, 1, H, W].

**How to avoid:** Use `transforms.Grayscale(num_output_channels=3)` to replicate channel 3x.

**Warning signs:**
- `RuntimeError: Given groups=1, weight of size [64, 3, 7, 7], expected input[N, 1, H, W]`
- First conv layer shape mismatch

**Source:** PITFALLS.md (1.1)

### Pitfall 2: Wrong Feature Extraction Layer

**What goes wrong:** Extracting from wrong layer loses spatial structure or semantic information.

**Why it happens:**
- **Too early** (layer1, layer2): Low-level features (edges, textures), miss semantic structure
- **Too late** (after avgpool, fc): Already pooled to [N, 2048], spatial structure lost
- **Just right** (layer4 before avgpool): Semantic features with 7x7 spatial structure

**How to avoid:** Extract from layer4 output (before avgpool). For this project, layer4 is specified in requirements (MDL-01).

**Warning signs:**
- Feature shape is [N, 2048] instead of [N, 2048, 7, 7] → extracted too late
- Graph construction fails due to no spatial structure

**Source:** PITFALLS.md (1.2)

### Pitfall 3: Memory Explosion with Feature Maps

**What goes wrong:** Storing 320k feature maps (2048x7x7 floats each) requires ~128GB RAM uncompressed.

**Why it happens:**
- ResNet layer4 output: 2048 × 7 × 7 = 100,352 floats per image
- 320k images × 100,352 × 4 bytes = ~128GB

**How to avoid for Phase 1 (100 images):**
- Cache to disk as individual .pt files (100 images × 100KB = 10MB, manageable)
- For full dataset on cluster: use disk-based caching with memory mapping

**Warning signs:**
- Out-of-memory errors during feature extraction
- System swap thrashing (slow performance)

**Source:** PITFALLS.md (1.4)

### Pitfall 4: Notebook Organization Chaos

**What goes wrong:** Cells executed out of order, unclear dependencies, unreproducible results.

**Why it happens:** Jupyter notebooks allow non-linear execution. State can become inconsistent.

**How to avoid:**
1. **Structure notebook in clear sections** (Setup → Data → Preprocessing → Features → Caching)
2. **Restart kernel and run all cells** before saving to verify order
3. **Use markdown headers** to separate sections
4. **Set random seeds at top** (`torch.manual_seed(42)`)

**Warning signs:**
- Cells fail when executed in order (indicates out-of-order development)
- Teammate can't reproduce results from notebook

**Source:** WebSearch - [Structuring Jupyter Notebooks for ML Experiments](https://towardsdatascience.com/structuring-jupyter-notebooks-for-fast-and-iterative-machine-learning-experiments-e09b56fa26bb/)

### Pitfall 5: Deprecated torchvision API

**What goes wrong:** Using `pretrained=True` triggers deprecation warnings.

**Why it happens:** torchvision updated API in recent versions. Old tutorials use deprecated syntax.

**How to avoid:**
```python
# OLD (deprecated):
model = resnet50(pretrained=True)

# NEW (correct):
from torchvision.models import ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
```

**Warning signs:**
- Deprecation warnings in notebook output
- Reference to "pretrained=True" in code

**Source:** STACK.md + torchvision docs

## Code Examples

Verified patterns from official sources:

### Load RVL-CDIP with Streaming (Small Sample)

```python
# Source: HuggingFace datasets Context7 docs
from datasets import load_dataset

# Streaming mode for local dev (doesn't download full 38.8GB)
ds = load_dataset("aharley/rvl_cdip", streaming=True, split="train")

# Take 100 samples
sample = list(ds.take(100))

print(f"Loaded {len(sample)} images")
# Output: Loaded 100 images
```

### Load RVL-CDIP Full Dataset (Cluster)

```python
# Source: Reference notebook
from datasets import load_dataset

# Non-streaming mode (downloads full dataset to cache)
ds = load_dataset("aharley/rvl_cdip", cache_dir="./.hf_cache")

print(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")
# Output: Train: 320000, Val: 40000, Test: 40000
```

### Extract Layer4 Features with create_feature_extractor

```python
# Source: torchvision docs (https://docs.pytorch.org/vision/stable/feature_extraction.html)
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import torch

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# Create feature extractor
return_nodes = {'layer4': 'layer4'}
feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
feature_extractor = feature_extractor.to(device)

# Extract features
image = torch.randn(1, 3, 224, 224).to(device)  # Example input
with torch.no_grad():
    features = feature_extractor(image)
    layer4_features = features['layer4']  # Shape: [1, 2048, 7, 7]

print(f"Feature shape: {layer4_features.shape}")
# Output: Feature shape: torch.Size([1, 2048, 7, 7])
```

### Preprocessing Pipeline (Grayscale to RGB)

```python
# Source: Reference notebook
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Replicate to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                        std=[0.229, 0.224, 0.225]),   # ImageNet std
])

# Apply to PIL image
processed = transform(pil_image)  # Output: [3, 224, 224]
```

### Cache Features to Disk

```python
# Source: PyTorch Context7 docs
import torch
from pathlib import Path

# Setup cache directory
cache_dir = Path("./cached_features/train")
cache_dir.mkdir(parents=True, exist_ok=True)

# Save features
for idx, image in enumerate(dataset):
    features = extract_features(image)  # [2048, 7, 7]
    torch.save(features, cache_dir / f"{idx:06d}.pt")

# Load features
loaded = torch.load(cache_dir / "000001.pt")
print(f"Loaded feature shape: {loaded.shape}")
# Output: Loaded feature shape: torch.Size([2048, 7, 7])
```

### Display Sample Images (Notebook Visualization)

```python
# Source: Standard ML notebook pattern
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for idx, ax in enumerate(axes.flat):
    image = sample[idx]['image']  # PIL image
    label = sample[idx]['label']
    ax.imshow(image, cmap='gray')
    ax.set_title(f"Class {label}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual forward hooks | `create_feature_extractor()` | torchvision 0.11 (2021) | Cleaner API, automatic cleanup, symbolic tracing. Replaces error-prone hook registration. |
| `pretrained=True` | `weights=ResNet50_Weights.IMAGENET1K_V2` | torchvision 0.13 (2022) | Explicit weight specification. Supports multiple pretrained versions (V1, V2). |
| Download full dataset | HuggingFace streaming | datasets 2.0+ (2022) | Enables sampling without full download. Critical for local dev with large datasets. |
| .npz caching | .pt caching for PyTorch | Current standard | .pt is native format, no conversion overhead. .npz requires numpy↔torch conversion. |

**Deprecated/outdated:**
- **`pretrained=True`:** Deprecated in torchvision 0.13. Use `weights=` parameter.
- **`IterableDataset.map()` with transformations in DataLoader:** 1000x slower than pre-mapping with `dataset.map()`. Pre-process before training loop.

## Open Questions

1. **Local mini dataset vs streaming?**
   - What we know: Mini dataset (`dvgodoy/rvl_cdip_mini`) is 4k images, fixed. Streaming with `.take(100)` is flexible.
   - What's unclear: Which is faster for repeated experiments?
   - Recommendation: Start with streaming (same code path as full dataset). Switch to mini if streaming overhead is noticeable.

2. **Feature caching: individual .pt vs single HDF5?**
   - What we know: Individual .pt files are simpler. HDF5 is faster for sequential access of large files.
   - What's unclear: For 100 images, overhead is negligible. For 320k images on cluster, HDF5 may be better.
   - Recommendation: Phase 1 uses individual .pt (simpler). Revisit for full-scale training in Phase 2.

3. **Should features be cached on CPU or GPU?**
   - What we know: Extracted on GPU, can save as CPU tensors (`.cpu()`) or GPU tensors.
   - What's unclear: Saving GPU tensors requires same device at load time (not portable).
   - Recommendation: Save as CPU tensors (`features.cpu()`) for portability. Load to device during training.

## Sources

### Primary (HIGH confidence)
- [PyTorch torchvision feature extraction](https://docs.pytorch.org/vision/stable/feature_extraction.html) - Official docs for create_feature_extractor
- [HuggingFace datasets Context7](https://huggingface.co/docs/datasets/main/stream) - Streaming mode, batching, iteration patterns
- [PyTorch torch.save/torch.load Context7](https://docs.pytorch.org/docs/stable/notes/serialization.html) - Tensor persistence patterns
- [Reference notebook](reference/RVL-CDIP_ResNet50.ipynb) - Working baseline: device setup, data loading, preprocessing
- [STACK.md](.planning/research/STACK.md) - Project technology stack decisions
- [PITFALLS.md](.planning/research/PITFALLS.md) - Phase-specific common mistakes

### Secondary (MEDIUM confidence)
- [Structuring Jupyter Notebooks for ML Experiments](https://towardsdatascience.com/structuring-jupyter-notebooks-for-fast-and-iterative-machine-learning-experiments-e09b56fa26bb/) - Notebook organization patterns
- [Guide to File Formats for Machine Learning](https://www.hopsworks.ai/post/guide-to-file-formats-for-machine-learning) - .npz vs .pt vs HDF5 comparison
- [PyTorch feature extraction with forward hooks](https://medium.com/the-owl/using-forward-hooks-to-extract-intermediate-layer-outputs-from-a-pre-trained-model-in-pytorch-1ec17af78712) - Alternative to create_feature_extractor (older approach)

### Tertiary (LOW confidence)
- None for this phase (all findings verified with official docs or Context7)

## Confidence Summary

| Area | Confidence | Reason |
|------|------------|--------|
| HuggingFace datasets streaming | HIGH | Verified via Context7 official docs, code examples tested |
| ResNet-50 feature extraction | HIGH | Official torchvision docs, create_feature_extractor API confirmed |
| Feature caching (.pt format) | HIGH | PyTorch Context7 docs, torch.save/load patterns |
| Notebook structure | MEDIUM | Based on WebSearch article + reference notebook, not official standard |
| Grayscale to RGB conversion | HIGH | Reference notebook + PITFALLS.md + torchvision transforms docs |
| Device-agnostic setup | HIGH | Reference notebook pattern, PyTorch MPS docs |

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified via Context7 or official docs
- Architecture: HIGH - torchvision patterns from official docs, reference notebook confirms viability
- Pitfalls: HIGH - From project PITFALLS.md research + official docs

**Research date:** 2026-03-25
**Valid until:** 60 days (stack is stable, notebook patterns are evergreen)
