# Technology Stack

**Project:** GNN-based Document Image Classification
**Researched:** 2026-03-25
**Overall Confidence:** HIGH

## Executive Summary

The standard 2025/2026 stack for GraphSAGE-based document image classification combines PyTorch 2.9.1 with PyTorch Geometric 2.7, using torchvision for feature extraction, HuggingFace datasets for data loading, and scikit-learn/scipy for graph construction. Poetry is the recommended package manager. This stack is production-ready, well-documented, and fully compatible with Mac MPS (Metal Performance Shaders) for local GPU acceleration.

## Recommended Stack

### Core Deep Learning Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **PyTorch** | 2.9.1 | Neural network training, tensor operations | Latest stable release with MPS support. Version 2.9.1 confirmed compatible with PyTorch Geometric 2.7. Supports Mac MPS for GPU acceleration on Apple Silicon (M1/M2/M3/M4). |
| **PyTorch Geometric** | 2.7.0 | Graph neural network operations, GraphSAGE implementation | Latest stable release. Fully compatible with PyTorch 2.9. Simplified installation (no external libraries required beyond PyTorch). 567 code snippets in Context7 (high documentation quality). Provides pre-built GraphSAGE, global pooling operations, and efficient batching for graph classification. |
| **torchvision** | 0.25+ | ResNet-50 feature extraction, image preprocessing | Maintains compatibility with PyTorch 2.9. Provides ResNet50 with ImageNet pretrained weights (IMAGENET1K_V2 recommended for better performance than V1). Modern API uses `weights` parameter instead of deprecated `pretrained` boolean. |

**Confidence:** HIGH - Verified through Context7 official documentation and PyTorch Geometric installation matrix.

**MPS Compatibility Notes:**
- PyTorch 2.9.1 requires macOS 12.3+ for MPS
- KNOWN ISSUE: macOS 26.0 (Tahoe) has reported MPS availability issues with PyTorch 2.9.1 (MPS built but not available)
- Workaround: Use macOS 13 (Ventura) through macOS 14 (Sonoma) for stable MPS support
- Use native ARM64 Python (not Rosetta x86 emulation) - common mistake causing MPS unavailability
- MPS does not support multi-GPU training (Apple Silicon exposes single logical GPU)

### Data Loading & Preprocessing

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **HuggingFace datasets** | 2.18+ | Loading RVL-CDIP dataset | Official RVL-CDIP dataset available as `aharley/rvl_cdip` on HuggingFace Hub (400k images, 16 classes, proper train/val/test splits). Automatic image decoding to PIL. High quality documentation (89.4 benchmark score on Context7). Supports streaming for large datasets. Alternative mini versions available for prototyping (`dvgodoy/rvl_cdip_mini`: 4k images). |
| **Pillow** | 10.0+ | Image I/O, preprocessing | Required by HuggingFace datasets for image decoding. Standard for Python image operations. |

**Confidence:** HIGH - Verified via Context7 and HuggingFace Hub search.

### Graph Construction

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **scikit-learn** | 1.8.0 | k-NN graph construction, evaluation metrics | Latest version supports Python 3.11-3.14 with free-threaded CPython. Provides `kneighbors_graph()` for building k-NN graphs from ResNet features. Supports 'connectivity' (adjacency matrix) and 'distance' (weighted edges) modes. Metrics module for classification evaluation (precision, recall, F1, confusion matrix). |
| **scipy** | 1.14+ | Sparse matrix operations, distance metrics | Required for sparse adjacency matrix handling. Integrated with scikit-learn's k-NN (uses `scipy.spatial.distance` metrics). Efficient for large graphs (400k nodes in full RVL-CDIP). |
| **numpy** | 1.26+ | Numerical operations, tensor conversions | Foundation for scikit-learn and scipy. Required for converting between PyTorch tensors and numpy arrays during graph construction. |

**Confidence:** HIGH - Verified through scikit-learn official docs and Context7.

**Rationale for k-NN Graph Construction:**
- Scikit-learn's `kneighbors_graph()` is the standard approach for spatial graph construction
- Supports ball-tree algorithm for efficient high-dimensional nearest neighbor search
- Scipy sparse matrices integrate seamlessly with PyTorch Geometric's edge_index format
- Alternative: PyGSP (falls back to scikit-learn when FLANN unavailable), but adds unnecessary dependency

### Training Utilities

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **tqdm** | 4.66+ | Progress bars for training loops | Standard for PyTorch training loops. Shows real-time metrics (loss, accuracy). Simple integration with `.set_description()` and `.set_postfix()`. Default in PyTorch Lightning (TQDMProgressBar). |
| **Weights & Biases (wandb)** | 0.18+ | Experiment tracking, hyperparameter logging | Superior to TensorBoard for reproducibility and team collaboration. Tracks metrics, hyperparameters, code versions, model checkpoints. 5-line integration with PyTorch. Cloud-hosted (no local management headaches). Better visualization than TensorBoard for multi-run comparisons. |
| **TensorBoard** (alternative) | 2.16+ | Basic experiment tracking | Use if: (1) no cloud storage allowed, (2) single-person project, (3) minimal tracking needs. Ships with PyTorch, zero additional setup. Good for getting started, but lacks reproducibility features (no data/code versioning). |

**Confidence:** MEDIUM-HIGH
- tqdm: HIGH (standard, well-documented)
- wandb vs TensorBoard: MEDIUM (based on WebSearch comparison articles, not official docs)

**Recommendation:** Start with wandb unless project constraints require local-only tracking (then use TensorBoard).

### Visualization

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **matplotlib** | 3.9+ | Basic plotting (loss curves, confusion matrices) | Foundation for scientific plotting in Python. Integrates with PyTorch tensors (convert to numpy first). |
| **seaborn** | 0.13+ | Statistical visualizations, prettier plots | Built on matplotlib with high-level interface. Simplifies heatmaps (confusion matrices), distribution plots. Convert PyTorch tensors to numpy before plotting with seaborn. |

**Confidence:** HIGH - Standard visualization stack for PyTorch projects.

### Package Management

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **Poetry** | 1.8+ | Dependency management, virtual environment management | Specified in project requirements. Deterministic builds via `poetry.lock`. Integrates with PEP 517/518 standards via `pyproject.toml`. All commands via `poetry run` (e.g., `poetry run python train.py`). Better dependency resolution than pip. Simpler than conda for Python-only projects. |

**Confidence:** HIGH - Specified in project context and verified through WebSearch.

**Note on 2026 Landscape:**
- `uv` is gaining traction as a faster alternative to Poetry
- However, Poetry still has the smoothest workflow for publishing to PyPI
- For this project (no PyPI publishing planned), both are viable
- Stick with Poetry as specified in project requirements

## PyTorch Geometric Optional Dependencies

PyTorch Geometric offers optional acceleration libraries. Install with:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.9.0+cpu.html
```

**For Mac MPS (Apple Silicon):**
```bash
# Use cpu wheel (MPS uses same binaries as CPU)
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.9.0+cpu.html
```

**Purpose of each library:**
- `pyg_lib`: Core PyG acceleration library
- `torch_scatter`: Efficient scatter/gather operations for GNN message passing
- `torch_sparse`: Sparse matrix operations (faster graph convolutions)
- `torch_cluster`: Graph pooling and sampling operations

**Confidence:** HIGH - From official PyTorch Geometric installation docs via Context7.

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Deep Learning Framework | PyTorch 2.9.1 | TensorFlow 2.x | PyTorch Geometric is PyTorch-native. TensorFlow has weaker GNN library support (TensorFlow-GNN less mature than PyG). |
| GNN Library | PyTorch Geometric 2.7 | DGL (Deep Graph Library) | PyG has better documentation (567 snippets), simpler installation (no external deps since 2.3), more active development. DGL comparable but PyG is standard for academic research. |
| Dataset Loading | HuggingFace datasets | Manual download + torchvision.datasets.ImageFolder | HuggingFace provides official RVL-CDIP with proper splits. Auto-caching, streaming support. Manual approach requires custom split logic. |
| Graph Construction | scikit-learn k-NN | PyTorch k-NN (custom) | Scikit-learn is battle-tested, optimized (ball-tree), well-documented. Custom implementation = reinventing the wheel. |
| Feature Extraction | torchvision ResNet-50 | timm (PyTorch Image Models) | torchvision sufficient for ResNet-50. timm adds 200+ models but unnecessary complexity for this project. Prefer simplicity. |
| Experiment Tracking | wandb | MLflow | wandb simpler setup, better UX for PyTorch. MLflow more enterprise-focused (model registry, serving), overkill for research project. |
| Experiment Tracking | wandb | TensorBoard | See "Training Utilities" section. TensorBoard for basic needs, wandb for reproducibility + collaboration. |
| Package Manager | Poetry | pip + venv | Poetry provides deterministic builds (poetry.lock), better dependency resolution, cleaner pyproject.toml. pip requires manual requirements.txt maintenance. |
| Package Manager | Poetry | conda | Conda better for mixed Python/non-Python stacks (e.g., CUDA, system libs). Poetry simpler for pure Python. PyTorch pip wheels work fine. |
| Package Manager | Poetry | uv | uv faster but newer (less proven). Poetry has better PyPI publishing workflow. Stick with Poetry as specified in requirements. |

## Installation (Poetry)

### 1. Initialize Project
```bash
poetry init
# OR if pyproject.toml already exists:
poetry install
```

### 2. Add Core Dependencies
```bash
# Deep learning
poetry add torch==2.9.1
poetry add torch-geometric==2.7.0
poetry add torchvision==0.25.0

# Data loading
poetry add datasets pillow

# Graph construction
poetry add scikit-learn==1.8.0 scipy numpy

# Training utilities
poetry add tqdm wandb

# Visualization
poetry add matplotlib seaborn
```

### 3. Add PyG Optional Dependencies
```bash
# These go in pyproject.toml as URLs (Poetry supports PEP 508)
# OR install after poetry install:
poetry run pip install pyg_lib torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.9.0+cpu.html
```

### 4. Dev Dependencies
```bash
poetry add --group dev pytest pytest-cov black flake8 mypy jupyter
```

### 5. Lock and Install
```bash
poetry lock
poetry install
```

### 6. Verify Installation
```bash
poetry run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
poetry run python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
poetry run python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## MPS Setup Checklist (Mac)

Ensure these for MPS GPU acceleration:

1. **macOS version:** 12.3+ (preferably 13.0-14.x, avoid 26.0 Tahoe due to known issue)
2. **Apple Silicon:** M1/M2/M3/M4 chip
3. **Python:** Native ARM64 build (NOT Rosetta x86)
   ```bash
   python -c "import platform; print(platform.machine())"  # Should print "arm64", not "x86_64"
   ```
4. **PyTorch build:** MPS-enabled
   ```python
   import torch
   print(torch.backends.mps.is_built())  # Should be True
   print(torch.backends.mps.is_available())  # Should be True
   ```
5. **Device usage in code:**
   ```python
   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   model = model.to(device)
   ```

## Version Compatibility Matrix

| PyTorch | PyTorch Geometric | torchvision | Python | MPS Support |
|---------|-------------------|-------------|--------|-------------|
| 2.9.1 | 2.7.0 | 0.25.0 | 3.11-3.13 | Yes (macOS 12.3+) |
| 2.10.0 | 2.7.0 | 0.26.0 | 3.11-3.14 | Yes (but not CUDA 12.9) |
| 2.8.0 | 2.6.0 | 0.24.0 | 3.10-3.13 | Yes |

**Recommended for this project:** PyTorch 2.9.1 + PyG 2.7.0 (stable, well-tested, good MPS support except macOS 26.0).

**Confidence:** HIGH - From official PyTorch Geometric compatibility matrix and PyTorch release notes.

## What NOT to Use

### Deprecated/Outdated
- **PyTorch Geometric < 2.3:** Required external libraries (torch_scatter, etc.) as mandatory deps. Modern PyG (2.3+) made them optional. Avoid old tutorials recommending PyG 1.x.
- **torchvision.models.resnet50(pretrained=True):** Deprecated API. Use `weights=ResNet50_Weights.IMAGENET1K_V2` instead.
- **Conda for PyTorch Geometric:** Conda packages discontinued for PyTorch > 2.5.0. Use pip/Poetry only.

### Unnecessary for This Project
- **DGL (Deep Graph Library):** Good library, but PyG is standard. No reason to switch.
- **timm (PyTorch Image Models):** Overkill for ResNet-50. torchvision sufficient.
- **PyTorch Lightning:** Useful for large-scale training with multi-GPU, but adds abstraction complexity. Not needed for single-GPU Mac training + SUTD cluster (can use raw PyTorch).
- **Horovod / DeepSpeed:** Distributed training frameworks. Not needed for this project scale.
- **ray[tune]:** Hyperparameter optimization framework. Nice-to-have but not essential for initial implementation.

### Alternatives to Avoid for Graph Construction
- **networkx:** Pure Python, too slow for 400k node graphs. Use PyG's native graph structures.
- **graph-tool:** C++ backend, complex installation. Overkill for k-NN graph construction.
- **igraph:** Good library, but scikit-learn k-NN + scipy sparse is simpler and sufficient.

## Known Issues & Workarounds

### 1. macOS 26.0 (Tahoe) MPS Issue
**Problem:** PyTorch 2.9.1 reports MPS as built but not available on macOS 26.0.

**Workaround:**
- Downgrade to macOS 14 (Sonoma) or 13 (Ventura)
- OR wait for PyTorch 2.10+ which may fix this
- OR use CPU for local dev, GPU cluster for training

**Confidence:** MEDIUM - Based on GitHub issue, not official docs.

### 2. HuggingFace Datasets Timeout
**Problem:** Large file uploads can timeout with "Failed to upload file" errors.

**Workaround:**
- Use streaming mode: `load_dataset("aharley/rvl_cdip", streaming=True)`
- Download manually and load from disk: `load_dataset("imagefolder", data_dir="path/to/rvl_cdip")`
- Use mini dataset for prototyping: `load_dataset("dvgodoy/rvl_cdip_mini")`

**Confidence:** MEDIUM - From WebSearch (wandb reliability discussion).

### 3. wandb Blocked Training
**Problem:** wandb upload can block training process for hours with slow network.

**Workaround:**
- Run wandb in offline mode during training: `wandb.init(mode="offline")`
- Sync logs after training: `wandb sync`
- OR use TensorBoard for local dev, wandb only for final runs

**Confidence:** MEDIUM - From WebSearch comparison article.

### 4. Seaborn + PyTorch Tensor Incompatibility
**Problem:** Seaborn doesn't directly accept PyTorch tensors.

**Workaround:**
```python
# Always convert to numpy before plotting
tensor = model_output.detach().cpu().numpy()
sns.heatmap(tensor)
```

**Confidence:** HIGH - Verified through WebSearch and common PyTorch practice.

## Sources

### Official Documentation (HIGH Confidence)
- [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) via Context7
- [PyTorch MPS Backend](https://docs.pytorch.org/docs/stable/notes/mps.html) via Context7
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) via Context7
- [PyTorch Geometric GitHub Releases](https://github.com/pyg-team/pytorch_geometric/releases)
- [torchvision Models Documentation](https://docs.pytorch.org/vision/stable/models.html)
- [scikit-learn 1.8.0 Documentation](https://scikit-learn.org/stable/)

### HuggingFace Hub (HIGH Confidence)
- [aharley/rvl_cdip Dataset](https://huggingface.co/datasets/aharley/rvl_cdip)
- [dvgodoy/rvl_cdip_mini Dataset](https://huggingface.co/datasets/dvgodoy/rvl_cdip_mini)

### Community Resources (MEDIUM Confidence)
- [Python Packaging Best Practices 2026](https://dasroot.net/posts/2026/01/python-packaging-best-practices-setuptools-poetry-hatch/)
- [WandB vs TensorBoard Comparison](https://neptune.ai/vs/wandb-tensorboard)
- [PyTorch MPS Setup Guide](https://tillcode.com/apple-silicon-pytorch-mps-setup-and-speed-expectations/)
- [tqdm for PyTorch Training](https://adamoudad.github.io/posts/progress_bar_with_tqdm/)

### Known Issues (MEDIUM-LOW Confidence)
- [macOS 26 MPS Issue](https://github.com/pytorch/pytorch/issues/167679) - GitHub Issue #167679

## Confidence Summary

| Area | Confidence | Reason |
|------|------------|--------|
| Core Stack (PyTorch, PyG, torchvision) | HIGH | Verified via Context7 official docs, installation matrices |
| Data Loading (HuggingFace) | HIGH | Verified via Context7 and HuggingFace Hub |
| Graph Construction (scikit-learn, scipy) | HIGH | Verified via official scikit-learn docs |
| Training Utilities (tqdm) | HIGH | Standard tool, well-documented |
| Training Utilities (wandb vs TensorBoard) | MEDIUM | Based on comparison articles, not official benchmarks |
| Visualization (matplotlib, seaborn) | HIGH | Standard Python visualization stack |
| Package Management (Poetry) | HIGH | Specified in requirements, verified via WebSearch |
| MPS Compatibility | MEDIUM-HIGH | Official docs confirm support, but macOS 26 issue noted |
| Version Numbers | HIGH | From official release pages and PyPI |

## Recommendation for SUTD GPU Cluster

For SUTD GPU cluster deployment (assuming NVIDIA GPUs):

1. **Use CUDA wheels instead of CPU:**
   ```bash
   # Check CUDA version on cluster first
   pip install torch==2.9.1 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
   pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
   ```
2. **Device selection:**
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```
3. **Multi-GPU (if available):**
   - Consider PyTorch Lightning for cleaner multi-GPU code
   - OR use `torch.nn.DataParallel` / `DistributedDataParallel`

**Note:** This STACK.md focuses on Mac MPS for local development. Adjust CUDA versions based on cluster specs.

## Next Steps for Roadmap

Based on this stack, suggested phase structure:

1. **Environment Setup:** Poetry project, install dependencies, verify MPS
2. **Data Pipeline:** HuggingFace datasets, ResNet-50 feature extraction, k-NN graph construction
3. **Model Implementation:** GraphSAGE in PyTorch Geometric, global pooling for graph classification
4. **Training Loop:** tqdm progress, wandb logging, evaluation metrics
5. **Evaluation & Visualization:** scikit-learn metrics, matplotlib/seaborn plots

All tools in this stack are production-ready and well-integrated. No research flags anticipated for standard implementation.
