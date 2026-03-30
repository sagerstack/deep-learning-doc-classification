# Architecture Patterns: CNN+GraphSAGE Document Image Classification

**Domain:** Document Image Classification with Graph Neural Networks
**Researched:** 2026-03-25
**Overall confidence:** MEDIUM-HIGH

## Executive Summary

A CNN+GraphSAGE pipeline for document image classification involves five core components with clear data flow: (1) Data Module handles loading and preprocessing, (2) Feature Extractor (ResNet-50) extracts spatial feature maps, (3) Graph Builder converts feature maps to PyG Data objects, (4) GNN Classifier (GraphSAGE + readout + head) performs graph-level classification, and (5) Training Orchestrator manages the training loop, validation, and checkpointing.

The critical architectural decision is **which ResNet-50 layer to extract features from** (layer4 output before avgpool preserves 7×7 spatial structure with 2048 channels for ResNet-50) and **how to construct the graph** (patch-based nodes with k-NN spatial connectivity is the most common pattern, though this remains a research variable for this project).

Build order follows natural dependencies: Data Module → Feature Extractor → Graph Builder → GNN Classifier → Training Orchestrator → Evaluation Module.

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Module                             │
│  - Load RVL-CDIP images + labels                               │
│  - Preprocessing (resize, normalize)                            │
│  - Caching (optional, for feature maps)                         │
└────────────────┬────────────────────────────────────────────────┘
                 │ Image tensors [B, 3, H, W]
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Feature Extractor                           │
│  - ResNet-50 (pretrained on ImageNet)                          │
│  - Extract from layer4 (before avgpool)                        │
│  - Output: Spatial feature maps [B, 2048, 7, 7]               │
│  - Decision: Frozen vs Fine-tuned                              │
└────────────────┬────────────────────────────────────────────────┘
                 │ Feature maps [B, 2048, 7, 7]
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Graph Builder                              │
│  - Convert feature maps to graph nodes                         │
│  - Define edge connectivity (k-NN spatial / grid / learned)    │
│  - Create PyG Data objects (x, edge_index, y)                  │
│  - Handle batching (batch vector for DataLoader)               │
└────────────────┬────────────────────────────────────────────────┘
                 │ PyG Data(x, edge_index, y, batch)
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GNN Classifier                              │
│  - GraphSAGE layers (2-3 layers typical)                       │
│  - Aggregation: mean/max/LSTM                                  │
│  - Global pooling (readout): mean/max/attention/Set2Set        │
│  - Classification head: Linear(hidden → num_classes)           │
└────────────────┬────────────────────────────────────────────────┘
                 │ Class logits [B, 16]
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Training Orchestrator                          │
│  - Training loop (forward, loss, backward, step)               │
│  - Validation loop (metrics: accuracy, F1, confusion matrix)   │
│  - Checkpointing (best model based on validation loss/F1)     │
│  - Early stopping (patience=10 typical for GNNs)               │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Evaluation Module                            │
│  - Per-class precision/recall/F1                               │
│  - Macro F1 score                                              │
│  - Confusion matrix visualization                              │
│  - Comparison with CNN baseline                                │
└─────────────────────────────────────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | Inputs | Outputs | Dependencies |
|-----------|---------------|--------|---------|--------------|
| **Data Module** | Load images, preprocess, cache | Raw image paths, labels | Image tensors [B, 3, H, W] | None |
| **Feature Extractor** | Extract spatial features from CNN | Image tensors | Feature maps [B, C, H_f, W_f] | Data Module |
| **Graph Builder** | Convert feature maps to PyG graphs | Feature maps, labels | PyG Data objects | Feature Extractor |
| **GNN Classifier** | Graph-level classification | PyG Data (x, edge_index, batch) | Class logits [B, num_classes] | Graph Builder |
| **Training Orchestrator** | Manage training/validation loops | DataLoader, model, optimizer | Trained model, checkpoints | GNN Classifier |
| **Evaluation Module** | Compute metrics, visualizations | Model predictions, ground truth | Metrics, plots, comparisons | Training Orchestrator |

### Data Flow

```
Image (PIL/tensor)
  → [Data Module: resize, normalize]
  → Tensor [B, 3, 224, 224]
  → [Feature Extractor: ResNet-50 layer4]
  → Feature map [B, 2048, 7, 7]
  → [Graph Builder: flatten spatial, build edges]
  → PyG Data(x=[N, 2048], edge_index=[2, E], y=[1])
  → [DataLoader: batch graphs]
  → Batch(x=[N_batch, 2048], edge_index=[2, E_batch], batch=[N_batch], y=[B])
  → [GraphSAGE: message passing]
  → Node embeddings [N_batch, hidden_dim]
  → [Global pooling: aggregate to graph-level]
  → Graph embeddings [B, hidden_dim]
  → [Classification head: Linear]
  → Class logits [B, 16]
  → [Loss: CrossEntropyLoss]
  → Scalar loss
```

**Key observations**:
- Batch dimension `B` varies by DataLoader batch_size
- Node count per graph `N` depends on graph construction strategy (e.g., 49 nodes for 7×7 patch-based)
- Edge count `E` depends on connectivity (k-NN: ~k×N edges)
- `batch` vector maps nodes to their graph index within the batch

## Critical Architectural Decisions

### 1. ResNet-50 Feature Extraction Layer

**Decision**: Extract features from **layer4 output (before avgpool)** to preserve spatial structure.

**Options**:
| Layer | Output Shape (ImageNet input) | Spatial Info | Channels | Use Case |
|-------|-------------------------------|--------------|----------|----------|
| layer1 | [B, 256, 56, 56] | High resolution | Low semantic | Early features, texture |
| layer2 | [B, 512, 28, 28] | Medium resolution | Medium semantic | Balanced |
| layer3 | [B, 1024, 14, 14] | Lower resolution | High semantic | Semantic features |
| **layer4** | **[B, 2048, 7, 7]** | **Spatial grid** | **Highest semantic** | **Graph construction** |
| avgpool | [B, 2048, 1, 1] | **No spatial info** | Highest semantic | Global features (not suitable) |

**Rationale**:
- **layer4** provides the best trade-off: high-level semantic features with preserved 7×7 spatial structure
- Avgpool destroys spatial information (collapses to 1×1), making graph construction impossible
- Earlier layers (layer1-3) have more spatial detail but less semantic content

**Sources**:
- [ResNet feature extraction layers](https://medium.com/the-owl/extracting-features-from-an-intermediate-layer-of-a-pretrained-model-in-pytorch-c00589bda32b)
- [ResNet spatial dimensions](https://discuss.pytorch.org/t/how-can-extract-the-features-map-of-resnet-50/97900)

### 2. Graph Construction Strategy

**Decision**: This is a **research variable** for the project. Common patterns:

#### Pattern A: Patch-Based Nodes (Recommended Starting Point)

**Method**:
1. Extract feature map from ResNet-50 layer4: `[B, 2048, 7, 7]`
2. Treat each spatial location `(i, j)` as a node
3. Node features: `feature_map[:, :, i, j]` → `[2048]` per node
4. Total nodes per image: `7 × 7 = 49` nodes

**Edge construction options**:
- **Grid connectivity**: Connect spatially adjacent patches (4-neighbors or 8-neighbors)
- **k-NN spatial**: Connect k nearest neighbors in 2D spatial grid (e.g., k=4 or k=6)
- **k-NN feature space**: Connect k nearest neighbors in feature space (dynamic, computed per layer)
- **Learned edges**: Use attention or learnable functions to determine connectivity

**Advantages**:
- Simple, interpretable
- Natural spatial structure
- Fixed graph size (49 nodes for ResNet-50 layer4)

**Disadvantages**:
- Coarse spatial granularity (7×7 grid)
- Fixed to CNN's spatial resolution

**Sources**:
- [Vision GNN patch-based approach](https://arxiv.org/abs/2206.00272)
- [PyG k-NN graph construction](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/point_cloud.rst)

#### Pattern B: Pixel-Based Nodes (Higher Resolution)

**Method**:
1. Use earlier ResNet layer (e.g., layer3: `[B, 1024, 14, 14]`)
2. Each pixel becomes a node: `14 × 14 = 196` nodes
3. Edge construction: k-NN or grid

**Trade-off**:
- ✅ More spatial detail
- ❌ Lower semantic features
- ❌ Larger graphs (slower GNN)

#### Pattern C: Region-Based Nodes (Superpixel-Like)

**Method**:
1. Cluster feature map spatial locations into regions
2. Each region = one node
3. Region features = aggregated (mean/max pool) over cluster

**Trade-off**:
- ✅ Flexible graph size
- ✅ Learned regions
- ❌ More complex preprocessing
- ❌ Less interpretable

**Sources**:
- [Image classification with multiscale wavelet superpixels](https://www.sciencedirect.com/science/article/abs/pii/S016786552300003X)
- [WaveMesh superpixeling algorithm](https://medium.com/data-science/graph-convolutional-networks-introduction-to-gnns-24b3f60d6c95)

### 3. Edge Definition

**Decision**: Start with **k-NN spatial connectivity**, experiment with k values (k=4, 6, 8) as ablation study.

**Options**:

| Edge Type | How It Works | Pros | Cons |
|-----------|-------------|------|------|
| **k-NN spatial** | Connect k nearest neighbors in 2D grid | Simple, local connectivity | Fixed k, grid-based |
| **Grid (4/8-neighbors)** | Connect to adjacent patches | Very simple, interpretable | Only immediate neighbors |
| **k-NN feature space** | Connect k nearest in feature embedding | Dynamic, learned structure | Recomputed per layer, expensive |
| **Radius-based** | Connect all nodes within radius r | Adaptive edge count | Variable graph structure |
| **Fully connected** | Connect all nodes | Max information flow | Expensive, loses spatial bias |
| **Attention-based** | Learn edge weights with attention | Learnable, flexible | Complex, harder to train |

**Recommendation for RVL-CDIP**:
- **Start with k-NN spatial (k=6)**: Balances local structure with enough connectivity
- **Ablation study**: Compare k=4, 6, 8 to understand impact
- **Advanced**: If time permits, try k-NN feature space (dynamic graph construction like DynamicEdgeConv)

**Implementation** (PyG):
```python
from torch_geometric.nn import knn_graph

# Option 1: k-NN in 2D spatial coordinates
pos = create_2d_grid_positions(7, 7)  # [49, 2] spatial coordinates
edge_index = knn_graph(pos, k=6, loop=False)

# Option 2: k-NN in feature space (dynamic)
edge_index = knn_graph(node_features, k=6, loop=False)
```

**Sources**:
- [PyG knn_graph API](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.rst)
- [Dynamic EdgeConv with k-NN](https://github.com/pyg-team/pytorch_geometric/blob/master/docs/source/tutorial/create_gnn.rst)

### 4. Frozen vs Fine-Tuned ResNet-50

**Decision**: **Start frozen, optionally fine-tune later** (standard transfer learning protocol).

**Training Strategy**:

| Phase | ResNet-50 Status | Learning Rate | Why |
|-------|-----------------|---------------|-----|
| **Phase 1** | Frozen (all layers) | LR=1e-3 (GNN only) | Train GNN classifier to convergence first |
| **Phase 2** | Fine-tune layer4 only | LR=1e-5 (ResNet), 1e-4 (GNN) | Adapt high-level features to document domain |
| **Phase 3** (optional) | Fine-tune layer3+layer4 | LR=1e-6 (ResNet), 1e-4 (GNN) | Full feature adaptation if needed |

**Rationale**:
- **Frozen first**: Prevents destroying pretrained features with random GNN gradients
- **Very low LR for fine-tuning**: ResNet features are generic; small updates sufficient
- **Layer-wise fine-tuning**: Later layers benefit more from domain adaptation

**Dynamic Backbone Freezing (DBF)**: Recent research (2024) shows dynamic freezing schedules can reduce training time by 1.7 days while improving performance (+0.77 mAP on remote sensing object detection). Consider if training time becomes a bottleneck.

**Sources**:
- [Fine-tuning strategy](https://towardsdatascience.com/cnn-transfer-learning-fine-tuning-9f3e7c5806b2/)
- [Dynamic Backbone Freezing](https://arxiv.org/abs/2407.15143)
- [Freezing layers guide](https://www.exxactcorp.com/blog/deep-learning/guide-to-freezing-layers-in-ai-models)

### 5. GraphSAGE Configuration

**Decision**: 2-layer GraphSAGE with mean aggregation, global mean pooling for readout.

**Architecture**:
```python
GraphSAGE Layer 1: [2048] → [512]   # Reduce dimensionality
GraphSAGE Layer 2: [512] → [256]     # Further compress
Global Mean Pool: [N, 256] → [1, 256]  # Aggregate to graph-level
Classification Head: [256] → [16]    # Map to 16 classes
```

**Aggregator Options**:
| Aggregator | How It Works | Pros | Cons |
|------------|-------------|------|------|
| **Mean** | Average neighbor features | Simple, permutation-invariant | Less expressive |
| **Max** | Max-pool neighbor features | Captures salient features | Loses information |
| **LSTM** | LSTM over neighbors | More expressive | Requires neighbor ordering |
| **Pooling** | MLP + max-pool | Learnable, expressive | More parameters |

**Recommendation**: **Start with mean aggregation** (standard GraphSAGE), ablate to max if needed.

**Global Pooling (Readout) Options**:
| Pooling | Use Case | Complexity |
|---------|----------|-----------|
| **Global Mean** | Simple averaging | Low |
| **Global Max** | Salient feature detection | Low |
| **Global Add** | Sum aggregation | Low |
| **Attention** | Weighted aggregation | Medium |
| **Set2Set** | Order-invariant, learnable | High |
| **GraphMultisetTransformer** | Transformer-based | High |

**Recommendation**: **Start with global mean pooling** (standard for graph classification), try attention if performance needs improvement.

**Sources**:
- [GraphSAGE paper (Hamilton et al. 2017)](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)
- [PyG GraphSAGE implementation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GraphSAGE.html)
- [Global pooling methods](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html)

## Build Order and Dependencies

### Phase 1: Data Module (Foundation)
**Priority**: Build first
**Rationale**: All other components depend on data loading

**Tasks**:
1. Create PyTorch Dataset for RVL-CDIP (using HuggingFace `datasets` library)
2. Implement preprocessing (resize to 224×224, normalize with ImageNet stats)
3. Test with subset (5-10% of data) for fast iteration
4. Add optional caching for feature maps (avoid recomputing ResNet forward pass)

**Output**: DataLoader yielding `(image, label)` batches

**No blockers**.

---

### Phase 2: Feature Extractor (CNN Backbone)
**Priority**: Build second
**Rationale**: Graph builder needs feature maps

**Tasks**:
1. Load pretrained ResNet-50 from torchvision
2. Extract layer4 output (before avgpool) using forward hooks or manual forward
3. Verify output shape: `[B, 2048, 7, 7]` for 224×224 input
4. Implement frozen mode (requires_grad=False) and fine-tuning mode

**Output**: Feature maps `[B, 2048, 7, 7]`

**Dependencies**: Data Module (Phase 1)

---

### Phase 3: Graph Builder (Feature-to-Graph)
**Priority**: Build third
**Rationale**: GNN classifier needs PyG Data objects

**Tasks**:
1. Flatten feature maps to nodes: `[B, 2048, 7, 7]` → `[49, 2048]` per image
2. Implement k-NN spatial edge construction (start with k=6)
3. Create PyG `Data(x, edge_index, y)` objects
4. Handle batching: PyG DataLoader automatically creates `batch` vector
5. Make k-NN parameter configurable for ablation studies

**Output**: PyG `Data` objects ready for GNN

**Dependencies**: Feature Extractor (Phase 2)

**Implementation pattern** (PyG Dataset):
```python
class GraphDocumentDataset(InMemoryDataset):
    def __init__(self, feature_maps, labels, k_neighbors=6):
        # feature_maps: precomputed or computed on-the-fly
        # labels: class labels
        super().__init__(root=None, transform=None)

    def process(self):
        data_list = []
        for feat_map, label in zip(feature_maps, labels):
            # feat_map: [2048, 7, 7]
            x = feat_map.view(2048, -1).T  # [49, 2048]
            pos = create_grid_positions(7, 7)  # [49, 2]
            edge_index = knn_graph(pos, k=self.k_neighbors)
            data = Data(x=x, edge_index=edge_index, y=label)
            data_list.append(data)
        self.save(data_list, self.processed_paths[0])
```

---

### Phase 4: GNN Classifier (GraphSAGE + Readout + Head)
**Priority**: Build fourth
**Rationale**: Training orchestrator needs the model

**Tasks**:
1. Implement GraphSAGE layers (2 layers: 2048→512→256)
2. Add global pooling (mean) for graph-level representation
3. Add classification head (Linear: 256→16)
4. Verify forward pass: `Data(x, edge_index, batch)` → `[B, 16]`
5. Make architecture configurable (num_layers, hidden_dim, aggregation type)

**Output**: Model producing class logits

**Dependencies**: Graph Builder (Phase 3)

**Implementation pattern**:
```python
class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512,
                 out_channels=256, num_classes=16, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.global_pool = global_mean_pool
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        x = self.global_pool(x, batch)  # [num_graphs, out_channels]
        x = self.classifier(x)  # [num_graphs, num_classes]
        return x
```

---

### Phase 5: Training Orchestrator (Training Loop)
**Priority**: Build fifth
**Rationale**: Evaluation needs a trained model

**Tasks**:
1. Implement training loop (forward, loss, backward, optimizer step)
2. Implement validation loop (no gradient updates)
3. Add checkpointing (save best model based on validation F1 or loss)
4. Add early stopping (patience=10 epochs typical for GNNs)
5. Log metrics to console or TensorBoard/Weights & Biases
6. Support reproducibility (seed setting, deterministic ops)

**Output**: Trained model checkpoint

**Dependencies**: GNN Classifier (Phase 4)

**Best Practices**:
- **Checkpointing**: Save model state + optimizer state + epoch for resumability
- **Validation metric**: Use macro F1 (accounts for class imbalance in RVL-CDIP)
- **Early stopping**: Monitor validation loss or F1, stop if no improvement for 10 epochs
- **Gradient checkpointing** (optional): Use if memory becomes bottleneck for deep GNNs

**Sources**:
- [GNN training best practices](https://nusit.nus.edu.sg/services/hpc-newsletter/deep-learning-best-practices-checkpointing-deep-learning-model-training/)
- [GNN checkpointing](https://dl.acm.org/doi/fullHtml/10.1145/3656019.3676892)

---

### Phase 6: Evaluation Module (Metrics & Visualization)
**Priority**: Build last
**Rationale**: Depends on trained model

**Tasks**:
1. Compute per-class precision, recall, F1
2. Compute macro F1 (primary metric for comparison)
3. Generate confusion matrix (16×16 for RVL-CDIP)
4. Compare against CNN baseline (from reference notebook)
5. Visualize results (plots, tables)
6. Test on RVL-CDIP-N (out-of-distribution evaluation)

**Output**: Metrics report, visualizations, baseline comparison

**Dependencies**: Training Orchestrator (Phase 5)

**Metrics**:
- Accuracy (overall)
- Per-class precision, recall, F1
- Macro F1 (average F1 across classes, accounts for imbalance)
- Confusion matrix (identify misclassification patterns)

---

## Patterns to Follow

### Pattern 1: Modular Architecture with Clear Interfaces

**What**: Separate each component into its own module with well-defined inputs/outputs.

**Why**: Enables independent testing, easier debugging, and flexible experimentation (e.g., swap graph construction strategies without touching GNN code).

**Example**:
```
src/
├── data/
│   ├── dataset.py         # RVL-CDIP Dataset
│   └── transforms.py      # Preprocessing
├── models/
│   ├── feature_extractor.py  # ResNet-50 wrapper
│   ├── graph_builder.py      # Feature map → PyG Data
│   └── gnn_classifier.py     # GraphSAGE + head
├── training/
│   ├── trainer.py         # Training loop
│   └── checkpointing.py   # Save/load logic
└── evaluation/
    └── metrics.py         # Evaluation metrics
```

### Pattern 2: Configuration-Driven Experiments

**What**: Store all hyperparameters in config files (YAML/JSON), not hardcoded.

**Why**: Reproducibility, easy ablation studies, clear experiment tracking.

**Example** (`config.yaml`):
```yaml
model:
  resnet_layer: layer4
  freeze_resnet: true
  graphsage_layers: 2
  hidden_dim: 512
  k_neighbors: 6
  aggregation: mean
  global_pool: mean

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10

data:
  subset_fraction: 0.1  # Use 10% for dev
```

### Pattern 3: PyG Dataset with Lazy vs Eager Processing

**What**: Choose between computing graphs on-the-fly (lazy) or precomputing all graphs (eager).

**When**:
- **Lazy** (compute during `__getitem__`): For large datasets, limited storage
- **Eager** (compute in `process()`, save to disk): For smaller datasets, faster training

**Trade-off**:
- Lazy: Lower storage, slower training (recomputes each epoch)
- Eager: Higher storage, faster training (one-time computation)

**Recommendation for RVL-CDIP**:
- **Dev (10% subset)**: Eager (precompute and cache)
- **Full (320k)**: Lazy or cache feature maps only (graphs are cheap to compute from feature maps)

### Pattern 4: Feature Map Caching

**What**: Save ResNet-50 feature maps to disk, load during graph construction.

**Why**: Avoids recomputing CNN forward pass every epoch (expensive on CPU/MPS).

**Implementation**:
```python
# 1. Extract and cache feature maps (one-time)
for image, label in tqdm(dataloader):
    with torch.no_grad():
        feat_map = resnet_feature_extractor(image)  # [B, 2048, 7, 7]
        save_to_disk(feat_map, label, image_id)

# 2. Load cached feature maps during training
class CachedGraphDataset(InMemoryDataset):
    def process(self):
        for feat_map, label in load_cached_features():
            data = build_graph(feat_map, label)
            data_list.append(data)
```

**Trade-off**:
- ✅ Faster training (no CNN forward pass)
- ✅ Enables frozen ResNet without runtime overhead
- ❌ Storage cost (~320k images × 7×7×2048 × 4 bytes = ~40 GB for full dataset)
- ❌ Not possible if fine-tuning ResNet

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Using AvgPool Output for Graph Construction

**What goes wrong**: Extracting from ResNet-50's avgpool layer gives `[B, 2048, 1, 1]` — no spatial structure, only one "node" per image.

**Why bad**: Cannot construct meaningful graph, defeats purpose of GNN.

**Instead**: Extract from **layer4** (before avgpool) to preserve 7×7 spatial grid.

---

### Anti-Pattern 2: Fully Connected Graphs

**What goes wrong**: Connecting all 49 nodes to all other nodes (48 edges per node × 49 nodes = 2352 edges per image).

**Why bad**:
- Computationally expensive (O(N²) edges)
- Loses spatial locality bias (why use a graph if everything connects to everything?)
- GNN message passing becomes equivalent to global attention

**Instead**: Use k-NN (k=4-8) or grid connectivity to preserve local structure.

---

### Anti-Pattern 3: Training GNN with Unfrozen Random ResNet

**What goes wrong**: Initializing ResNet randomly and training end-to-end from scratch.

**Why bad**:
- Destroys pretrained features
- Requires far more data and compute
- Large gradient updates from random GNN destroy CNN features early

**Instead**: Use pretrained ResNet-50, freeze initially, fine-tune later with low LR.

---

### Anti-Pattern 4: Ignoring Class Imbalance in Validation

**What goes wrong**: Using only accuracy to evaluate model on RVL-CDIP.

**Why bad**: RVL-CDIP may have class imbalance (some document types more common than others). Accuracy can be misleading.

**Instead**: Use **macro F1** as primary metric (averages F1 across all classes equally, regardless of class frequency).

---

### Anti-Pattern 5: Large Batch Size for GNNs

**What goes wrong**: Using batch_size=128 or 256 (common for CNNs) with GNNs.

**Why bad**:
- PyG batches graphs by concatenating nodes and edges (creates very large graph)
- Memory usage scales with number of nodes and edges in batch
- May cause OOM errors

**Instead**: Start with batch_size=32 or 64, tune based on memory availability.

---

## Scalability Considerations

| Concern | At 10% Subset (32k) | At Full Dataset (320k) | Mitigation |
|---------|---------------------|------------------------|------------|
| **Feature map storage** | ~4 GB | ~40 GB | Use lazy computation or compress (fp16) |
| **Graph construction time** | Negligible (precompute) | ~1-2 hours (one-time) | Cache graphs to disk (InMemoryDataset) |
| **Training time per epoch** | ~5-10 min (GPU) | ~50-100 min (GPU) | Use GPU cluster, reduce batch size if OOM |
| **Memory usage** | Fits in 16 GB GPU | May require 32 GB GPU or gradient checkpointing | Use gradient checkpointing for deep GNNs |
| **Validation time** | ~1-2 min | ~10-20 min | Acceptable, no mitigation needed |

---

## Open Questions for Experimentation

These are **research variables** that should be explored through ablation studies:

1. **Optimal k value for k-NN edges**: k=4 vs k=6 vs k=8
2. **GraphSAGE depth**: 2 layers vs 3 layers vs 4 layers (watch for over-smoothing)
3. **Aggregation function**: mean vs max vs LSTM
4. **Global pooling**: mean vs max vs attention
5. **ResNet layer**: layer3 vs layer4 (trade-off: spatial resolution vs semantic level)
6. **Fine-tuning strategy**: Frozen vs fine-tune layer4 vs fine-tune layer3+layer4
7. **Edge definition**: k-NN spatial vs grid vs k-NN feature space

**Ablation study suggestion**: Keep all variables fixed except one, measure impact on macro F1.

---

## Sources

**High Confidence (Official Documentation, Papers)**:
- [GraphSAGE paper (Hamilton et al. 2017)](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [PyG Creating Datasets Tutorial](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html)
- [PyG k-NN Graph Construction](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/point_cloud.rst)
- [Doc-GCN Paper (Luo et al. 2022)](https://aclanthology.org/2022.coling-1.256/)
- [Doc-GCN GitHub Implementation](https://github.com/adlnlp/doc_gcn)

**Medium Confidence (Verified with Multiple Sources)**:
- [ResNet Feature Extraction Layers](https://medium.com/the-owl/extracting-features-from-an-intermediate-layer-of-a-pretrained-model-in-pytorch-c00589bda32b)
- [ResNet Spatial Dimensions Discussion](https://discuss.pytorch.org/t/how-can-extract-the-features-map-of-resnet-50/97900)
- [CNN Fine-Tuning Strategy](https://towardsdatascience.com/cnn-transfer-learning-fine-tuning-9f3e7c5806b2/)
- [Dynamic Backbone Freezing (DBF) Paper](https://arxiv.org/abs/2407.15143)
- [Vision GNN Paper (2022)](https://arxiv.org/abs/2206.00272)
- [Image Classification with Superpixel Graphs](https://www.sciencedirect.com/science/article/abs/pii/S016786552300003X)

**Low Confidence (General Guidance, Not Project-Specific)**:
- [GNN Training Best Practices](https://nusit.nus.edu.sg/services/hpc-newsletter/deep-learning-best-practices-checkpointing-deep-learning-model-training/)
- [Graph Convolutional Networks Overview](https://medium.com/data-science/graph-convolutional-networks-introduction-to-gnns-24b3f60d6c95)

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Component boundaries | HIGH | Clear separation based on PyG patterns, verified with official docs |
| Data flow | HIGH | Standard PyG pipeline, verified with Context7 code examples |
| ResNet-50 layer extraction | HIGH | Verified with PyTorch forums and multiple sources |
| Graph construction patterns | MEDIUM-HIGH | Multiple valid approaches exist; patch-based with k-NN is well-established but project-specific choice remains experimental |
| GraphSAGE implementation | HIGH | Official PyG implementation available, clear API |
| Build order | HIGH | Natural dependency chain, no circular dependencies |
| Training best practices | MEDIUM | General GNN practices verified, but RVL-CDIP-specific tuning needed |

---

## Summary for Roadmap

**Recommended Phase Structure**:

1. **Phase 1: Data Foundation** - Build Data Module, verify subset loading
2. **Phase 2: Feature Extraction** - Integrate ResNet-50, extract layer4 features
3. **Phase 3: Graph Construction** - Implement patch-based graph builder with k-NN
4. **Phase 4: GNN Classifier** - Implement GraphSAGE + readout + classification head
5. **Phase 5: Training Pipeline** - Build training loop, checkpointing, early stopping
6. **Phase 6: Evaluation** - Compute metrics, compare with CNN baseline
7. **Phase 7: Ablation Studies** - Experiment with k values, depth, aggregation
8. **Phase 8: Full Training** - Scale to full 320k dataset on GPU cluster

**Key Dependency**: Phases 1-6 are sequential (each depends on previous). Phases 7-8 can iterate based on findings.

**Critical Path**: Data Module → Feature Extractor → Graph Builder → GNN Classifier (these cannot be parallelized).

**Research Flags**: Phase 3 (graph construction strategy) and Phase 7 (ablation studies) require deeper investigation and experimentation — they are the core research contribution of this project.
