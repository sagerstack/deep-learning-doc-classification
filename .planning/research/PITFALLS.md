# Common Pitfalls: CNN+GNN Document Image Classification

Research findings on critical mistakes to avoid when building a GraphSAGE-based document classification pipeline. Each pitfall includes warning signs, prevention strategies, and phase mapping.

---

## 1. ResNet-50 Feature Extraction Pitfalls

### 1.1 Grayscale vs RGB Channel Mismatch

**Problem**: ImageNet pretrained ResNet-50 expects 3-channel RGB input, but RVL-CDIP contains grayscale document images (1 channel).

**Why it happens**: The first convolutional layer of ResNet-50 has filters expecting 3 channels. Directly feeding 1-channel images will fail.

**Warning signs**:
- Tensor shape errors in first conv layer: `RuntimeError: Given groups=1, weight of size [64, 3, 7, 7], expected input[N, 1, H, W]`
- Degraded performance if naive conversion (single channel → duplicate to 3)

**Prevention**:
- **Option A (Recommended)**: Replicate grayscale channel 3x to create pseudo-RGB: `img_rgb = img_gray.repeat(3, 1, 1)`
- **Option B**: Modify first conv layer to accept 1 channel (loses ImageNet pretrained weights for first layer)
- **Option C**: Use specially pretrained grayscale ResNet-50 models (rare, performance slightly weaker)

**Phase**: Data Preprocessing (Phase 1) + Model Architecture (Phase 2)

**References**:
- [Transfer Learning on Greyscale Images](https://towardsdatascience.com/transfer-learning-on-greyscale-images-how-to-fine-tune-pretrained-models-on-black-and-white-9a5150755c7a/)
- [Pretrained ResNet with Grayscale images](https://www.kaggle.com/code/romanrybalko/pretrained-resnet-with-grayscale-images)

---

### 1.2 Wrong Feature Extraction Layer Selection

**Problem**: Extracting features from the wrong ResNet layer leads to suboptimal graph representations.

**Layer trade-offs**:
- **Too early** (e.g., layer1, layer2): Low-level features (edges, textures), too fine-grained, miss semantic structure
- **Too late** (e.g., after avgpool, fc): Over-abstracted for spatial graph construction, already optimized for ImageNet classes
- **Just right** (layer3 or layer4 before pooling): Semantic features with spatial structure intact

**Warning signs**:
- Poor graph connectivity if features too low-level (nodes don't meaningfully relate)
- Loss of spatial resolution if extracted after global average pooling
- Graph construction yields disconnected components or trivial graphs

**Prevention**:
- Start with **layer3 or layer4** (before avgpool) for 7×7 or 14×14 spatial resolution
- Experiment in ablation: compare layer2 (28×28), layer3 (14×14), layer4 (7×7)
- Monitor graph statistics: avg degree, connected components, edge homophily

**Phase**: Architecture Design (Phase 2) + Ablation Studies (Phase 5)

**References**:
- [Exploring Feature Extraction with CNNs](https://towardsdatascience.com/exploring-feature-extraction-with-cnns-345125cefc9a/)
- [Deep Feature extraction](https://www.sciencedirect.com/topics/computer-science/deep-feature)

---

### 1.3 Frozen vs Fine-Tuned Backbone Decision

**Problem**: Unclear when to freeze ResNet vs fine-tune it end-to-end with GraphSAGE.

**Decision matrix**:

| Dataset Size | Domain Similarity | Recommendation |
|--------------|-------------------|----------------|
| Small (<10k) | Similar (documents) | **Freeze** all layers, train only GNN + classifier |
| Medium (10k-100k) | Similar | **Freeze** early layers (layer1-2), fine-tune layer3-4 + GNN |
| Large (>100k) | Similar | **Fine-tune** entire network with very low LR for CNN |
| Any | Very different | Fine-tune, but carefully with low LR |

**Warning signs**:
- Overfitting with small data + full fine-tuning (train-val gap widens)
- Destroying pretrained features (performance drops below frozen baseline)
- Training instability (loss spikes, gradient explosions)

**Prevention**:
1. **Always start frozen**: Train GNN+classifier first as baseline
2. If frozen performance plateaus, gradually unfreeze:
   - First unfreeze layer4
   - Then layer3
   - Rarely unfreeze layer1-2
3. Use **differential learning rates**:
   - Frozen layers: 0
   - Layer4: 1e-5
   - Layer3: 5e-5
   - GNN: 1e-3
4. Only fine-tune after frozen model converges

**Phase**: Training Strategy (Phase 3)

**References**:
- [Fine-tuning vs. Feature Extraction](https://apxml.com/courses/cnns-for-computer-vision/chapter-6-advanced-transfer-learning-domain-adaptation/fine-tuning-feature-extraction-advanced)
- [ResNet50 Fine-Tuning: Frozen Backbones Work Until They Don't](https://medium.com/betahumanai/i-used-resnet50-transfer-learning-and-my-accuracy-jumped-from-68-to-95-3e01fdf0f5e2)

---

### 1.4 Memory Explosion with 320k Feature Maps

**Problem**: Storing feature maps for 320k training images causes memory issues (100GB+ if not compressed).

**Scale calculation**:
- ResNet layer4 output: 2048 × 7 × 7 = 100,352 floats per image
- 320k images × 100,352 × 4 bytes = ~128GB uncompressed
- Plus graph construction memory overhead

**Warning signs**:
- Out-of-memory errors during dataset loading
- System swap thrashing (slow training)
- Unable to fit dataset in RAM

**Prevention strategies**:

**Option A: On-the-fly feature extraction**
- Extract features during training loop, don't store
- Trade compute for memory (acceptable if GPU has spare capacity)
- Implement caching for validation set only

**Option B: Disk-based storage with memory mapping**
- Save features to disk as memory-mapped arrays (np.memmap or HDF5)
- Load batches on demand
- Use compression (e.g., HDF5 with gzip, reduces 2-3x)

**Option C: Feature dimension reduction**
- Apply PCA or learned projection after feature extraction
- Reduce 2048-d to 512-d (4x memory savings, minimal performance loss)

**Option D: Mixed approach (Recommended)**
- Cache first epoch features to disk
- Load batches with num_workers > 0 in DataLoader
- Use pin_memory=True for faster GPU transfer

**Phase**: Data Engineering (Phase 1) + Infrastructure (Phase 2)

**References**:
- [HuggingFace datasets massively slows data reads](https://github.com/huggingface/datasets/issues/7753)
- [Use with PyTorch - datasets library](https://huggingface.co/docs/datasets/en/use_with_pytorch)

---

## 2. Graph Construction Pitfalls

### 2.1 Wrong k-NN Parameter

**Problem**: Choosing k too low or too high degrades graph quality.

**k too low** (e.g., k=3):
- Disconnected graph (multiple components)
- Insufficient message passing paths
- GNN cannot propagate information globally

**k too high** (e.g., k=50):
- Over-connected graph (nearly complete)
- Expensive computation (O(k) per node per layer)
- Noisy edges dilute meaningful connections
- Over-smoothing accelerates

**Warning signs**:
- Check graph statistics:
  - Number of connected components > 1 (k too low)
  - Average degree > 30 (k likely too high)
  - Edge homophily < 0.3 (k too high, random connections)
- Training time explosion (k too high)
- Poor performance compared to CNN baseline (k too low or too high)

**Prevention**:
1. **Grid search k in validation**: Try k ∈ {5, 10, 15, 20} on small subset
2. **Measure graph properties**:
   - Average degree: target 10-20
   - Connected components: should be 1
   - Diameter: should be < 10 for good propagation
3. **Adaptive k**: Use different k based on spatial density (corners get fewer neighbors)

**Recommended starting point**: k=10 for 7×7 grids (49 nodes), k=15 for 14×14 grids

**Phase**: Graph Construction (Phase 2)

**References**:
- [Efficient k-nearest neighbor graph construction](https://www.cs.princeton.edu/cass/papers/www11.pdf)
- [K-Nearest Neighbors - Neo4j Graph Data Science](https://neo4j.com/docs/graph-data-science/current/algorithms/knn/)

---

### 2.2 Uniform Spatial Treatment

**Problem**: Treating all spatial positions equally ignores semantic importance in documents.

**Why it matters for documents**:
- Document headers/titles (top region) are semantically critical
- Margins and whitespace (edges) carry little information
- Central text regions contain discriminative content

**Warning signs**:
- Graph connects many background/whitespace nodes
- Performance doesn't improve over global average pooling baseline
- Learned attention weights are nearly uniform

**Prevention**:

**Option A: Semantic k-NN** (weight distance by feature similarity + spatial proximity)
```python
spatial_dist = euclidean_distance(positions)
feature_sim = cosine_similarity(features)
combined_dist = alpha * spatial_dist + (1-alpha) * (1 - feature_sim)
# Connect k nearest neighbors by combined_dist
```

**Option B: Content-aware node filtering**
- Compute feature magnitude: `node_importance = ||features||_2`
- Remove low-magnitude nodes (likely background) before graph construction
- Reduces graph size and noise

**Option C: Position embeddings**
- Add learned position embeddings to features before k-NN
- Allows model to learn importance of spatial regions

**Phase**: Advanced Graph Construction (Phase 4)

---

### 2.3 Graph Size Explosion

**Problem**: High-resolution feature maps create massive graphs (e.g., 28×28 = 784 nodes per image).

**Memory and compute costs**:
- 784 nodes/image × 320k images = 250M nodes
- With k=15: 3.75B edges
- GraphSAGE neighbor sampling becomes bottleneck

**Warning signs**:
- Training extremely slow (hours per epoch on GPU)
- GPU memory errors during graph batching
- Majority of time spent in data loading, not training

**Prevention**:

**Option A: Lower resolution** (Recommended for initial experiments)
- Use layer4 (7×7 = 49 nodes) instead of layer3 (14×14 = 196 nodes)
- 4x fewer nodes, 4x fewer edges
- Start here, only increase if accuracy plateaus

**Option B: Graph pooling**
- Apply graph pooling layer to downsample graph
- TopKPooling or SAGPooling to select most important nodes
- Reduce 196 nodes → 49 nodes before message passing

**Option C: Hierarchical graphs**
- Multi-scale graph: coarse graph (7×7) + fine graph (14×14)
- Message passing at coarse level, attention to fine level when needed

**Option D: Neighbor sampling reduction**
- Reduce GraphSAGE sampling: [15, 10, 5] instead of [25, 25]
- 36x speedup with minimal accuracy loss

**Phase**: Architecture Optimization (Phase 4)

**References**:
- [Accelerating Training and Inference of GNNs](https://arxiv.org/pdf/2110.08450)
- [Graph pooling review](https://link.springer.com/article/10.1007/s10462-024-10918-9)

---

### 2.4 Feature Map Resolution Trade-off

**Problem**: High resolution preserves spatial detail but creates computational burden.

**Resolution options**:

| Layer | Resolution | Nodes/image | Pros | Cons |
|-------|-----------|-------------|------|------|
| layer2 | 28×28 | 784 | Fine spatial detail | Huge graphs, slow training |
| layer3 | 14×14 | 196 | Balance | Moderate compute |
| layer4 | 7×7 | 49 | Fast, manageable | May lose fine structure |

**Warning signs**:
- Training takes >4 hours/epoch (too high resolution for your hardware)
- Accuracy doesn't improve with higher resolution (plateau indicates sufficient granularity)

**Prevention**:
1. **Start with 7×7** (layer4), establish baseline
2. **Ablate resolution**: Compare 7×7 vs 14×14 on validation set
3. **Early stopping rule**: If 14×14 doesn't improve >2% over 7×7, not worth the cost
4. Monitor **training time vs accuracy** trade-off

**Phase**: Architecture Design (Phase 2) + Ablation (Phase 5)

**References**:
- [Improving the Resolution of CNN Feature Maps](https://arxiv.org/pdf/1805.10766)
- [Multiscale Deep Learning review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9573412/)

---

## 3. GraphSAGE-Specific Pitfalls

### 3.1 Wrong Aggregation Function

**Problem**: Choice of aggregation function (mean, LSTM, pool) impacts performance and training time.

**Performance comparison** (from original GraphSAGE paper):
- **LSTM aggregator**: Highest accuracy, but 2× slower than pool
- **Pool aggregator**: Nearly identical accuracy to LSTM, faster
- **Mean aggregator**: Slightly lower accuracy, fastest, most stable

**For document images specifically**:
- Documents have **structured spatial layout** (unlike social networks)
- **Pool aggregator** works best: captures max activation patterns (e.g., presence of logos, stamps)
- **Mean aggregator** is safer fallback, less prone to over-smoothing

**Warning signs**:
- LSTM aggregator diverges or overfits on small data
- Mean aggregator underperforms if document features have high variance
- Pool aggregator gives NaN/Inf (numerical instability with ReLU)

**Prevention**:
1. **Default to pool aggregator** for document images
2. **Ablate**: Compare mean vs pool on validation set
3. **Skip LSTM** unless pool/mean plateau (not worth 2× slowdown for marginal gain)
4. Use proper initialization for pool aggregator to avoid numerical issues

**Phase**: Model Architecture (Phase 2) + Hyperparameter Tuning (Phase 3)

**References**:
- [GraphSAGE paper](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)
- [Why LSTM aggregator obtained highest f1 score?](https://github.com/williamleif/GraphSAGE/issues/20)
- [GraphSAGE aggregation functions comparison](https://www.emergentmind.com/topics/graphsage-with-lstm-aggregation)

---

### 3.2 Over-Smoothing with Too Many Layers

**Problem**: Deep GNNs cause node representations to become indistinguishable (over-smoothing).

**What happens**:
- Layer 1-2: Node features incorporate local neighborhood
- Layer 3-4: Features start converging
- Layer 5+: All nodes have nearly identical representations (useless for classification)

**Why it happens**:
- Graph convolution is a **smoothing operator** (averaging neighbors)
- Repeated averaging causes features to converge to graph mean
- Spatial graphs (like 7×7 grids) have short diameter → faster over-smoothing

**Warning signs**:
- Validation accuracy decreases when adding more GNN layers
- Feature similarity matrix becomes nearly uniform (all ~1.0)
- Gradient norms collapse in later GNN layers
- Performance with 3 layers < performance with 2 layers

**Prevention**:

**Rule of thumb**: Use 2-3 GraphSAGE layers maximum for spatial grids

**Techniques to mitigate**:
1. **Residual connections**: `h_new = h_old + GraphSAGE(h_old)`
2. **Normalization**: Apply LayerNorm or BatchNorm between GNN layers
3. **Adaptive layer weights**: Learn to weight each layer's contribution
4. **Early stopping**: Monitor validation performance, stop adding layers when it degrades

**Don't**:
- Don't blindly stack 5+ GNN layers (common beginner mistake)
- Don't ignore validation performance degradation when going deeper

**Phase**: Architecture Design (Phase 2) + Hyperparameter Tuning (Phase 3)

**References**:
- [Over-smoothing issue in GNN](https://towardsdatascience.com/over-smoothing-issue-in-graph-neural-network-bddc8fbc2472/)
- [Oversmoothing in GNNs: why does it happen so fast?](https://medium.com/@xinyiwu98/oversmoothing-in-gnns-why-does-it-happen-so-fast-6bbe93ef97a7)
- [Mitigating over-smoothing through Adaptive Early Embedding](https://www.sciencedirect.com/science/article/abs/pii/S0950705125006616)

---

### 3.3 Graph-Level Readout Strategy

**Problem**: How to aggregate node features into a single graph-level representation for classification.

**Common readout functions**:

| Method | Pros | Cons | When to use |
|--------|------|------|-------------|
| **Mean pooling** | Simple, stable | Treats all nodes equally | Baseline, works for balanced importance |
| **Max pooling** | Captures salient features | Ignores majority of nodes | Sparse discriminative features (logos, stamps) |
| **Sum pooling** | Preserves magnitude info | Sensitive to graph size | Variable-size graphs (not applicable here) |
| **Attention pooling** | Learns node importance | More parameters, can overfit | When specific regions matter (e.g., headers) |

**For document classification**:
- **Mean pooling**: Safe default, works well for whole-page documents
- **Attention pooling**: Better if document structure varies (some classes have headers, others don't)
- **Max pooling**: Works if discriminative features are sparse (e.g., letterhead logos)

**Warning signs**:
- Mean pooling gives uniform attention weights when visualized (should try attention)
- Attention pooling overfits (train acc >> val acc)
- Performance doesn't improve over simple mean (stick with mean)

**Prevention**:
1. **Start with mean pooling** (simplest, fewest parameters)
2. **Ablate attention pooling** only if mean plateaus
3. **Visualize attention weights** to verify they make semantic sense
4. **Compare readout in ablation study** (Phase 5)

**Phase**: Architecture Design (Phase 2) + Ablation Studies (Phase 5)

**References**:
- [Graph Pooling and Readout Functions](https://apxml.com/courses/graph-neural-networks-gnns/chapter-4-advanced-gnn-tasks-techniques/graph-pooling-readout)
- [Multi-level attention pooling for GNNs](https://www.sciencedirect.com/science/article/pii/S0893608021004299)

---

### 3.4 Batch Size and Memory with Variable-Size Graphs

**Problem**: Even though RVL-CDIP has fixed-size images, the graphs can have variable sizes if using adaptive k-NN or node filtering.

**Memory scaling**:
- PyTorch Geometric batches graphs by concatenating into one giant disjoint graph
- Batch of 32 graphs with 49 nodes each = 1568 nodes in batch
- With GraphSAGE neighbor sampling [15, 10], each node samples neighbors recursively
- **Neighborhood explosion**: 1 node requires 15×10 = 150 neighbor features

**Warning signs**:
- OOM errors with large batch sizes (>32)
- Training much slower than expected
- GPU utilization low (<50%) despite large model

**Prevention**:

**Reduce neighbor sampling**:
- Default [25, 25] is overkill for spatial graphs
- Try [15, 10] or [10, 5] → 36x speedup with minimal accuracy loss
- Spatial graphs are dense, don't need aggressive sampling

**Adjust batch size**:
- Smaller batch size (16-32) for graph models vs CNNs (64-128)
- Monitor GPU memory usage, target 80-90% utilization
- Use gradient accumulation if need larger effective batch size

**Optimize DataLoader**:
- Use `num_workers > 0` for parallel data loading
- Set `pin_memory=True` for faster CPU→GPU transfer
- Pre-compute and cache graphs to disk if on-the-fly construction is bottleneck

**Phase**: Training Infrastructure (Phase 3)

**References**:
- [Advanced Mini-Batching in PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/2.5.2/advanced/batching.html)
- [GraphSAGE batch size selection](https://github.com/dmlc/dgl/issues/3906)
- [Variable size graphs batch training](https://github.com/stellargraph/stellargraph/issues/1275)

---

## 4. Training Pitfalls

### 4.1 MPS (Mac) Compatibility Issues

**Problem**: PyTorch Geometric and its dependencies don't have pre-built binaries for Apple Silicon (M1/M2/M3).

**What breaks**:
- `torch_scatter`, `torch_sparse`, `torch_cluster` require building from source
- Some operations fall back to CPU (slow)
- MPS backend doesn't support all PyTorch ops (PYTORCH_ENABLE_MPS_FALLBACK needed)

**Warning signs**:
- Installation errors: "No matching distribution found for torch-scatter"
- Runtime errors: "operation not implemented for MPS"
- Training on MPS is slower than CPU (indicates CPU fallback)

**Prevention**:

**For local development (Mac)**:
1. Install dependencies from source (requires Xcode, CMake, Boost, Eigen)
2. Set environment variable: `export PYTORCH_ENABLE_MPS_FALLBACK=1`
3. **Test on small dataset first** to verify MPS compatibility
4. **Expect slower training** than CUDA (MPS backend less mature)

**Recommended workflow**:
- **Prototyping**: Mac with small data subset (100 images), CPU is fine
- **Full training**: SUTD GPU cluster with CUDA (much faster)
- **Don't** try to do full 320k training on Mac MPS

**For course project**:
- Assign one person to handle infrastructure setup
- Document installation steps in README
- Provide both Mac (MPS) and Linux (CUDA) instructions

**Phase**: Infrastructure Setup (Phase 0)

**References**:
- [MPS PyTorch Nightly with Mac M1 does not work](https://github.com/rusty1s/pytorch_cluster/issues/172)
- [How to Install PyTorch Geometric with Apple Silicon Support](https://medium.com/@dessi.georgieva8/how-to-install-pytorch-geometric-with-apple-silicon-support-m1-m2-m3-39f1a5ad33b6)
- [Developing Pytorch Geometric on M1](https://project-delphi.github.io/ml-blog/posts/developing-pytorch-geometric-on-m1/)

---

### 4.2 Mixed Precision Training Gotchas

**Problem**: Automatic Mixed Precision (AMP) can cause numerical instability with GNNs.

**Common issues**:
1. **Gradient underflow**: fp16 range is [6e-8, 65504], GNN gradients can underflow
2. **Loss scaling issues**: GradScaler may not recover from NaN/Inf
3. **Incompatible operations**: Some PyG operations don't support fp16

**Warning signs**:
- NaN/Inf in loss after a few iterations
- Gradients go to zero (underflow)
- Training diverges with AMP but works in fp32
- "operation not supported for half" errors

**Prevention**:

**Don't use AMP initially**:
- Start with fp32 training to establish baseline
- Only try AMP after confirming model works

**If using AMP**:
1. Wrap only forward pass (not backward) in autocast
2. Use GradScaler with conservative settings
3. Unscale gradients before clipping: `scaler.unscale_(optimizer)` then `clip_grad_norm_`
4. Monitor loss for NaN/Inf, disable AMP if unstable

**For course project**:
- **Skip AMP** unless training is painfully slow
- Focus on model architecture, not training speedups
- Document if AMP doesn't work (acceptable for project report)

**Phase**: Training Optimization (Phase 4) - optional

**References**:
- [Automatic Mixed Precision in PyTorch](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)

---

### 4.3 Learning Rate Scheduling for CNN+GNN

**Problem**: End-to-end training requires different learning rates for CNN and GNN components.

**Why differential LR matters**:
- **ResNet** (if fine-tuning): Pretrained, needs low LR (1e-5) to avoid destroying features
- **GraphSAGE**: Random init, needs higher LR (1e-3) to learn quickly
- Using same LR for both leads to slow convergence or instability

**Warning signs**:
- CNN fine-tuning degrades performance below frozen baseline (LR too high)
- GNN doesn't learn (LR too low)
- Loss oscillates (LR too high for CNN)

**Prevention**:

**Setup 1: Frozen CNN + trainable GNN** (recommended start)
```python
optimizer = torch.optim.Adam([
    {'params': gnn.parameters(), 'lr': 1e-3},
], lr=1e-3)
```

**Setup 2: Fine-tune CNN + train GNN**
```python
optimizer = torch.optim.Adam([
    {'params': resnet.layer4.parameters(), 'lr': 1e-5},
    {'params': resnet.layer3.parameters(), 'lr': 5e-5},
    {'params': gnn.parameters(), 'lr': 1e-3},
], lr=1e-3)
```

**Scheduler recommendations**:
- **CosineAnnealingLR** or **ReduceLROnPlateau**: work well for most cases
- **1cycle policy**: achieves fast convergence, worth trying
- **Warmup**: Use 5-10% of total steps for warmup when fine-tuning

**Don't**:
- Don't fine-tune CNN in first epoch (train GNN first)
- Don't use aggressive schedules (StepLR with gamma=0.1 every 10 epochs is too harsh)

**Phase**: Training Strategy (Phase 3)

**References**:
- [Learning Rate Scheduling best practices](https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler)
- [Learning to Schedule Learning Rate with GNNs](https://openreview.net/forum?id=k7efTb0un9z)
- [1cycle policy for neural networks](https://apxml.com/courses/cnns-for-computer-vision/chapter-2-advanced-training-optimization/learning-rate-schedules)

---

### 4.4 Data Loading Bottleneck with On-the-Fly Graph Construction

**Problem**: Constructing graphs on-the-fly during training creates CPU bottleneck, GPU sits idle.

**Scale of problem**:
- Feature extraction: ~50ms/image on GPU
- k-NN graph construction: ~100ms/image on CPU
- Result: GPU utilization <30%, training 3x slower than necessary

**Warning signs**:
- `nvidia-smi` shows low GPU utilization (<50%)
- Training time dominated by data loading (monitor with `tqdm`)
- Increasing batch size doesn't speed up training

**Prevention**:

**Option A: Pre-compute and cache graphs** (Recommended)
1. Run offline preprocessing: extract features + build graphs for all images
2. Save to disk: `.pt` files or HDF5
3. Training loop just loads pre-built graphs
4. Trade: disk space (~20GB) for 3-5x training speedup

**Option B: Parallelize on-the-fly construction**
1. Use DataLoader with `num_workers=4` (or more)
2. Implement graph construction in `__getitem__` (runs in parallel)
3. Use `pin_memory=True` for faster GPU transfer
4. Trade: CPU cores for no disk overhead

**Option C: Mixed approach**
1. Cache graphs for training set (320k images)
2. On-the-fly for validation set (40k images, less critical)

**HuggingFace datasets caveat**:
- IterableDataset with transformations can be 1000x slower
- Use `dataset.map()` to preprocess before training, not in DataLoader
- Avoid `with_transform()` for heavy preprocessing

**Phase**: Data Engineering (Phase 1) + Training Infrastructure (Phase 3)

**References**:
- [HuggingFace datasets slow loading issue](https://github.com/huggingface/datasets/issues/7753)
- [Use with PyTorch - datasets library](https://huggingface.co/docs/datasets/en/use_with_pytorch)
- [Prefetching for IterableDataset](https://github.com/huggingface/datasets/issues/5878)

---

## 5. Evaluation Pitfalls

### 5.1 Only Reporting Overall Accuracy on Balanced Dataset

**Problem**: RVL-CDIP is balanced (25k images/class), but overall accuracy hides per-class performance issues.

**Why this matters**:
- Some classes are inherently harder (e.g., "letter" vs "email" are visually similar)
- Model may excel on easy classes, fail on hard ones
- Overall accuracy of 85% could mean 95% on 10 classes, 60% on 6 classes

**What you're missing**:
- Class-specific failure modes
- Confusion between similar classes
- Whether GNN helps uniformly or only for specific document types

**Warning signs**:
- High overall accuracy but poor performance on specific classes when manually inspected
- Confusion matrix shows model predicts only 8/16 classes confidently

**Prevention**:

**Report per-class metrics**:
- Per-class precision, recall, F1
- Confusion matrix (16×16)
- Identify hardest classes (lowest F1)

**Analyze GNN benefit per class**:
- Compare CNN baseline vs GNN on each class
- GNN should help on classes with spatial structure (forms, invoices)
- If GNN hurts on some classes, investigate why

**For course project report**:
- Include confusion matrix heatmap
- Table of per-class F1 scores
- Discuss which classes benefit from GNN

**Phase**: Evaluation & Analysis (Phase 5)

**References**:
- [On Evaluation of Document Classification with RVL-CDIP](https://aclanthology.org/2023.eacl-main.195.pdf)
- [RVL-CDIP dataset documentation](https://adamharley.com/rvl-cdip/)

---

### 5.2 Unfair CNN Baseline Comparison

**Problem**: Comparing GNN against a weak CNN baseline inflates perceived GNN benefit.

**Common mistakes**:
1. **Weak CNN**: Using vanilla ResNet-50 with random init vs GNN with pretrained ResNet
2. **Different training setups**: CNN trained for 10 epochs, GNN for 50 epochs
3. **Different hyperparameters**: CNN with suboptimal LR, GNN with tuned LR
4. **Unfair data augmentation**: GNN uses augmentation, CNN doesn't

**What you're actually measuring**:
- Not "Does GNN help?"
- But "Does GNN + pretrained ResNet + more epochs + better hyperparams beat random init CNN?"

**Warning signs**:
- CNN baseline accuracy is far below published results on RVL-CDIP
- GNN shows huge improvement (>10%) over baseline (too good to be true)
- Different training configurations for baseline vs GNN

**Prevention**:

**Fair CNN baseline**:
1. Use **same pretrained ResNet-50** as GNN backbone
2. Apply global average pooling → linear classifier (standard approach)
3. Use **same hyperparameters**: LR, optimizer, epochs, augmentation
4. Train with same random seed for reproducibility

**Report both**:
- CNN baseline (global avg pooling)
- GNN model (graph construction + GraphSAGE)
- Improvement: `(GNN - CNN) / CNN × 100%`

**Expected improvement**:
- If GNN gives >5% improvement: likely significant
- If GNN gives <1% improvement: may not be worth complexity
- If GNN is worse than CNN: implementation bug or over-smoothing

**Phase**: Baseline Establishment (Phase 2) + Final Comparison (Phase 5)

---

### 5.3 Forgetting RVL-CDIP-N Generalization

**Problem**: Only evaluating on RVL-CDIP test set, not testing generalization to RVL-CDIP-N (noisy version).

**What is RVL-CDIP-N**:
- Subset of RVL-CDIP with added noise, realistic document degradation
- Tests robustness to real-world conditions (scanning artifacts, compression)
- State-of-the-art models show accuracy drop on noisy data

**Why it matters for course project**:
- Shows model robustness (important for real deployment)
- Differentiates over-fitted models from truly robust ones
- GNNs may be more/less robust than CNNs (research question!)

**Warning signs**:
- Model has 95% accuracy on clean test set but fails on noisy documents
- Forgot to evaluate on any robustness benchmark

**Prevention**:

**Evaluation plan**:
1. Primary metric: RVL-CDIP test set accuracy
2. Secondary metric: RVL-CDIP-N accuracy (if available)
3. Report accuracy drop: `Δ = acc_clean - acc_noisy`

**If time permits** (Phase 6 - nice-to-have):
- Evaluate on RVL-CDIP-N
- Analyze which classes degrade most with noise
- Test if GNN is more robust than CNN baseline

**Phase**: Evaluation (Phase 5) + Stretch Goals (Phase 6)

**References**:
- [On Evaluation of Document Classification using RVL-CDIP](https://arxiv.org/abs/2306.12550)
- [Document Image Classification with Intra-Domain Transfer Learning](https://arxiv.org/pdf/1801.09321)

---

### 5.4 Ablation Studies That Don't Control for Confounders

**Problem**: Changing multiple variables at once makes it impossible to attribute performance gains.

**Bad ablation example**:
- Model A: 2-layer GraphSAGE, mean aggregation, 7×7 resolution, frozen ResNet
- Model B: 3-layer GraphSAGE, pool aggregation, 14×14 resolution, fine-tuned ResNet
- Result: B outperforms A by 5%
- **Question**: What caused the improvement? Can't tell! (4 variables changed)

**Good ablation example**:
- Baseline: 2-layer GraphSAGE, mean aggregation, 7×7 resolution, frozen ResNet
- Ablation 1: Change only aggregation (mean → pool), keep rest same
- Ablation 2: Change only resolution (7×7 → 14×14), keep rest same
- Ablation 3: Change only depth (2 layers → 3 layers), keep rest same
- Now you can attribute performance change to each factor

**Warning signs**:
- Ablation table changes multiple hyperparameters per row
- Cannot explain which design choice contributed to improvement
- Reviewers (TAs/professor) ask "How do you know it was the GNN and not the fine-tuning?"

**Prevention**:

**Ablation best practices**:
1. **Change one variable at a time**
2. **Fix random seed** across ablations (use same train/val split)
3. **Report variance** (run each ablation with 3 different seeds, report mean ± std)
4. **Statistical significance**: Use t-test to check if improvement is significant

**Required ablations for course project**:
| Ablation | Variables | Purpose |
|----------|-----------|---------|
| Baseline | CNN only (global avg pooling) | Establish if GNN helps at all |
| Aggregation | mean vs pool vs LSTM | Find best aggregation |
| Depth | 1 vs 2 vs 3 GNN layers | Find optimal depth |
| Resolution | 7×7 vs 14×14 | Check if higher resolution helps |
| k-NN | k=5 vs 10 vs 15 vs 20 | Optimal graph connectivity |

**Phase**: Ablation Studies (Phase 5)

**References**:
- [Ablation Studies in Deep Learning](https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence))
- [AutoAblation: Automated Parallel Ablation Studies](https://www.researchgate.net/publication/352015513_AutoAblation_Automated_Parallel_Ablation_Studies_for_Deep_Learning)

---

## 6. Course Project Specific Pitfalls

### 6.1 Spending Too Much Time on Infrastructure vs Model Work

**Problem**: Over-engineering data pipeline, Docker setup, logging infrastructure instead of focusing on model experiments.

**Time budget reality** (April 17 deadline):
- Infrastructure (10%): Environment setup, data loading, basic training loop
- Model development (40%): Implementing GNN variants, debugging, initial experiments
- Experiments (30%): Ablation studies, hyperparameter tuning, analysis
- Report + presentation (20%): Writing, making slides, practicing

**Warning signs**:
- Week 1-2: Still setting up Docker, complex data pipeline, wandb integration
- Week 3: Haven't run a single GNN model yet
- Week 4: Panic mode, no time for ablations or analysis

**Prevention**:

**Minimum viable infrastructure** (Week 1):
- Load RVL-CDIP from HuggingFace datasets ✓
- Extract ResNet-50 features (frozen) ✓
- Build k-NN graph (simple scipy/sklearn implementation) ✓
- Train 2-layer GraphSAGE with mean aggregation ✓
- Compute accuracy on val set ✓

**Don't prematurely optimize**:
- Skip Docker (use conda/venv, everyone has same requirements.txt)
- Skip wandb (use TensorBoard or simple CSV logging)
- Skip complex data pipeline (cache features to disk, load with PyG DataLoader)
- Skip distributed training (single GPU is enough for 320k images)

**For team of 3**:
- Person A: Baseline CNN + GraphSAGE-mean
- Person B: GraphSAGE-pool variant
- Person C: GraphSAGE with attention readout
- Everyone shares same infrastructure code

**Phase**: Project planning (Phase 0)

---

### 6.2 Not Documenting for Reproducibility

**Problem**: Code that only runs on one person's laptop, unclear how to reproduce results.

**Common reproducibility failures**:
1. **Hardcoded paths**: `/Users/alice/data/rvl-cdip/` instead of relative paths
2. **Missing random seeds**: Results change every run
3. **Undocumented hyperparameters**: Used LR=1e-3 but didn't write it down
4. **Missing dependencies**: Works on Mac, breaks on Linux (e.g., PyG installation)
5. **No training logs**: Can't reproduce the "85% accuracy" claim

**Warning signs**:
- Teammate can't run your code
- TA asks "How did you get 87%?" and you don't remember hyperparameters
- Can't reproduce your own results from 2 weeks ago

**Prevention**:

**Reproducibility checklist**:
- [ ] `requirements.txt` or `pyproject.toml` with exact versions
- [ ] README with installation instructions (Mac + Linux)
- [ ] Config file (YAML or Python dict) with all hyperparameters
- [ ] Set random seeds: `torch.manual_seed(42)`, `np.random.seed(42)`
- [ ] Training log with hyperparameters: save to `logs/experiment_name/config.yaml`
- [ ] Model checkpoints: save best model with filename including hyperparameters

**Minimal documentation**:
```python
# config.yaml
model:
  gnn_layers: 2
  aggregation: "mean"
  hidden_dim: 256
  k_neighbors: 10
  resolution: "7x7"  # layer4

training:
  lr: 0.001
  batch_size: 32
  epochs: 50
  seed: 42

data:
  dataset: "aharley/rvl_cdip"
  train_size: 320000
  val_size: 40000
```

**For course project**:
- Include `reproduce.sh` script: installs deps → trains model → reports accuracy
- 1-2 page setup guide in README
- Upload training logs + config to submission

**Phase**: Throughout project (Phase 0-6)

**References**:
- [Towards Training Reproducible Deep Learning Models](https://arxiv.org/pdf/2202.02326)
- [How to Solve Reproducibility in ML](https://neptune.ai/blog/how-to-solve-reproducibility-in-ml)
- [Reproducibility in Machine Learning](https://reproducible.cs.princeton.edu/)

---

### 6.3 Code That Only Runs on One Machine

**Problem**: Code works on Mac (MPS) but breaks on SUTD GPU cluster (CUDA), or vice versa.

**Common portability issues**:
1. **Device hardcoded**: `model.to('cuda')` fails on Mac
2. **Path separators**: Windows `\` vs Unix `/`
3. **PyG installation differences**: Mac requires source build, Linux has wheels
4. **Data paths**: `/scratch/dataset/` on cluster, `./data/` locally

**Warning signs**:
- Works on your laptop, crashes on cluster
- Teammate with different OS can't run code
- Demo fails during presentation because presenter's laptop is different

**Prevention**:

**Device-agnostic code**:
```python
device = torch.device('cuda' if torch.cuda.is_available()
                     else 'mps' if torch.backends.mps.is_available()
                     else 'cpu')
model.to(device)
```

**Path handling**:
```python
from pathlib import Path
data_dir = Path(__file__).parent / "data"  # works on all OS
```

**Environment variables for cluster**:
```python
import os
data_path = os.getenv('RVL_CDIP_PATH', './data/rvl-cdip')
```

**Test on both environments early**:
- Week 1: Get code running on both Mac (local) and cluster
- Week 2: Automate with shell scripts for both environments

**For course project**:
- Designate one person as "infrastructure lead"
- Test on all 3 team members' machines before major deadlines
- Provide separate instructions: `README_MAC.md`, `README_CLUSTER.md`

**Phase**: Infrastructure setup (Phase 0-1)

---

## Summary: Phase-Wise Pitfall Priorities

### Phase 0: Setup (Week 1)
- **Critical**: MPS compatibility issues (4.1)
- **Critical**: Code portability (6.3)
- **Important**: Infrastructure vs model work balance (6.1)

### Phase 1: Data Preprocessing (Week 1-2)
- **Critical**: Grayscale vs RGB handling (1.1)
- **Critical**: Memory explosion with 320k images (1.4)
- **Important**: Feature extraction layer selection (1.2)

### Phase 2: Model Architecture (Week 2-3)
- **Critical**: Over-smoothing with too many layers (3.2)
- **Critical**: Wrong k-NN parameter (2.1)
- **Important**: Aggregation function choice (3.1)
- **Important**: Graph size explosion (2.3)
- **Important**: Readout strategy (3.3)

### Phase 3: Training (Week 3-4)
- **Critical**: Data loading bottleneck (4.4)
- **Critical**: Frozen vs fine-tuned decision (1.3)
- **Important**: Learning rate scheduling (4.3)
- **Important**: Batch size and memory (3.4)

### Phase 4: Optimization (Week 4-5)
- **Important**: Feature map resolution trade-off (2.4)
- **Optional**: Mixed precision training (4.2)
- **Optional**: Semantic k-NN and advanced graph construction (2.2)

### Phase 5: Evaluation & Analysis (Week 5-6)
- **Critical**: Fair CNN baseline (5.2)
- **Critical**: Proper ablation studies (5.4)
- **Critical**: Per-class analysis (5.1)
- **Critical**: Reproducibility documentation (6.2)
- **Important**: RVL-CDIP-N generalization (5.3)

---

## Quick Reference: Most Critical Pitfalls

**Top 5 pitfalls that will tank your project**:

1. **Over-smoothing** (3.2): Using >3 GNN layers, performance degrades
2. **Memory issues** (1.4, 2.3): Running out of RAM/GPU memory, can't train
3. **Unfair baseline** (5.2): Comparing against weak CNN, results are meaningless
4. **Bad ablations** (5.4): Changing multiple variables, can't draw conclusions
5. **Non-reproducible code** (6.2): Can't reproduce results for report/demo

**Top 3 pitfalls that waste the most time**:

1. **Infrastructure over-engineering** (6.1): Spending 3 weeks on Docker instead of models
2. **Data loading bottleneck** (4.4): GPU idle, training 10x slower than necessary
3. **MPS compatibility** (4.1): Fighting with Mac installation instead of using cluster

---

## Sources

This research draws from the following sources:

**Transfer Learning & CNNs:**
- [Transfer Learning on Greyscale Images](https://towardsdatascience.com/transfer-learning-on-greyscale-images-how-to-fine-tune-pretrained-models-on-black-and-white-9a5150755c7a/)
- [Exploring Feature Extraction with CNNs](https://towardsdatascience.com/exploring-feature-extraction-with-cnns-345125cefc9a/)
- [Fine-tuning vs. Feature Extraction: Advanced Considerations](https://apxml.com/courses/cnns-for-computer-vision/chapter-6-advanced-transfer-learning-domain-adaptation/fine-tuning-feature-extraction-advanced)

**Graph Neural Networks:**
- [GraphSAGE Original Paper](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)
- [Over-smoothing issue in Graph Neural Network](https://towardsdatascience.com/over-smoothing-issue-in-graph-neural-network-bddc8fbc2472/)
- [Graph Pooling and Readout Functions](https://apxml.com/courses/graph-neural-networks-gnns/chapter-4-advanced-gnn-tasks-techniques/graph-pooling-readout)
- [Advanced Mini-Batching — PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/2.5.2/advanced/batching.html)

**PyTorch & Infrastructure:**
- [How to Install PyTorch Geometric with Apple Silicon Support](https://medium.com/@dessi.georgieva8/how-to-install-pytorch-geometric-with-apple-silicon-support-m1-m2-m3-39f1a5ad33b6)
- [PyTorch Mixed Precision Training](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
- [HuggingFace datasets - Use with PyTorch](https://huggingface.co/docs/datasets/en/use_with_pytorch)

**Evaluation & Reproducibility:**
- [On Evaluation of Document Classification with RVL-CDIP](https://aclanthology.org/2023.eacl-main.195.pdf)
- [Towards Training Reproducible Deep Learning Models](https://arxiv.org/pdf/2202.02326)
- [How to Solve Reproducibility in ML](https://neptune.ai/blog/how-to-solve-reproducibility-in-ml)
