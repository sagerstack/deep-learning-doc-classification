# Project Research Summary

**Project:** GNN-based Document Image Classification (SUTD Course Project)
**Domain:** Deep Learning - Graph Neural Networks for Document Classification
**Researched:** 2026-03-25
**Confidence:** MEDIUM-HIGH

## Executive Summary

This is an academic deep learning project applying GraphSAGE (a Graph Neural Network architecture) to document image classification on the RVL-CDIP dataset (400k document images, 16 classes). The standard expert approach uses a CNN+GNN pipeline: extract spatial features from pretrained ResNet-50, construct graphs from feature maps, then apply GraphSAGE for graph-level classification. This architecture is well-documented in academic literature with established patterns from 2017-2026.

The recommended implementation uses PyTorch 2.9.1 + PyTorch Geometric 2.7 for the GNN stack, HuggingFace datasets for RVL-CDIP loading, and scikit-learn for k-NN graph construction. The project has strict grading criteria (50% technical implementation, 20% performance, 30% report/presentation) requiring specific features: data pipeline, CNN+GraphSAGE architecture, training with validation, accuracy/precision/recall/F1 metrics, confusion matrix, CNN baseline comparison, and reproducibility. Expected completion timeline is 4-6 weeks (April 17 deadline).

Key risks center on three technical challenges: (1) managing 320k+ feature maps in memory (requires caching strategy or lazy computation), (2) avoiding over-smoothing from deep GNN layers (stick to 2-3 layers maximum), and (3) ensuring fair CNN baseline comparison (same pretrained weights, hyperparameters, training setup). The core research contribution is graph construction strategy—converting CNN feature maps to meaningful graph structures—which lacks standardized approaches for document images and requires experimentation.

## Key Findings

### Recommended Stack

The 2026 standard for GraphSAGE document classification combines mature, production-ready libraries with strong Mac MPS support for local development and CUDA compatibility for GPU cluster training. All components have been verified as compatible with each other and the SUTD infrastructure constraints.

**Core technologies:**
- **PyTorch 2.9.1 + PyTorch Geometric 2.7**: Neural network framework with GNN operations — latest stable release pair, fully compatible, supports Mac MPS for local GPU acceleration and CUDA for SUTD cluster
- **torchvision 0.25**: ResNet-50 pretrained feature extractor — provides ImageNet weights, modern API with `weights` parameter, maintains spatial structure at layer4 (7×7 feature maps)
- **HuggingFace datasets 2.18**: RVL-CDIP data loading — official dataset available as `aharley/rvl_cdip` (400k images), automatic train/val/test splits, streaming support for large datasets
- **scikit-learn 1.8 + scipy 1.14**: k-NN graph construction — battle-tested `kneighbors_graph()` for spatial connectivity, efficient ball-tree algorithm, sparse matrix integration with PyG
- **Poetry 1.8**: Package management — specified in project requirements, deterministic builds via poetry.lock, better dependency resolution than pip
- **wandb 0.18 or TensorBoard 2.16**: Experiment tracking — wandb recommended for reproducibility and multi-run comparison, TensorBoard acceptable for local-only tracking

**Mac MPS compatibility notes**: PyTorch 2.9.1 supports MPS on macOS 12.3+ but has a known issue on macOS 26.0 (Tahoe) where MPS reports as built but unavailable. Recommended development environment is macOS 13-14 with native ARM64 Python (not Rosetta). For production training, SUTD GPU cluster with CUDA is strongly recommended over Mac MPS.

### Expected Features

The feature landscape is tightly constrained by the course rubric, which explicitly requires specific components for technical implementation (50% of grade), performance evaluation (20%), and reproducibility (part of coding 25%). Features are categorized by grading impact, not user value, since this is an academic project evaluated on implementation quality and research depth.

**Must have (table stakes):**
- Data loading pipeline with RVL-CDIP 320k/40k/40k splits — required for reproducibility grading
- ResNet-50 CNN feature extraction with fine-tuning decision — core architecture component
- Graph construction from feature maps — the project's research contribution, evaluation criteria unclear but implementation required
- GraphSAGE implementation with k-hop sampling and aggregation — assigned GNN variant per project spec
- Training loop with validation and early stopping — required for performance evaluation rubric
- Accuracy, precision, recall, F1, confusion matrix — rubric explicitly requires these metrics
- CNN baseline comparison — required to demonstrate GNN value proposition
- Reproducibility setup (seeds, config files, documented commands) — "thoroughly documented code" worth 25% of coding grade

**Should have (competitive):**
- Ablation studies: k parameter (k=3,5,7,9), GraphSAGE depth (1,2,3 layers), aggregation function (mean/pool/LSTM) — demonstrates systematic evaluation for performance score (20%) and creativity bonus (5%)
- Per-class performance analysis and confusion matrix heatmap — deeper evaluation beyond macro metrics, shows understanding of failure modes
- Out-of-distribution evaluation on RVL-CDIP-N — tests generalization to noisy documents, strengthens performance evaluation
- Failure case visualization with explanations — rubric explicitly mentions "show model failure cases"
- Graph visualization for sample documents — interpretability component, targets creativity bonus
- Multiple graph construction strategies (k-NN spatial vs grid vs region adjacency) — research depth for creativity bonus, though implementation complexity is high

**Defer (v2+ or out of scope):**
- Over-engineered deployment infrastructure (FastAPI, Docker serving) — not evaluated, wastes time before core work done
- Real-time serving optimizations or latency monitoring — not a rubric requirement, premature optimization
- Custom data augmentation research or multiple CNN backbone comparison — CNN is means to feature extraction, graph construction is the contribution
- Novel GNN architecture design — project is about applying GraphSAGE, not inventing new architectures
- Production-grade MLOps pipeline or distributed training — academic project, single-GPU SUTD cluster sufficient
- Extensive related work literature review — report section, not implementation effort

### Architecture Approach

A CNN+GraphSAGE pipeline follows a five-stage sequential architecture with clear component boundaries and data flow. The critical architectural decision is which ResNet-50 layer to extract features from (layer4 output at 7×7 spatial resolution before avgpool preserves spatial structure with 2048-channel semantic features) and how to construct graph edges (patch-based nodes with k-NN spatial connectivity is the most common pattern, though this remains a research variable).

**Major components:**
1. **Data Module** — loads RVL-CDIP images and labels from HuggingFace, applies preprocessing (resize to 224×224, normalize with ImageNet stats, convert grayscale to 3-channel), outputs image tensors [B, 3, 224, 224]
2. **Feature Extractor (ResNet-50)** — extracts spatial feature maps from layer4 output (before avgpool) to preserve 7×7 spatial grid, decision between frozen (faster, baseline) vs fine-tuned (better performance, more compute), outputs feature maps [B, 2048, 7, 7]
3. **Graph Builder** — converts feature maps to PyG Data objects by treating each spatial position (i, j) as a node with features [2048], constructs edges using k-NN spatial connectivity (recommended k=6-10 for 7×7 grids), produces Data(x=[49, 2048], edge_index=[2, E], y=[1]) per image
4. **GNN Classifier (GraphSAGE + readout)** — 2-3 layer GraphSAGE with mean/pool aggregation for message passing, global mean pooling to aggregate node features to graph-level representation, linear classification head mapping to 16 classes, outputs class logits [B, 16]
5. **Training Orchestrator** — manages training loop (forward, CrossEntropyLoss, backward, optimizer step), validation loop with metrics (accuracy, F1, confusion matrix), checkpointing best model based on validation F1, early stopping with patience=10 typical for GNNs

**Build order follows natural dependencies**: Data Module → Feature Extractor → Graph Builder → GNN Classifier → Training Orchestrator → Evaluation Module. Components 1-4 are strictly sequential (each depends on previous output). Component 5 (Training) depends on components 1-4. Component 6 (Evaluation) can proceed in parallel with training for visualization/metric code development using dummy data.

**Key pattern**: Separate frozen ResNet training (Phase 1: train GNN classifier only, LR=1e-3) from optional fine-tuning (Phase 2: unfreeze layer4 with LR=1e-5 for ResNet, LR=1e-4 for GNN) to prevent destroying pretrained features with random GNN gradients.

### Critical Pitfalls

Research identified 24 detailed pitfalls across 6 categories (ResNet extraction, graph construction, GraphSAGE configuration, training, evaluation, project management). The top 5 that will "tank the project" and top 3 that waste the most time are extracted below with prevention strategies.

1. **Over-smoothing with too many GNN layers (3.2)** — Using 4+ GraphSAGE layers causes node representations to become indistinguishable due to repeated neighbor averaging, validation accuracy degrades despite more parameters. **Prevention**: Use 2-3 layers maximum for spatial grids (7×7 has short graph diameter), add residual connections if going deeper, monitor validation performance when adding layers, stop if 3-layer model performs worse than 2-layer.

2. **Memory explosion with 320k feature maps (1.4)** — Storing ResNet layer4 outputs for full dataset requires ~128GB uncompressed (320k images × 2048 × 7 × 7 × 4 bytes), causes OOM errors during dataset loading or training. **Prevention**: Use on-the-fly feature extraction during training (trade compute for memory), OR cache to disk with memory mapping (HDF5 with compression reduces 2-3x), OR apply PCA projection to reduce 2048-d to 512-d (4x memory savings), recommended mixed approach is disk caching with DataLoader num_workers > 0 and pin_memory=True.

3. **Unfair CNN baseline comparison (5.2)** — Comparing GNN (with pretrained ResNet + tuned hyperparameters + 50 epochs) against weak CNN baseline (random init or suboptimal training) inflates perceived GNN benefit, makes results meaningless. **Prevention**: Use same pretrained ResNet-50 for baseline, apply global average pooling → linear classifier (standard approach), use identical hyperparameters (LR, optimizer, epochs, augmentation, random seed), expect realistic GNN improvement of 2-5% over strong baseline (>10% improvement suggests unfair comparison).

4. **Non-reproducible code with missing documentation (6.2)** — Hardcoded paths, missing random seeds, undocumented hyperparameters, unclear dependencies prevent teammates or TAs from reproducing the claimed results. **Prevention**: Maintain requirements.txt with exact versions, write README with installation instructions (Mac + Linux), create config.yaml with all hyperparameters, set random seeds (torch.manual_seed(42), np.random.seed(42)), save training logs with config to logs/experiment_name/, provide reproduce.sh script.

5. **Bad ablation studies that change multiple variables (5.4)** — Changing k-NN parameter, aggregation function, resolution, and fine-tuning strategy simultaneously makes it impossible to attribute performance gains to specific design choices. **Prevention**: Change one variable at a time, fix random seed across ablations, run each ablation with 3 seeds and report mean ± std, use t-test for statistical significance, required ablations are (1) CNN baseline, (2) aggregation (mean/pool/LSTM), (3) depth (1/2/3 layers), (4) resolution (7×7 vs 14×14), (5) k-NN (k=5/10/15/20).

**Time wasters:**
- **Infrastructure over-engineering (6.1)** — Spending 3+ weeks on Docker, complex data pipelines, wandb integration instead of implementing GNN models. **Prevention**: Build minimum viable infrastructure in Week 1 (load data, extract features, build k-NN graph, train 2-layer GraphSAGE-mean, compute accuracy), defer Docker/wandb/distributed training, focus 40% time on model development and 30% on experiments.
- **Data loading bottleneck with on-the-fly graph construction (4.4)** — Constructing graphs during training creates CPU bottleneck (~100ms/image graph construction vs ~50ms/image GPU forward pass), GPU utilization drops below 30%, training 3-5x slower than necessary. **Prevention**: Pre-compute and cache graphs to disk (one-time preprocessing), OR use DataLoader with num_workers=4+ and pin_memory=True for parallel construction.
- **MPS (Mac) compatibility issues (4.1)** — PyTorch Geometric dependencies (torch_scatter, torch_sparse, torch_cluster) lack pre-built Mac binaries, require building from source with Xcode/CMake/Boost, some operations fall back to CPU making MPS slower than expected. **Prevention**: Use Mac only for prototyping with small data subset (100 images), CPU is acceptable for development, do full 320k training on SUTD GPU cluster with CUDA, document installation steps for both environments in README.

## Implications for Roadmap

Based on research, suggested phase structure aligns with natural technical dependencies (data → features → graphs → GNN → training → evaluation) and course grading priorities (technical implementation 50%, performance 20%, report 30%). The critical path is sequential through phases 1-5, with phase 6-7 offering parallelization opportunities.

### Phase 1: Data Foundation & Feature Extraction
**Rationale:** All downstream components depend on data loading and CNN features. ResNet-50 feature extraction is a prerequisite for graph construction, making this the unavoidable first phase. Starting with a small subset (5-10% of 320k) enables fast iteration and debugging before scaling to full dataset.

**Delivers:**
- RVL-CDIP dataset loading from HuggingFace with proper train/val/test splits (320k/40k/40k)
- Image preprocessing pipeline (resize to 224×224, grayscale→RGB conversion, ImageNet normalization)
- ResNet-50 feature extractor (frozen) extracting layer4 output (7×7 spatial feature maps)
- Feature caching strategy to avoid repeated CNN forward passes
- Subset development workflow (32k training images for rapid experimentation)

**Addresses features from FEATURES.md:**
- Data loading pipeline (table stakes)
- Data preprocessing (table stakes)
- CNN feature extraction (table stakes)
- Subset development strategy (differentiator - good engineering practice)

**Avoids pitfalls:**
- Grayscale vs RGB channel mismatch (1.1) — implement 3-channel replication early
- Memory explosion with 320k feature maps (1.4) — design caching strategy from the start
- Wrong feature extraction layer (1.2) — validate layer4 output shape [B, 2048, 7, 7] in Phase 1

**Stack elements:** PyTorch 2.9.1, torchvision 0.25, HuggingFace datasets, Poetry

### Phase 2: Graph Construction & Baseline GNN
**Rationale:** Graph construction is the core research contribution of this project. Implementing one working graph strategy (k-NN spatial with k=6) establishes the baseline for later ablations. This phase also includes the first end-to-end GNN model to validate the full pipeline before optimization.

**Delivers:**
- Graph builder module converting feature maps [B, 2048, 7, 7] to PyG Data objects
- k-NN spatial graph construction (k=6 as starting point, using scikit-learn kneighbors_graph)
- PyG Dataset/DataLoader integration with batching
- 2-layer GraphSAGE classifier with mean aggregation and global mean pooling
- Working end-to-end pipeline: image → features → graph → GNN → predictions

**Addresses features from FEATURES.md:**
- Graph construction from feature maps (table stakes, research contribution)
- GraphSAGE implementation (table stakes)
- End-to-end pipeline (table stakes - rubric requirement with diagram)

**Avoids pitfalls:**
- Wrong k-NN parameter (2.1) — start with k=6 based on research recommendations for 49-node graphs
- Graph size explosion (2.3) — use 7×7 resolution (49 nodes) for initial implementation, defer 14×14 to ablations
- Over-smoothing (3.2) — implement 2 layers only, validate before going deeper
- Feature map resolution trade-off (2.4) — establish 7×7 baseline before exploring higher resolutions

**Stack elements:** PyTorch Geometric 2.7, scikit-learn 1.8, scipy 1.14

### Phase 3: Training Infrastructure & CNN Baseline
**Rationale:** A robust training loop with checkpointing and early stopping is required before running experiments. The CNN baseline must be implemented with the same pretrained ResNet and hyperparameters as the GNN model to ensure fair comparison (critical for evaluation integrity).

**Delivers:**
- Training loop with forward pass, CrossEntropyLoss, backward pass, optimizer step
- Validation loop computing accuracy and macro F1
- Checkpointing (save best model based on validation F1)
- Early stopping with patience=10 epochs
- Experiment tracking (TensorBoard or wandb, depending on team preference)
- Reproducibility setup (random seeds, config.yaml, documented commands)
- CNN baseline: ResNet-50 (frozen) + global average pooling + linear classifier

**Addresses features from FEATURES.md:**
- Training loop with validation (table stakes)
- CNN baseline comparison (table stakes - required to show GNN value)
- Reproducibility (seeds, config) (table stakes - 25% of coding grade)
- Code documentation (table stakes - rubric requirement)

**Avoids pitfalls:**
- Unfair CNN baseline comparison (5.2) — implement baseline with same pretrained weights, hyperparameters, training setup
- Non-reproducible code (6.2) — build reproducibility infrastructure from the start
- Frozen vs fine-tuned backbone decision (1.3) — start frozen for both CNN baseline and GNN, defer fine-tuning to Phase 5
- Learning rate scheduling (4.3) — use simple setup for Phase 3 (Adam with LR=1e-3), defer differential LR to fine-tuning phase

**Stack elements:** PyTorch optimizer, TensorBoard or wandb, tqdm

### Phase 4: Evaluation & Metrics
**Rationale:** Comprehensive evaluation beyond overall accuracy is required by the rubric (precision/recall/F1, confusion matrix) and demonstrates research depth. Per-class analysis reveals which document types benefit from graph structure vs CNN alone, forming a key insight for the report.

**Delivers:**
- Evaluation module computing per-class precision, recall, F1
- Macro F1 score (primary metric for model comparison)
- Confusion matrix visualization (16×16 heatmap with seaborn)
- Per-class performance analysis comparing GNN vs CNN baseline
- Training curves visualization (accuracy and loss over epochs)
- Failure case analysis identifying misclassified examples

**Addresses features from FEATURES.md:**
- Accuracy metric (table stakes - rubric requirement)
- Precision/Recall/F1 (table stakes - rubric requirement)
- Confusion matrix (table stakes - rubric requirement)
- Training curves visualization (table stakes - ML requirements)
- Per-class performance analysis (differentiator - deeper evaluation)
- Confusion matrix heatmap analysis (differentiator - shows failure modes)
- Failure case visualization (differentiator - rubric mentions this)

**Avoids pitfalls:**
- Only reporting overall accuracy (5.1) — implement per-class metrics from the start
- Bad ablation studies (5.4) — establish evaluation methodology before running ablations in Phase 5

**Stack elements:** scikit-learn metrics, matplotlib, seaborn

### Phase 5: Ablation Studies & Hyperparameter Tuning
**Rationale:** Systematic ablation studies demonstrate understanding of GraphSAGE mechanics and graph construction choices, directly supporting the technical implementation grade (50%) and creativity bonus (5%). This phase requires Phase 1-4 infrastructure to be complete and can run multiple experiments in parallel.

**Delivers:**
- k-NN parameter ablation: k ∈ {5, 10, 15, 20}
- GraphSAGE depth ablation: 1, 2, 3 layers (watch for over-smoothing)
- Aggregation function ablation: mean vs pool vs LSTM
- Global pooling ablation: mean vs max vs attention
- Feature map resolution ablation: layer4 (7×7) vs layer3 (14×14)
- Statistical significance testing (3 seeds per configuration, mean ± std, t-test)
- Hyperparameter tuning documentation (learning rate, batch size, optimizer choices)

**Addresses features from FEATURES.md:**
- Ablation studies (k parameter, depth, aggregation) (differentiator - systematic evaluation, creativity bonus)
- Hyperparameter tuning documentation (differentiator - shows rigor)

**Avoids pitfalls:**
- Bad ablation studies changing multiple variables (5.4) — change one variable at a time, fix random seed
- Wrong k-NN parameter (2.1) — grid search reveals optimal k for this dataset
- Wrong aggregation function (3.1) — ablation identifies best choice (pool recommended for documents)
- Over-smoothing (3.2) — depth ablation validates 2-layer is optimal, deeper hurts performance

**Research flag:** This phase has well-documented patterns for ablation studies in GNN literature, no additional research needed.

### Phase 6: Full-Scale Training & Fine-Tuning (Optional Enhancement)
**Rationale:** After establishing optimal hyperparameters on subset (Phase 5), scale to full 320k training set on SUTD GPU cluster. Optionally fine-tune ResNet-50 layer4 with very low learning rate (LR=1e-5) to adapt ImageNet features to grayscale document domain. This phase is optional if time is limited—frozen baseline may be sufficient.

**Delivers:**
- Training on full RVL-CDIP training set (320k images) using SUTD GPU cluster
- Optional fine-tuning: unfreeze ResNet layer4 with differential learning rates (LR=1e-5 for ResNet, LR=1e-4 for GNN)
- Final model checkpoints with best validation performance
- Computational efficiency analysis (training time, memory usage, inference speed)
- Out-of-distribution evaluation on RVL-CDIP-N (noisy documents)

**Addresses features from FEATURES.md:**
- Out-of-distribution evaluation (RVL-CDIP-N) (differentiator - tests generalization)
- Computational efficiency analysis (differentiator - shows practical thinking)

**Avoids pitfalls:**
- Frozen vs fine-tuned decision (1.3) — fine-tune only after frozen model converges, use very low LR for ResNet
- Learning rate scheduling (4.3) — implement differential LR for CNN vs GNN
- Data loading bottleneck (4.4) — use pre-computed cached graphs from Phase 2, num_workers > 0
- MPS compatibility issues (4.1) — train on SUTD GPU cluster with CUDA, not Mac MPS

**Research flag:** Fine-tuning strategies are well-documented for transfer learning, no additional research needed.

### Phase 7: Visualization & Report (Parallel with Phase 5-6)
**Rationale:** Presentation and report account for 50% of the grade (20% presentation, 30% report). This phase can proceed in parallel with ablation studies and full-scale training, as most content draws from completed phases 1-4 plus interim results from phase 5.

**Delivers:**
- Graph visualization for sample documents (NetworkX or PyG visualization)
- Attention/importance visualization if using attention pooling
- Comprehensive results tables (per-class metrics, ablation study results)
- Final presentation slides and demo
- Final written report with methodology, results, analysis, conclusions

**Addresses features from FEATURES.md:**
- Graph visualization (differentiator - interpretability, creativity bonus)
- Attention/importance weights (differentiator - advanced interpretability, LOW priority due to complexity)

**Avoids pitfalls:**
- Infrastructure over-engineering (6.1) — focus on report content, not fancy demos
- Not documenting for reproducibility (6.2) — ensure README and report have complete reproduction instructions

**Research flag:** Visualization techniques for GNNs are standard, no additional research needed.

### Phase Ordering Rationale

**Sequential dependencies (cannot parallelize):**
- Phase 1 → Phase 2: Graph construction requires feature maps from ResNet
- Phase 2 → Phase 3: Training requires GNN model and graph dataset
- Phase 3 → Phase 4: Evaluation requires trained models (baseline and GNN)
- Phase 4 → Phase 5: Ablations require established evaluation methodology

**Parallelization opportunities:**
- Phase 7 (Visualization & Report) can start after Phase 4 completes, running in parallel with Phase 5-6
- Multiple ablation experiments within Phase 5 can run in parallel (k-NN ablation, depth ablation, aggregation ablation are independent)
- Phase 6 (Fine-tuning) is optional and can be deferred if time-constrained

**Pitfall avoidance through ordering:**
- Building reproducibility infrastructure in Phase 3 (before ablations) prevents the "can't reproduce results" crisis in Phase 7
- Implementing CNN baseline in Phase 3 (before full GNN experiments) ensures fair comparison from the start
- Starting with 7×7 resolution in Phase 2 (deferring 14×14 to Phase 5) avoids memory/compute issues during initial development
- Frozen ResNet in Phase 1-5 with optional fine-tuning in Phase 6 prevents destroying pretrained features early

**Time allocation (4-6 week timeline):**
- Week 1: Phase 1 (Data Foundation & Feature Extraction)
- Week 2: Phase 2 (Graph Construction & Baseline GNN)
- Week 3: Phase 3 (Training Infrastructure & CNN Baseline)
- Week 4: Phase 4 (Evaluation & Metrics) + start Phase 7 (Report background sections)
- Week 5: Phase 5 (Ablation Studies) + Phase 7 (Results & Analysis)
- Week 6: Phase 6 (Full-Scale Training, optional) + Phase 7 (Final Report & Presentation)

### Research Flags

**Phases likely needing deeper research during planning:**
- **Phase 2 (Graph Construction)**: Core research contribution with no standardized approach for document images. Multiple strategies exist (k-NN spatial, k-NN feature space, grid connectivity, region adjacency) with unclear performance trade-offs. Will require experimentation and potentially additional literature review during implementation. Consider invoking `/gsd:research-phase` for graph construction strategy selection.

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (Data & Features)**: Well-documented PyTorch/torchvision/HuggingFace patterns, ResNet feature extraction is standard transfer learning
- **Phase 3 (Training Infrastructure)**: Standard PyTorch training loop, checkpointing, early stopping patterns
- **Phase 4 (Evaluation)**: Standard scikit-learn metrics, confusion matrix visualization
- **Phase 5 (Ablation Studies)**: Well-established ablation methodology in GNN literature (GraphSAGE paper, recent 2024-2026 ablation studies)
- **Phase 6 (Fine-Tuning)**: Standard transfer learning fine-tuning strategies, differential learning rate patterns
- **Phase 7 (Visualization)**: Standard matplotlib/seaborn plotting, NetworkX graph visualization

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All technologies verified via Context7 official docs, PyTorch Geometric installation matrix, and PyPI version compatibility. Mac MPS support confirmed with caveat for macOS 26.0. CUDA support for SUTD cluster confirmed. |
| Features | MEDIUM-HIGH | Table stakes features clearly defined by rubric with explicit grading criteria. Differentiator features based on academic GNN best practices (ablation studies standard in 2024-2026 papers). Lower confidence on attention/importance weights (GraphSAGE doesn't have native attention like GAT, would require custom implementation). |
| Architecture | MEDIUM-HIGH | Component boundaries and data flow verified with PyG documentation and multiple academic implementations (Doc-GCN, Vision GNN papers). Patch-based graph construction is well-established pattern. Lower confidence on optimal hyperparameters (k, depth, aggregation) which require experimentation on RVL-CDIP specifically. |
| Pitfalls | MEDIUM | Critical pitfalls (over-smoothing, memory issues, unfair baselines) verified across multiple sources including official papers, blog posts, and GitHub issues. Mac MPS issues based on GitHub issue reports (medium confidence, not official docs). Time-wasting pitfalls based on general academic project experience patterns. |

**Overall confidence:** MEDIUM-HIGH

Research is comprehensive for stack selection, feature requirements, and architecture patterns. Confidence is reduced by three factors: (1) graph construction strategy for document images is a research variable with no standard "best practice", (2) RVL-CDIP-specific hyperparameters (optimal k, depth, aggregation) require empirical validation, and (3) Mac MPS compatibility issues are based on community reports rather than official documentation.

### Gaps to Address

**Gap 1: Graph construction strategy selection**
- **Issue**: Multiple valid approaches (k-NN spatial, k-NN feature space, grid connectivity, region adjacency) with unclear performance trade-offs for document images specifically. Literature shows these strategies for general image classification but not document-specific benchmarks.
- **Resolution**: Implement k-NN spatial with k=6 as baseline (Phase 2), then ablate to k-NN feature space in Phase 5. Document rationale in report. Accept that this is a research variable requiring experimentation. Consider this the core contribution of the project.
- **Risk**: Medium. If k-NN spatial underperforms, may need to pivot to alternative strategy, but Phase 5 ablations provide opportunity to explore.

**Gap 2: Optimal ResNet-50 layer for feature extraction**
- **Issue**: Research suggests layer4 (7×7) for semantic features with spatial structure, but layer3 (14×14) offers higher resolution. Trade-off between spatial granularity and computational cost is dataset-dependent.
- **Resolution**: Start with layer4 (Phase 1-4), ablate to layer3 in Phase 5. If layer3 doesn't improve >2% over layer4, stick with layer4 for computational efficiency.
- **Risk**: Low. Layer4 is safe default; layer3 is optional enhancement tested in ablations.

**Gap 3: Fine-tuning necessity for grayscale document domain**
- **Issue**: ImageNet pretrained on RGB natural images may have domain gap with grayscale documents. Unclear if fine-tuning ResNet-50 on RVL-CDIP improves performance enough to justify computational cost.
- **Resolution**: Start frozen (Phase 1-5) to establish baseline, optionally fine-tune in Phase 6 if time permits. Compare frozen vs fine-tuned performance. If improvement <2%, document in report but don't deploy fine-tuned model.
- **Risk**: Low. Frozen baseline is acceptable for course project; fine-tuning is optional enhancement.

**Gap 4: Mac MPS vs SUTD cluster allocation**
- **Issue**: Team may attempt local development on Mac MPS despite known compatibility issues (PyG dependencies require source build, some ops fall back to CPU). Full 320k training on Mac MPS is infeasible (slow, memory constraints).
- **Resolution**: Use Mac only for prototyping with 32k subset (10% of data). Plan SUTD GPU cluster access for Phase 5-6 (ablations and full-scale training). Document installation for both environments in README.
- **Risk**: Medium. If SUTD cluster unavailable or has queue delays, may need to reduce dataset size or extend timeline.

**Gap 5: RVL-CDIP-N dataset availability**
- **Issue**: Research mentions RVL-CDIP-N (noisy version) for out-of-distribution evaluation, but HuggingFace availability unclear. May not be publicly available or may require custom preprocessing.
- **Resolution**: Defer to Phase 6 (optional). Check HuggingFace Hub for `rvl_cdip_n` during Phase 4. If unavailable, document as limitation in report ("would test on RVL-CDIP-N if available"). Not critical for passing grade.
- **Risk**: Low. RVL-CDIP-N is nice-to-have for robustness evaluation, not required by rubric.

## Sources

### Primary (HIGH confidence)
- **Context7: PyTorch Geometric Installation** — installation matrix, version compatibility, MPS support verification
- **Context7: PyTorch MPS Backend** — MPS availability requirements, macOS version compatibility
- **Context7: HuggingFace Datasets** — RVL-CDIP dataset structure, loading API, streaming support
- **Context7: PyTorch Geometric Tutorial** — k-NN graph construction, Dataset/DataLoader patterns
- **GraphSAGE Original Paper (Hamilton et al. 2017, Stanford)** — aggregation functions, neighbor sampling, performance benchmarks
- **PyTorch Geometric GitHub Releases** — version changelog, compatibility notes
- **torchvision Models Documentation** — ResNet-50 architecture, layer structure, pretrained weights API
- **scikit-learn 1.8.0 Documentation** — kneighbors_graph API, ball-tree algorithm, sparse matrix integration

### Secondary (MEDIUM confidence)
- **HuggingFace Hub: aharley/rvl_cdip** — dataset card, splits, class distribution
- **Doc-GCN Paper (Luo et al. 2022, ACL)** — document graph construction patterns
- **Vision GNN Paper (2022, arXiv)** — patch-based node representation for images
- **On Evaluation of Document Classification with RVL-CDIP (arXiv 2023)** — dataset limitations (8.1% label noise), evaluation best practices
- **Dynamic Backbone Freezing Paper (2024, arXiv)** — fine-tuning strategies, dynamic freezing schedules
- **Python Packaging Best Practices 2026 (dasroot.net)** — Poetry vs pip vs uv comparison
- **WandB vs TensorBoard Comparison (neptune.ai)** — experiment tracking pros/cons
- **PyTorch MPS Setup Guide (tillcode.com)** — Mac installation, performance expectations
- **tqdm for PyTorch Training (adamoudad.github.io)** — progress bar integration patterns

### Tertiary (LOW confidence, needs validation)
- **GitHub Issue #167679** — macOS 26 MPS availability issue (community report, not official)
- **WandB Reliability Discussion (neptune.ai blog)** — upload blocking training process (anecdotal)
- **ResNet Feature Extraction Layers (Medium post)** — layer output shapes, semantic level trade-offs
- **GraphSAGE Aggregation Comparison (emergentmind.com)** — LSTM vs pool vs mean performance (not RVL-CDIP specific)
- **GNN Ablation Studies in Blockchain/Bridge Detection (arXiv 2026, Nature 2024-2025)** — general ablation methodology patterns (not document classification)

---
*Research completed: 2026-03-25*
*Ready for roadmap: yes*
