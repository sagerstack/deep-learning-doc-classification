# Requirements: GraphSAGE Document Image Classification

**Defined:** 2026-03-25
**Core Value:** The GraphSAGE model must demonstrate whether graph-based spatial reasoning improves document classification over CNN-only baselines — with clear, reproducible evidence.

## v1 Requirements

### Data Pipeline

- [ ] **DATA-01**: Load RVL-CDIP from HuggingFace with predefined train/val/test splits
- [ ] **DATA-02**: Configurable subset sampling (sample mode for local dev, full mode for cluster)
- [ ] **DATA-03**: Grayscale-to-RGB preprocessing for ResNet-50 compatibility
- [ ] **DATA-04**: Cache extracted CNN feature maps to disk to avoid recomputation

### Model

- [ ] **MDL-01**: ResNet-50 feature extractor (frozen, layer4 output preserving 7x7 spatial structure)
- [ ] **MDL-02**: k-NN spatial graph construction from feature map patches (49 nodes per image)
- [ ] **MDL-03**: GraphSAGE classifier (message passing + global pooling + classification head)
- [ ] **MDL-04**: CNN baseline (ResNet-50 + global avg pool + classifier) for fair comparison

### Training

- [ ] **TRN-01**: Training loop with validation, early stopping, and model checkpointing
- [ ] **TRN-02**: Device-agnostic execution (MPS for Mac, CUDA for cluster, CPU fallback)
- [ ] **TRN-03**: Reproducibility setup (fixed seeds, config file for hyperparameters, documented commands)
- [ ] **TRN-04**: Experiment logging (wandb or TensorBoard integration)

### Evaluation

- [ ] **EVL-01**: Accuracy, per-class precision/recall/F1, macro F1, and confusion matrix
- [ ] **EVL-02**: Training/validation loss and accuracy curves over epochs
- [ ] **EVL-03**: GNN vs CNN per-class comparison analysis (which document types benefit from graph structure)
- [ ] **EVL-04**: Failure case analysis with discussion of misclassifications

### Ablation Studies

- [ ] **ABL-01**: k-NN connectivity ablation (vary k parameter, e.g., k=4,6,8,10)
- [ ] **ABL-02**: Network depth ablation (vary GraphSAGE layers 1,2,3 — measure over-smoothing)
- [ ] **ABL-03**: Aggregation function comparison (mean vs max vs pool in GraphSAGE)
- [ ] **ABL-04**: Out-of-distribution evaluation on RVL-CDIP-N (1k color documents)

## v2 Requirements

### Model Enhancements

- **MDL-05**: ResNet-50 fine-tuning with differential learning rates (CNN 1e-5, GNN 1e-3)
- **MDL-06**: Attention-based global pooling (instead of mean pooling)
- **MDL-07**: Multiple graph construction strategies comparison (spatial k-NN vs feature-space k-NN vs grid)

### Deployment

- **DEP-01**: Streamlit or FastAPI demo for document classification inference
- **DEP-02**: Model weights hosted on Google Drive/Dropbox

## Out of Scope

| Feature | Reason |
|---------|--------|
| GCN implementation | Teammate (Seow Chun Yong) owns this |
| GAT implementation | Teammate (Prathosh Chander) owns this |
| Custom data augmentation | Not the project's research focus |
| MLOps / deployment infrastructure | Academic project, not production |
| Real-time serving optimization | Out of scope for course project |
| Report writing | TBD with team, not tracked here |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | — | Pending |
| DATA-02 | — | Pending |
| DATA-03 | — | Pending |
| DATA-04 | — | Pending |
| MDL-01 | — | Pending |
| MDL-02 | — | Pending |
| MDL-03 | — | Pending |
| MDL-04 | — | Pending |
| TRN-01 | — | Pending |
| TRN-02 | — | Pending |
| TRN-03 | — | Pending |
| TRN-04 | — | Pending |
| EVL-01 | — | Pending |
| EVL-02 | — | Pending |
| EVL-03 | — | Pending |
| EVL-04 | — | Pending |
| ABL-01 | — | Pending |
| ABL-02 | — | Pending |
| ABL-03 | — | Pending |
| ABL-04 | — | Pending |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 0
- Unmapped: 20

---
*Requirements defined: 2026-03-25*
*Last updated: 2026-03-25 after initial definition*
