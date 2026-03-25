# Roadmap: GraphSAGE Document Image Classification

## Overview

This roadmap delivers a complete GraphSAGE-based document classification system on RVL-CDIP, structured as an incrementally-built Jupyter notebook across four phases: building notebook sections for data loading and feature extraction, adding graph construction and model definitions, extending with training and evaluation infrastructure, and completing with ablation studies and final analysis.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Notebook Foundation - Data & Features** - Notebook sections: imports, data loading, preprocessing, feature extraction, caching
- [ ] **Phase 2: Notebook Extension - Graph & Models** - Notebook sections: graph construction, GraphSAGE model, CNN baseline, end-to-end forward pass
- [ ] **Phase 3: Notebook Completion - Training & Evaluation** - Notebook sections: training loop, evaluation metrics, comparison plots, failure analysis
- [ ] **Phase 4: Notebook Finalization - Ablation Studies** - Notebook sections: ablation experiments, results tables, RVL-CDIP-N evaluation

## Phase Details

### Phase 1: Notebook Foundation - Data & Features
**Goal**: Build notebook sections for data pipeline and CNN feature extraction that run end-to-end and cache features to disk
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, MDL-01
**Success Criteria** (what must be TRUE):
  1. Notebook section "Setup & Imports" runs and loads all required libraries (torch, torchvision, PyG, HuggingFace datasets)
  2. Notebook section "Load RVL-CDIP" runs and displays correct split sizes (train/val/test: 320k/40k/40k)
  3. Notebook section "Preprocessing Pipeline" runs and shows sample preprocessed images (grayscale → RGB, 224×224, normalized)
  4. Notebook section "ResNet-50 Feature Extraction" runs and outputs feature maps with shape [B, 2048, 7, 7] for sample batch
  5. Notebook section "Feature Caching" runs and saves features to disk, then reloads them successfully
  6. Notebook runs end-to-end from top to feature extraction and produces cached feature files on disk
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md — Project setup, dependencies, config module, data loading and preprocessing
- [ ] 01-02-PLAN.md — ResNet-50 feature extraction, caching, notebook cell snippets

### Phase 2: Notebook Extension - Graph & Models
**Goal**: Add notebook sections for graph construction, GraphSAGE model definition, and end-to-end forward pass verification
**Depends on**: Phase 1
**Requirements**: MDL-02, MDL-03, MDL-04
**Note**: MDL-04 (CNN baseline) is already implemented in `reference/RVL-CDIP_ResNet50.ipynb`. This phase references those results for comparison rather than rebuilding.
**Success Criteria** (what must be TRUE):
  1. Notebook section "Graph Construction" runs and converts feature maps to PyG Data objects, displaying graph statistics (49 nodes, k*49 edges per image)
  2. Notebook section "GraphSAGE Model Definition" runs and displays model architecture summary (layer dimensions, parameters)
  3. Notebook section "Forward Pass Test" runs GraphSAGE on sample batch and outputs logits [B, 16]
  4. Notebook runs end-to-end from top through model forward pass and produces class predictions
**Plans**: TBD

Plans:
- [ ] TBD after planning

### Phase 3: Notebook Completion - Training & Evaluation
**Goal**: Add notebook sections for training loop, evaluation metrics, comparison analysis, and comprehensive visualizations
**Depends on**: Phase 2
**Requirements**: TRN-01, TRN-02, TRN-03, TRN-04, EVL-01, EVL-02, EVL-03, EVL-04
**Success Criteria** (what must be TRUE):
  1. Notebook section "Training Configuration" runs and displays hyperparameters, device (MPS/CUDA/CPU), and reproducibility setup
  2. Notebook section "Training Loop" runs and trains GraphSAGE with progress bars, validation checks, early stopping
  3. Notebook section "Training Curves" runs and displays loss/accuracy plots over epochs
  4. Notebook section "Evaluation Metrics" runs and displays accuracy, per-class F1, macro F1, and 16x16 confusion matrix heatmap
  5. Notebook section "GNN vs CNN Comparison" runs and compares GraphSAGE results against CNN baseline from reference notebook (per-class breakdown)
  6. Notebook section "Failure Analysis" runs and displays misclassified examples with predicted/actual labels
  7. Notebook runs end-to-end from top through complete training and evaluation, producing trained model checkpoint and all visualizations
**Plans**: TBD

Plans:
- [ ] TBD after planning

### Phase 4: Notebook Finalization - Ablation Studies
**Goal**: Add notebook sections for systematic ablation experiments and final out-of-distribution evaluation
**Depends on**: Phase 3
**Requirements**: ABL-01, ABL-02, ABL-03, ABL-04
**Success Criteria** (what must be TRUE):
  1. Notebook section "k-NN Ablation" runs experiments for k={4,6,8,10} and displays results table + plot
  2. Notebook section "Depth Ablation" runs experiments for {1,2,3} GraphSAGE layers and displays over-smoothing analysis
  3. Notebook section "Aggregation Ablation" runs experiments for mean/max/pool aggregation and displays comparison
  4. Notebook section "RVL-CDIP-N Evaluation" runs model on 1k color documents and displays OOD performance metrics
  5. Notebook section "Final Summary" displays consolidated results table comparing all ablations
  6. Notebook runs end-to-end from top through all ablations and produces complete experimental results
**Plans**: TBD

Plans:
- [ ] TBD after planning

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Notebook Foundation - Data & Features | 0/2 | Not started | - |
| 2. Notebook Extension - Graph & Models | 0/TBD | Not started | - |
| 3. Notebook Completion - Training & Evaluation | 0/TBD | Not started | - |
| 4. Notebook Finalization - Ablation Studies | 0/TBD | Not started | - |
