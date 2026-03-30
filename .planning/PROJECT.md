# GraphSAGE Document Image Classification

## What This Is

A GraphSAGE-based model for classifying scanned document images into 16 categories (letter, form, email, invoice, etc.) on the RVL-CDIP dataset. Part of a team project for 61.502 Deep Learning for Enterprise (SUTD, Y2026) where each member builds a different GNN variant. This repo covers the GraphSAGE implementation, graph construction from CNN feature maps, and ablation studies.

## Core Value

The GraphSAGE model must demonstrate whether graph-based spatial reasoning improves document classification over CNN-only baselines — with clear, reproducible evidence.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Fine-tune ResNet-50 as feature extractor for document images
- [ ] Design and implement graph construction from CNN spatial feature maps
- [ ] Implement GraphSAGE for graph-level document classification
- [ ] Run ablation studies (k-NN connectivity, network depth)
- [ ] Evaluate with accuracy, per-class precision/recall/F1, macro F1, confusion matrix
- [ ] Compare GraphSAGE results against CNN baseline (from reference notebook)
- [ ] Train on RVL-CDIP (320k train / 40k val / 40k test)
- [ ] Evaluate on RVL-CDIP-N (1k out-of-distribution color documents)
- [ ] Build end-to-end pipeline (data loading -> feature extraction -> graph construction -> GNN -> classification)
- [ ] Produce reproducible code with documented pipeline

### Out of Scope

- GCN implementation — teammate (Seow Chun Yong) owns this
- GAT implementation — teammate (Prathosh Chander) owns this
- Standalone CNN baseline training — reference notebook covers this
- Streamlit/FastAPI demo — deferred, only if time permits
- Final report structure — TBD with team
- Presentation slides — TBD with team

## Context

**Course**: 61.502 Deep Learning for Enterprise, SUTD Y2026
**Team**: Seow Chun Yong, Prathosh Chander, Sagar Pratap Singh
**Deadline**: April 17th, 11:59pm

**Dataset**:
- RVL-CDIP: 400k grayscale document images, 16 classes, pre-split 320k/40k/40k (HuggingFace)
- RVL-CDIP-N: 1k modern born-digital color documents, same 16 classes (HuggingFace)

**Reference material**:
- `reference/RVL-CDIP_ResNet50.ipynb` — CNN baseline notebook (ResNet-50 on RVL-CDIP)
- `docs/EDL-Group1-Project-Proposal.docx` — Full project proposal
- `docs/Project Instructions-updated Rubrics.pdf` — Grading rubrics

**Graph construction is a research piece**: The approach for converting CNN feature maps to graph structure (patch strategy, k-NN parameter, edge definition) is not predetermined — experimenting with this is part of the project's contribution.

**Grading breakdown** (full team project):
- Technical Implementation: 50% (Concept 5%, Coding 25%, Performance & Evaluation 20%)
- Creativity & Innovation: 5% bonus
- Presentation: 20%
- Report: 30%

## Constraints

- **Compute**: Local Mac (MPS) for development/debugging, SUTD AI Mega Cluster (GPU) for full training
- **Data strategy**: Develop on subset first (5-10%), scale to full 320k for final training
- **Package manager**: Poetry (per personal standards)
- **Timeline**: Submission April 17th
- **Reproducibility**: Code must be reproducible with documented seeds, commands, and hardware notes (rubric requirement)
- **Framework**: PyTorch + PyTorch Geometric (per proposal references)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| GraphSAGE as personal focus | Team divided GNN variants: one per member | — Pending |
| Graph construction as experimentation | Not predetermined — finding the right approach is part of the contribution | — Pending |
| Subset-first development | Iterate on architecture locally before full training on cluster | — Pending |
| CNN baseline from reference notebook | Avoid duplicating work, use existing baseline for comparison | — Pending |

---
*Last updated: 2026-03-25 after initialization*
