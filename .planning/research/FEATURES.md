# Feature Landscape: GNN-Based Document Image Classification

**Domain:** Academic deep learning course project - GraphSAGE document classification
**Researched:** 2026-03-25
**Context:** Team project for 61.502 Deep Learning for Enterprise (SUTD). GraphSAGE variant implementation with grading focus on Technical Implementation (50%), Presentation (20%), Report (30%), Creativity (5% bonus).

## Table Stakes

Features required to meet course requirements. Missing = marks lost.

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| Data loading pipeline | Required for reproducibility (Coding 25%) | Low | HuggingFace datasets | Must handle RVL-CDIP 320k/40k/40k splits |
| Data preprocessing | Standard ML pipeline component | Low | PIL/torchvision | Grayscale normalization, resizing to ResNet input |
| CNN feature extraction | Core architecture component | Medium | torchvision ResNet-50, pretrained weights | Fine-tuning on grayscale documents (domain shift from ImageNet) |
| Graph construction from feature maps | Core innovation of project | High | PyTorch Geometric | This IS the research contribution - not predetermined |
| GraphSAGE implementation | Project requirement (GNN variant assigned) | Medium | PyTorch Geometric | K-hop sampling, aggregation functions (mean/pool/LSTM) |
| Training loop with validation | Required for Performance & Evaluation (20%) | Low | PyTorch standard | Epoch-based training with validation metrics |
| Accuracy metric | Rubric explicitly requires this | Low | sklearn.metrics | Overall test accuracy |
| Precision/Recall/F1 | Rubric explicitly requires these | Low | sklearn.metrics | Per-class and macro-averaged |
| Confusion matrix | Required for error analysis | Low | sklearn.metrics, matplotlib/seaborn | 16x16 for RVL-CDIP classes |
| CNN baseline comparison | Required to show value of GNN approach | Medium | Reference notebook | Use existing ResNet-50 baseline from reference/RVL-CDIP_ResNet50.ipynb |
| Reproducibility (seeds, config) | Rubric: "Reproducibility of code" (Coding 25%) | Low | PyTorch manual_seed, config file | Document exact commands, seeds, hardware used |
| Code documentation | Rubric: "thoroughly documented code" (Coding 25%) | Low | Docstrings, README | Installation, usage instructions, module-level docs |
| End-to-end pipeline | Rubric requirement with diagram | Medium | All above | Data → CNN → Graph → GNN → Prediction |
| Training curves visualization | ML Requirements: accuracy/loss curves | Low | matplotlib | Train/val accuracy and loss over epochs |

## Differentiators

Features that earn bonus marks or strengthen Technical Implementation score.

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| Multiple graph construction strategies | Shows research depth (Creativity 5%) | High | Multiple k-NN implementations | Compare k-NN variants, spatial grid, region adjacency |
| Ablation studies (k parameter) | Shows systematic evaluation (Performance 20%) | Medium | Multiple training runs | Vary k in k-NN graph construction (e.g., k=3,5,7,9) |
| Ablation studies (depth) | Shows understanding of GNN mechanics | Medium | Multiple model configs | Vary GraphSAGE layers (1,2,3 layers) |
| Ablation studies (aggregation) | Shows GraphSAGE-specific expertise | Medium | GraphSAGE config | Compare mean vs pool vs LSTM aggregators |
| Per-class performance analysis | Deeper than macro metrics (Performance 20%) | Low | sklearn classification_report | Show which document types benefit from graph structure |
| Confusion matrix heatmap analysis | Shows failure mode understanding | Low | seaborn | Identify systematically confused class pairs (e.g., letter vs email) |
| Out-of-distribution evaluation (RVL-CDIP-N) | Tests generalization (Performance 20%) | Low | HuggingFace dataset | 1k color documents vs grayscale training - domain shift test |
| Failure case visualization | Rubric: "Show model failure cases" | Medium | Visualization code | Show misclassified examples with explanations |
| Graph visualization | Shows interpretability (Creativity 5%) | Medium | NetworkX, matplotlib | Visualize constructed graphs for sample documents |
| Attention/importance weights | Advanced interpretability (Creativity 5%) | High | Custom GraphSAGE hooks | Show which graph regions influence predictions (LOW confidence - GraphSAGE doesn't have native attention like GAT) |
| Hyperparameter tuning documentation | Shows rigor (Performance 20%) | Low | Logging/tracking | Document learning rate, batch size, optimizer choices |
| Computational efficiency analysis | Shows practical thinking | Low | Time tracking | Report training time, memory usage, inference speed |
| Subset development strategy | Shows good engineering (Coding 25%) | Low | Data sampling | Document 5-10% subset used for local iteration before full training |

## Anti-Features

Features to explicitly NOT build. Common mistakes or scope creep for a course project.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Over-engineered deployment infrastructure | Not evaluated, wastes time before core work done | Simple batch inference script or optional lightweight FastAPI demo (if time permits) |
| Real-time serving optimizations | Not a rubric requirement, premature optimization | Mention latency monitoring in report "Future Work" section |
| Custom data augmentation research | Not the core contribution, distracts from graph construction | Use standard transforms (resize, normalize) only |
| Multiple CNN backbones comparison | CNN is means, not end - graph is the contribution | Stick to ResNet-50 (aligns with reference baseline) |
| Distributed training infrastructure | Overkill for 320k images, unnecessary complexity | Single-GPU training on SUTD cluster is sufficient |
| Novel GNN architecture design | Project is about applying GraphSAGE, not inventing new GNN | Implement standard GraphSAGE as specified |
| Extensive related work literature review | Report section, not implementation effort | Brief related work in report, focus on implementation |
| Production-grade MLOps pipeline | No deployment requirement, academic project | Track experiments with simple logging, not full MLOps stack |
| Custom evaluation metrics | Standard metrics required by rubric | Use sklearn metrics: accuracy, precision, recall, F1, confusion matrix |
| Handling label noise in RVL-CDIP | Known issue (8.1% noise) but not project scope | Acknowledge in report limitations, don't attempt to fix dataset |
| Multi-label classification extensions | RVL-CDIP is single-label (16 classes) | Stick to single-label classification as per dataset |
| Advanced graph neural network theory | Implementation project, not theory research | Demonstrate understanding through good implementation and ablations |

## Feature Dependencies

Critical path for implementation:

```
Data Loading Pipeline
    ↓
Data Preprocessing (Resize, Normalize)
    ↓
CNN Feature Extraction (ResNet-50 fine-tuning)
    ↓
Graph Construction (CORE CONTRIBUTION - multiple strategies)
    ↓
GraphSAGE Implementation (k-hop, aggregation)
    ↓
Training Loop + Validation
    ↓
Evaluation Metrics (Accuracy, P/R/F1, Confusion Matrix)
    ↓
CNN Baseline Comparison
    ↓
Ablation Studies (k, depth, aggregation) [PARALLEL after base model works]
    ↓
Out-of-Distribution Evaluation (RVL-CDIP-N)
    ↓
Visualization + Analysis (Failure cases, per-class analysis)
```

**Blocking dependencies:**
- Graph construction blocks GraphSAGE (can't train GNN without graphs)
- CNN feature extraction blocks graph construction (graphs built from feature maps)
- Base model blocks ablation studies (need working baseline to vary)
- Training blocks all evaluation (need trained model first)

**Parallel work opportunities:**
- Documentation can proceed alongside implementation
- Visualization code can be developed with dummy data
- Report writing (background, related work) can start early
- Ablation study design can be planned while base model trains

## MVP Recommendation

For minimum viable submission (meets all rubric requirements):

### Phase 1: Core Pipeline (Table Stakes)
1. Data loading (RVL-CDIP train/val/test splits)
2. ResNet-50 feature extractor (fine-tuned or frozen - experiment)
3. ONE graph construction strategy (k-NN with k=5 as starting point)
4. GraphSAGE model (2 layers, mean aggregator as baseline)
5. Training loop with validation
6. Evaluation: Accuracy, P/R/F1, Confusion Matrix
7. CNN baseline comparison (use reference notebook results)
8. Reproducibility: Seeds, config file, documented commands

### Phase 2: Enhanced Evaluation (Strengthens Performance & Evaluation score)
9. Per-class analysis (which document types benefit from graph?)
10. Confusion matrix heatmap with interpretation
11. Training curves visualization
12. Failure case analysis with visualizations
13. Out-of-distribution evaluation on RVL-CDIP-N

### Phase 3: Differentiators (Targets Creativity bonus + higher Technical Implementation)
14. Ablation study: k parameter (k=3,5,7,9)
15. Ablation study: network depth (1,2,3 layers)
16. Ablation study: aggregation function (mean, pool, LSTM)
17. Graph visualization for sample documents
18. Computational efficiency analysis

**Defer to post-MVP (only if time permits):**
- Multiple graph construction strategies comparison (spatial grid, region adjacency)
- Attention/importance weights visualization (HIGH complexity, LOW priority)
- Streamlit/FastAPI demo (out of scope per PROJECT.md)

## Complexity Assessment

| Complexity Level | Features | Estimated Effort | Risk |
|------------------|----------|------------------|------|
| Low | Data loading, preprocessing, metrics, visualization, config | 2-3 days | Low - standard patterns |
| Medium | CNN fine-tuning, GraphSAGE implementation, training loop, baseline comparison | 5-7 days | Medium - domain shift (grayscale), debugging GNN |
| High | Graph construction strategies, multiple ablations, interpretability | 7-10 days | High - research component, may not improve results |

**Critical path bottleneck:** Graph construction (High complexity, research-oriented, core contribution). Allocate significant time here.

## Grading Alignment

| Rubric Category | Key Features | Percentage | Strategy |
|-----------------|--------------|------------|----------|
| Technical Implementation - Concept [5%] | GraphSAGE understanding shown via aggregation ablations, depth experiments | 5% | Document design choices in code and report |
| Technical Implementation - Coding [25%] | Reproducibility, documentation, modularity, end-to-end pipeline | 25% | Seeds, config files, clear README, docstrings |
| Technical Implementation - Performance [20%] | Hyperparameter tuning, CNN baseline comparison, ablation studies, metrics | 20% | Systematic ablations, multiple metrics, failure analysis |
| Creativity & Innovation [5% bonus] | Novel graph construction strategies, interpretability, efficiency analysis | 5% | Multiple graph strategies, visualizations |
| Presentation [20%] | (Not feature-based - presentation skill) | 20% | Clear demo, effective communication |
| Report [30%] | (Not feature-based - writing quality) | 30% | Document all features in report with figures |

**Feature prioritization for maximum marks:**
1. MUST HAVE (Core 50%): Data pipeline, CNN, graph construction (ONE strategy), GraphSAGE, training, metrics, baseline comparison, reproducibility
2. STRONGLY RECOMMENDED (Performance 20%): Ablation studies (k, depth, aggregation), per-class analysis, failure cases, RVL-CDIP-N evaluation
3. BONUS (Creativity 5%): Multiple graph strategies, graph visualization, interpretability

## Known Dataset Challenges (from Research)

**RVL-CDIP limitations (acknowledged in literature, February 2026):**
- Label noise: ~8.1% (range 1.6% to 16.9% per class)
- Ambiguous/multi-label documents present
- Test/train overlap (inflates metrics)
- Contains PII (Social Security numbers)
- Limited scope and diversity

**Implication for project:**
- Acknowledge in report limitations section
- Do NOT attempt to fix dataset (out of scope)
- Use RVL-CDIP-N for out-of-distribution evaluation (tests generalization beyond noisy training set)
- Per-class analysis may reveal which classes are more affected by noise

## Sources

### GraphSAGE and GNN
- [Inductive Representation Learning on Large Graphs (Stanford)](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)
- [GraphSAGE for Classification in Python](https://antonsruberts.github.io/graph/graphsage/)
- [A Comprehensive Case-Study of GraphSAGE (Towards Data Science)](https://towardsdatascience.com/a-comprehensive-case-study-of-graphsage-algorithm-with-hands-on-experience-using-pytorchgeometric-6fc631ab1067/)
- [GraphSAGE: Inductive Representation Learning on Graphs (SERP AI)](https://serp.ai/posts/graphsage/)
- [Common Aggregation Functions (APXML)](https://apxml.com/courses/introduction-to-graph-neural-networks/chapter-2-the-message-passing-mechanism/common-aggregation-functions)

### RVL-CDIP Dataset and Evaluation
- [On Evaluation of Document Classification using RVL-CDIP (arXiv)](https://arxiv.org/abs/2306.12550)
- [Evaluating Out-of-Distribution Performance on Document Image Classifiers (OpenReview)](https://openreview.net/forum?id=uDlkiCI5N7Y)
- [RVL-CDIP Dataset (HuggingFace)](https://huggingface.co/datasets/aharley/rvl_cdip)

### Evaluation Metrics
- [Document Classification: End-to-End ML Workflow (Label Your Data)](https://labelyourdata.com/articles/document-classification)
- [Deep Learning Course - Lesson 11: Model Evaluation Metrics (Medium)](https://medium.com/@nerdjock/deep-learning-course-lesson-11-model-evaluation-metrics-d85d0b85bcca)
- [Confusion Matrix for Multi-Class Classification (Analytics Vidhya)](https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/)
- [Confusion Matrix: How to Read & Interpret Classification Results (Label Your Data)](https://labelyourdata.com/articles/machine-learning/confusion-matrix)

### Graph Construction from CNN Features
- [CNN2Graph: Building Graphs for Image Classification (WACV 2023)](https://openaccess.thecvf.com/content/WACV2023/papers/Trivedy_CNN2Graph_Building_Graphs_for_Image_Classification_WACV_2023_paper.pdf)
- [Deep Learning with Graph Convolutional Networks (Wiley)](https://onlinelibrary.wiley.com/doi/10.1155/2023/8342104)
- [Understanding Convolutions on Graphs (Distill)](https://distill.pub/2021/understanding-gnns/)

### GNN Ablation Studies (Recent Academic Work)
- [LGSTA-GNN for Bridge Structural Damage Detection (MDPI, Jan 2026)](https://www.mdpi.com/2075-5309/16/2/348)
- [Research on GNNs with stable learning (Nature, Aug 2025)](https://www.nature.com/articles/s41598-025-12840-8)
- [DenseGNN for Material Properties (Nature, Dec 2024)](https://www.nature.com/articles/s41524-024-01444-x)
- [GNN for Blockchain Anomaly Detection (arXiv, Feb 2026)](https://arxiv.org/abs/2602.23599)

### ResNet-50 Feature Extraction
- [ResNet50 Integrated Vision Transformer (IEEE, 2026)](https://ieeexplore.ieee.org/document/10574771/)
- [What Is ResNet-50? (Roboflow)](https://blog.roboflow.com/what-is-resnet-50/)
- [A machine learning-based feature extraction method using ResNet (ScienceDirect, 2026)](https://www.sciencedirect.com/science/article/abs/pii/S1051200425000582)

### Reproducibility
- [How to Solve Reproducibility in ML (Neptune.ai)](https://neptune.ai/blog/how-to-solve-reproducibility-in-ml)
- [Reproducibility in PyTorch (PyTorch Docs)](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Ensuring Training Reproducibility in PyTorch (LearnOpenCV)](https://learnopencv.com/ensuring-training-reproducibility-in-pytorch/)

### GNN Interpretability and Visualization
- [Graph Attention Neural Networks for Interpretable Prediction (Wiley, 2026)](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aidi.202500061)
- [Chapter 7 Interpretability in Graph Neural Networks](https://graph-neural-networks.github.io/static/file/chapter7.pdf)
- [GNNExplainer: Generating Explanations for Graph Neural Networks (Stanford)](https://cs.stanford.edu/people/jure/pubs/gnnexplainer-neurips19.pdf)
- [DARTS-GT: Quantifiable Instance-Specific Interpretability (arXiv, 2025)](https://arxiv.org/html/2510.14336)

### CNN Baseline Comparison
- [A Comprehensive Overview and Comparative Analysis on Deep Learning Models (arXiv)](https://arxiv.org/abs/2305.17473)
- [Top 30+ Computer Vision Models For 2026 (Analytics Vidhya)](https://www.analyticsvidhya.com/blog/2025/03/computer-vision-models/)
- [An Evaluation of Deep CNN Baselines (arXiv)](https://arxiv.org/pdf/1805.06086)

---

**Research confidence:** MEDIUM-HIGH
- Table stakes features: HIGH confidence (standard ML pipeline, well-documented)
- Differentiators: MEDIUM confidence (ablation studies standard practice, but graph-specific strategies less documented for document images)
- Anti-features: HIGH confidence (based on academic project scope and rubric analysis)
- Dependencies: HIGH confidence (logical pipeline flow, validated against rubric requirements)

**Open questions for implementation:**
1. Graph construction strategy: Which k-NN variant performs best for document spatial structure? (Research contribution)
2. Feature map layer: Which ResNet-50 layer to use for graph nodes? (Experimentation needed)
3. Fine-tuning vs frozen CNN: Does fine-tuning on grayscale help or hurt? (Domain shift question)
4. GraphSAGE depth: How many layers before over-smoothing on document graphs? (Ablation will answer)
