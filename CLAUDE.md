# Deep Learning Document Image Classification

## Course
61.502 Deep Learning for Enterprise, Y2026 (SUTD)

## Team
| # | Name | Student ID |
|---|------|------------|
| 1 | Seow Chun Yong | 1000108 |
| 2 | Prathosh Chander | 1010948 |
| 3 | Sagar Pratap Singh | 1010736 |

## Deadline
- Submission: April 17th, 11:59pm

## Reference Materials
- `docs/EDL-Group1-Project-Proposal.docx` — Project proposal (objective, dataset, deliverables)
- `docs/Project Instructions-updated Rubrics.pdf` — Course instructions and grading rubrics
- `reference/RVL-CDIP_ResNet50.ipynb` — Reference notebook for ResNet-50 on RVL-CDIP

---

## Project Objective (from Proposal)

**Problem**: Classify scanned document images using GNNs that capture spatial layout structure, rather than treating images as flat pixel grids (CNN-only approach).

**Approach**: Convert CNN feature maps into graphs (nodes = image regions, edges = spatial relationships via k-NN), then apply GNN architectures for classification.

**Dataset**:
- **RVL-CDIP**: 400,000 grayscale document images, 16 classes (letter, form, email, invoice, etc.), split 320k/40k/40k train/val/test
- **RVL-CDIP-N**: 1,000 modern born-digital color documents, same 16 classes (out-of-distribution evaluation)
- Both available on HuggingFace

**Input/Output**:
- Input: Document image (grayscale or color)
- Output: Predicted class label (16 categories) + confidence score

**Key Challenges**:
1. Fine-tuning pretrained CNN (ResNet-50) on grayscale documents (domain shift from ImageNet)
2. Designing effective graph construction from feature maps
3. Comparing GNN architectures: GCN vs GAT vs GraphSAGE
4. Ablation studies showing when/why graph-based reasoning beats CNN-only

**Deliverables**:
1. Trained CNN baseline (fine-tuned ResNet-50)
2. Trained GNN models (GCN, GAT, GraphSAGE) with comparative results
3. Ablation study results (graph connectivity k-NN parameter, network depth)
4. End-to-end pipeline with pipeline diagram
5. Streamlit/FastAPI demo application
6. Documented GitHub repo with reproducible code and notebooks
7. Final report (10-15 pages) and presentation

---

## Grading Rubrics (from Project Instructions)

### Technical Implementation (50%)

- **Concept and Relevance [5%]**: Thorough understanding of relevant concepts and techniques demonstrated in implementation
- **Coding [25%]**: Well-structured, efficient, thoroughly documented code. Reproducibility of code
- **Performance & Evaluation [20%]**: Hyperparameter tuning, comparison to baseline. Metrics: Accuracy, Precision-Recall, F1-score (classification)

### Creativity and Innovation (5% bonus)
- Creative solutions or innovative approaches beyond state-of-the-art
- Unique features or functionalities that add significant value

### Presentation and Communication (20%)
- Clear and well-organized presentation
- Effective communication of objectives, methodology, and results
- Demonstration of impact / achievement of stated objectives

### Project Report (30%)
- Introduction: Problem statement, objectives, scope, constraints
- Method: Architecture and model presentation. Include tried-but-failed methods
- Experiments: Specifications used, comparison of results
- GitHub link (public), dataset/weights on Drive/Dropbox if heavy
- Conclusion

---

## Project Expectations (from Instructions)

### Code Quality
- Well-structured, well-documented repository
- Clear usage and installation instructions
- Reproducible, extensible, modular code
- pyproject.toml for dependencies, optional Dockerfile
- Document exact commands to reproduce pipeline and figures (include seeds, hardware notes)

### ML Requirements
- End-to-end ML pipeline with pipeline diagram in report
- Train/validation/test split
- Compare multiple models, choose best performer
- Visualization: accuracy/loss curves, performance on validation data
- Show model failure cases and discuss reasons
- All figures must have clear recreation instructions

### Serving/Deployment
- Lightweight demo (FastAPI/Flask or batch script)
- Mention monitoring needs (latency, drift) if deployed

### Report Structure (10-15 pages excl. appendix/references)
1. Executive Summary (max 1 page)
2. Background and Introduction
3. Related Work
4. Problem Formulation and Solution Overview
5. Data Description
6. Solution Details (methods, tools, hyperparameters, code link)
7. Evaluation (results, plots, precision-recall curves, bias audit)
8. Discussion of Results
9. Recommendations
10. Limitations, Caveats, Future Work
11. (Optional) Future Research Proposals

### Delivery
- Upload to GitHub (public): report PDF, documented code/notebooks
- Include directions for re-training from scratch
- Include directions to recreate exact trained model and results
- Submit link on eDimension
- Presentation in Week 13 (demo + slides/video)
