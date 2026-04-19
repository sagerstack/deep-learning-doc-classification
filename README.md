# Document Classification Using GNNs

**61.502 Deep Learning for Enterprise, Y2026 — SUTD**
Group 1: Seow Chun Yong (1000108), Prathosh Chander (1010948), Sagar Pratap Singh (1010736)

---

## What This Project Does

This project classifies scanned document images into 16 categories using Graph Neural Networks (GNNs) that reason over spatial layout structure. Instead of treating a document as a flat pixel grid, CNN feature maps are converted into graphs (nodes = image regions, edges = spatial relationships) and passed through GNN architectures for classification.

**Dataset**: RVL-CDIP — 400,000 grayscale scanned document images across 16 classes (letter, form, email, invoice, memo, advertisement, budget, news article, handwritten, scientific publication, resume, and more).

---

## Project Report

The final report and supporting materials are in `docs/`:

| File | Description |
|------|-------------|
| `docs/(EDL Group 1) Document Classification Using GNN- Final Project Report.pdf` | **Final report (PDF)** — submit this |
| `docs/(EDL Group 1) Document Classification Using GNN- Final Project Report.docx` | Editable Word source |
| `docs/EDL-Group1-Project-Proposal.docx` | Original project proposal |
| `docs/architecture.excalidraw` | Full system architecture diagram |
| `docs/fusionsage-architecture.excalidraw` | FusionGraphSAGE model architecture diagram |

---

## Repository Structure

```
.
├── final-notebooks/                  # Final experiment notebooks (one per model)
│   ├── RVL-CDIP_ResNet50 1.ipynb     # CNN baseline — fine-tuned ResNet-50
│   ├── 400K_GCN.ipynb                # Inductive GCN on full 400k dataset
│   ├── RVL-CDIP_GAT_multimodal.ipynb # Multimodal GAT with OCR-derived features
│   └── exp16_fusion_sage_results 1.ipynb  # FusionGraphSAGE (best OOD model)
│
├── final-models/                     # Trained model checkpoints
│   ├── best_model_resnet50_03.pt     # CNN Baseline (ResNet-50)
│   ├── inductive_gcn_320k.pt         # Inductive GCN
│   ├── best_gat_multimodal_k8_L2.pt  # Multimodal GAT
│   └── fusion_gnn_feat_knn_best.pt   # FusionGraphSAGE
│
├── app/                              # Demo web application (FastAPI)
│   └── src/
│       ├── main.py                   # FastAPI entry point
│       ├── routes/                   # classify + monitoring endpoints
│       ├── services/                 # model registry, inference, visualization
│       └── templates/                # HTML UI
│
├── src/                              # Shared model/graph definitions
│   ├── model.py                      # FusionGraphSAGE, InductiveGCN, DocumentGAT
│   ├── graph.py                      # Graph construction utilities
│   ├── data.py                       # Dataset loaders
│   └── train.py                      # Training loop
│
├── scripts/
│   ├── startup.sh                    # Start/stop all services
│   ├── populate_mlflow.py            # Load experiment results into MLflow
│   ├── monitoring/
│   │   ├── bootstrap_reference.py    # Generate reference dataset for drift monitoring
│   │   └── run_evidently.py          # Run Evidently drift report
│   └── restore_seq_dashboard.py      # Restore Seq log dashboard
│
├── monitoring/                       # Monitoring data and workspaces
│   ├── data/                         # SQLite inference event log
│   ├── evidently_workspace/          # Evidently drift reports
│   ├── mlflow/                       # MLflow experiment database
│   └── reference/                    # Reference dataset for drift detection
│
├── docs/                             # Reports, diagrams
├── docker-compose.yml                # App + Seq containers
├── pyproject.toml                    # Python dependencies (Poetry)
└── .env.example                      # Environment variable template
```

---

## Models

Four models are trained and served in the demo app:

| Model | Architecture | ID Test Acc | OOD Test Acc |
|-------|-------------|-------------|--------------|
| CNN Baseline | Fine-tuned ResNet-50 | 89.47% | 62.08% |
| Inductive GCN | GCN over 7×7 spatial grid | 86.57% | 73.95% |
| Fusion GraphSAGE | 2-layer GraphSAGE + CNN fusion shortcut | 88.47% | 67.07% |
| Multimodal GAT | GAT with OCR-derived BOC features | 89.55% | — |

**ID** = RVL-CDIP test set (10k held-out scanned documents).
**OOD** = RVL-CDIP-N (1,002 born-digital modern documents, same 16 classes).

---

## Quickstart

### Prerequisites

- Docker + Docker Compose
- Python 3.11+ with [Poetry](https://python-poetry.org/)
- Model checkpoints in `final-models/` (four `.pt` files — see table above)

### 1. Install dependencies

```bash
poetry install
```

### 2. Configure environment

```bash
cp .env.example .env.local
# Edit .env.local if needed (defaults work out of the box)
```

### 3. Start all services

```bash
scripts/startup.sh
```

This starts:

| Service | URL | Description |
|---------|-----|-------------|
| Demo app | http://localhost:9000 | Document classification UI |
| Evidently | http://localhost:8080 | Feature drift dashboard |
| MLflow | http://localhost:5050 | Experiment tracking UI |
| Seq | http://localhost:5341 | Structured inference log viewer |

### 4. Stop all services

```bash
scripts/startup.sh --stop
```

### Other startup options

```bash
scripts/startup.sh --reset       # Rebuild Docker image before starting
scripts/startup.sh --logs        # Tail app logs after startup
scripts/startup.sh --monitoring  # Also run Evidently batch job on startup
```

---

## Demo App Features

The demo app is a single-page FastAPI + Jinja2 web application accessible at `http://localhost:9000`.

### Upload and Classify

- Upload any document image (JPG, PNG, PDF page)
- Select which model to run: CNN Baseline, Inductive GCN, Fusion GraphSAGE, or Multimodal GAT
- View the predicted class and confidence score across all 16 categories

### Model Explainability Panels

Each GNN model shows an interactive breakdown of how the prediction was made:

- **CNN Features panel**: ResNet-50 feature activation heatmap over the 7×7 spatial grid
- **Graph Construction panel**: Visualisation of the k-NN graph built from the document's feature nodes
- **Model Predictions panel**: Per-class probability distribution with top-5 predictions

### Sample Documents

The app ships with pre-loaded sample images for quick testing:

- `app/src/static/samples/in-dist/` — 7 in-distribution samples (email, form, invoice, resume, handwritten, news article, scientific report)
- `app/src/static/samples/oo-dist/` — 9 out-of-distribution samples from RVL-CDIP-N (born-digital documents)

---

## Monitoring Stack

The project includes a production-grade monitoring stack:

| Tool | Purpose | Access |
|------|---------|--------|
| **MLflow** | Experiment tracking — all training runs, hyperparameters, metrics | http://localhost:5050 |
| **Evidently** | Feature and prediction drift detection | http://localhost:8080 |
| **Seq** | Structured inference event log (model name, class, confidence, latency) | http://localhost:5341 |

### Populate MLflow

```bash
poetry run python scripts/populate_mlflow.py
```

### Run a drift report

```bash
poetry run python scripts/monitoring/run_evidently.py --window-hours 24
```

### Bootstrap reference dataset (first run)

```bash
poetry run python scripts/monitoring/bootstrap_reference.py --synthetic
```

---

## Reproducing Training

Each model is fully reproducible from its notebook in `final-notebooks/`. The notebooks load pre-extracted ResNet-50 feature caches to avoid re-running the CNN backbone on every experiment.

### Feature extraction (one-time, ~13 hours on Apple Silicon MPS)

Run the feature extraction notebook first to generate the cache, or download pre-extracted features.

### Training each model

Open and run the corresponding notebook end-to-end:

| Notebook | Model | Dataset |
|----------|-------|---------|
| `RVL-CDIP_ResNet50 1.ipynb` | CNN Baseline | Full 400k RVL-CDIP |
| `400K_GCN.ipynb` | Inductive GCN | Full 400k RVL-CDIP |
| `RVL-CDIP_GAT_multimodal.ipynb` | Multimodal GAT | 100k subset |
| `exp16_fusion_sage_results 1.ipynb` | FusionGraphSAGE | Full 400k RVL-CDIP (cached) |

All notebooks include ablation study cells for graph connectivity (k-NN k parameter) and network depth (number of GNN layers).

---

## Dependencies

Managed via Poetry (`pyproject.toml`). Key libraries:

- `torch`, `torchvision` — model training and inference
- `torch-geometric` — GNN layers (GCN, GAT, GraphSAGE)
- `fastapi`, `uvicorn` — demo application server
- `evidently` — drift monitoring
- `mlflow` — experiment tracking
- `seqlog` — structured log shipping to Seq
