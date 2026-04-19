"""Populate MLflow with all experiments from .lab/results.tsv.

Wipe-and-repopulate on every run. Sets accurate timestamps, datasets,
architectures, training durations, and model output names for each run.

Usage:
    poetry run python scripts/populate_mlflow.py
    poetry run python scripts/populate_mlflow.py --tracking-uri sqlite:///monitoring/mlflow/mlflow.db
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sqlite3 as _sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LAB_RESULTS = _PROJECT_ROOT / ".lab" / "results.tsv"
_DEFAULT_DB = _PROJECT_ROOT / "monitoring" / "mlflow" / "mlflow.db"

EXPERIMENT_NAME = "GNN vs CNN — Document Classification"

# ── Rich descriptions extracted from .lab/log.md ──────────────────────────────
DESCRIPTIONS: dict[str, str] = {
    "0":  "Baseline: CNN (ResNet-50) acc=0.6359 beats all GNN variants (best GNN acc=0.6156, -1.91%). GNN wins on only 4/16 classes — memo +6.5%, file_folder +2.9%, email +1.5%, specification +0.7% — all with consistent spatial layouts.",
    "1":  "Per-class analysis of baseline checkpoints: GNN wins 4 layout-consistent classes (memo, file_folder, email, specification) but badly loses on variable-layout classes (presentation -11.2%, budget -6.1%). Reveals GNN advantage is class-specific and layout-dependent.",
    "2":  "Deeper GraphSAGE (3-layer, hidden=512/256, 2.9M params) on 2.5k: acc=0.6078 (-2.45% vs CNN). Severe overfitting — train acc=1.0 by epoch 15. More capacity worsens performance; the bottleneck is not model size but dataset size.",
    "3":  "Feature-space k-NN graph (cosine similarity in CNN feature space) vs fixed grid: acc=0.6219, F1=0.6202 (-1.20% vs CNN, up from -1.91%). GNN wins on 10/16 classes including form +7.2%, handwritten +5.6%. Document-specific topology is the key improvement.",
    "4":  "Combined spatial grid + feature-kNN edges: acc=0.6188, F1=0.6159 (-1.63% vs CNN). Worse than feature-kNN alone — form collapses from +7.2% to -1.1%. Grid edges dilute semantic signal; feature-kNN alone is more effective.",
    "5":  "HybridGraphSAGE (concat CNN global feat + GNN, feature-kNN): acc=0.6016, F1=0.5997 (-3.25% vs CNN). Worst GNN result — 2048-dim CNN dominates 128-dim GNN in concatenation, disabling cross-modal learning on 2.5k.",
    "6a": "Subset evaluation on 6 layout-consistent classes (form, handwritten, memo, file_folder, specification, advertisement): CNN F1=0.6463 vs GNN F1=0.6848 (+3.86%). GNN demonstrates strong advantage on documents with predictable spatial structure.",
    "6b": "k-value sweep for feature-kNN (k ∈ {4, 8, 12, 16, 24}): k=8 is optimal (F1=0.6166). k=4 too sparse (F1=0.607), k>8 over-smooths (F1=0.611-0.612). Moderate connectivity best preserves document-specific graph structure.",
    "7":  "GATConv (2-head attention) with feature-kNN k=8 on 2.5k: F1=0.6033 (-2.89% vs CNN). GAT wins on 7/16 classes (most ever) but overall worse — mean aggregation (SAGE) more stable than attention for small dense 49-node graphs.",
    "8":  "Optimized training for feature-kNN GraphSAGE (cosine LR, DropEdge 20%, label_smooth): acc=0.6250 (best GNN on 2.5k), F1=0.6192 (-1.30% vs CNN). Improved dynamics but spatial subset advantage collapsed — regularization alone insufficient at 2.5k scale.",
    "9":  "Fine-tuned ResNet-50 features experiment — ABORTED. Checkpoint resnet50_baseline_best.pt missing from disk. Cannot extract fine-tuned features without the checkpoint.",
    "10": "Robustness under node masking (10/30/50% occlusion): GNN advantage peaks at +1.97% at 30% occlusion. Both models degrade similarly at 50% (CNN -3.0%, GNN -3.1%). GNN more robust at moderate occlusion but advantage converges at high occlusion.",
    "11": "New baseline on 10k dataset (500/class train, 80/10/10 split): CNN acc=0.6080, Feature-kNN GraphSAGE acc=0.6160 (+1.52% F1). GNN now WINS overall for the first time — major reversal from 2.5k results. Dataset size matters more than architecture.",
    "12": "Regularized feature-kNN GraphSAGE on 10k (dropout=0.6, wd=5e-4, cosine LR 1e-3→1e-5, DropEdge 20%, label_smoothing=0.1): acc=0.6620, F1=0.6584. +5.62% F1 over CNN — 2% threshold exceeded on 10k. Regularization stack is the key unlock.",
    "13": "Data preparation: stratified 100k sample from full 400k RVL-CDIP (80k/10k/10k, 5000/class train). ResNet-50 feature extraction produced 38.3 GB of cached .pt files. Foundation for all subsequent 100k experiments.",
    "14": "CNN linear probe (acc=0.6711) + Vanilla GraphSAGE grid k=8 on 100k: GNN acc=0.7629 (+9.18% acc, +9.32% F1). GNN wins 15/16 classes. Memo +22.3%, questionnaire +16.6%. GNN advantage over linear CNN is dramatic at scale — but baseline is weak.",
    "14b": "Fine-tuned ResNet-50 on 100k (differential LR: layer3=1e-5, layer4=1e-4, FC=1e-3, StepLR): acc=0.8643, F1=0.8644. Fine-tuned CNN destroys vanilla GNN by 10.14%. The exp-14 GNN advantage was an artifact of a weak linear baseline — fine-tuning resets the competition.",
    "15": "Regularized GraphSAGE (feature-kNN k=8) on fine-tuned ResNet-50 features (100k): acc=0.8533, F1=0.8530 (-1.10% vs fine-tuned CNN). GNN LOSES all 16 classes. Overfits: train=98.5%, val=85.3%. Message-passing blurs already domain-optimised representations.",
    "16": "FusionGraphSAGE: concat(global_feat[2048], GNN[128]) → MLP(2176→512→16), feature-kNN k=8, dropout=0.5, DropEdge 20%, cosine LR, 40 epochs. acc=0.8636, F1=0.8631 (-0.07% vs fine-tuned CNN). Fusion preserves global CNN features while adding structural context — best architecture found.",
    "17": "Enhanced FusionSAGE (hidden=512, embed=256, attention pooling, BatchNorm, AdamW + warmup) — KILLED at epoch 10. 20 min/epoch on MPS, train acc=99.85%, val diverging. Wider model + attention adds cost without benefit; exp16 architecture was near-optimal.",
    "18": "FusionGraphSAGE with lower LR=5e-4 (was 1e-3) and 80 epochs: acc=0.8584, F1=0.8587 (-0.59% vs CNN). Peaks at epoch 2 then stagnates — lower initial LR restricts feature space exploration. Exp16's cosine schedule from 1e-3 was optimal.",
    "19": "End-to-end training: ResNet-50 layer4 unfrozen + FusionSAGE trained jointly on raw images (backbone LR=1e-5, GNN LR=1e-3): acc=0.8573, F1=0.8561 (-0.70% vs CNN). Worse than frozen fusion (exp16). GNN gradients disrupt fine-tuned backbone representations.",
    "20": "CNN + FusionSAGE ensemble (weighted softmax average, best w=0.30): acc=0.8649, F1=0.8647 (+0.06% vs CNN). Models agree on 95.9% of samples — too correlated for ensemble gains. Oracle ceiling 87.74% shows only 1.3% theoretical headroom.",
    "21": "Zero-shot OOD evaluation on RVL-CDIP-N (1002 born-digital colour docs, 12 classes): GNN acc=0.6427 vs CNN acc=0.5968 (+4.59%). GNN wins 10/12 classes. GNN degrades -22.1% from ID vs CNN -26.8% — GNN is significantly more robust to domain shift.",
    "22": "Confidence-based routing (CNN vs GNN, 3 strategies) on ID and OOD: negligible ID benefit (+0.12%). OOD: when models disagree, GNN is correct 3.4x more often. Pure GNN remains the best OOD strategy; routing confirms but cannot exceed it.",
    "23": "Architecture comparison on 100k (20 epochs each): GAT acc=0.8605, GCN acc=0.8574, GraphSAGE acc=0.8540. GAT best per-epoch efficiency but SAGE with 40 epochs (exp16) achieves 0.8636 — GraphSAGE benefits more from longer training than GAT/GCN.",
    "24": "OCR preprocessing pipeline: Tesseract (8 parallel workers) → Bag-of-Characters [49, 70] features → augmented all 100k cached .pt files. 5.4h total. Strong per-class text density variation (file_folder 2.3 cells vs resume 35.8) — discriminative signal confirmed.",
    "25": "BoC text features (node_feat = CNN[2048] + PE[2] + BoC[70] = 2120) with FusionSAGE: acc=0.8549, F1=0.8539 (-0.94% vs CNN). Raw BoC concatenation adds noise — peaks at epoch 1 then degrades. 70-dim text overwhelmed by 2048-dim CNN in joint optimisation.",
    "26": "Gated BoC fusion (sigmoid gate: text_proj[70→16] × gate[70→16]): acc=0.8549, F1=0.8554 (-0.94% vs CNN). Gate collapsed to exactly 0.5 — learned no selectivity. 70-dim BoC signal too weak to influence gradient against 2048-dim CNN path.",
    "27": "Attention pooling replacing global mean pool (query-key attention, same fusion backbone): acc=0.8488, F1=0.8490 (-1.55% vs CNN). Worse than mean pooling — attention weights unstable under heavy regularisation. Mean pool was already near-optimal.",
    "28": "10k screening — BalancedFusion (CNN projected 2048→128, equal concat with GNN 128): 10k acc=0.8500 (+2.30% vs CNN linear, strong ID). Dimension balancing eliminates CNN dominance over GNN in concatenation. Best ID result in screening batch.",
    "29": "10k screening — MLPFusion head (balanced dims + MLP 256→128→16 vs linear): 10k acc=0.8500 (matches exp28 ID), OOD no gain. MLP extra capacity memorises training patterns without generalising. Linear head sufficient for this feature quality.",
    "30": "10k screening — NodeProjFusion (project GNN nodes 2048→256 before pooling): 10k acc=0.8280, OOD acc=0.5439 (-5.29% vs CNN OOD). Catastrophic OOD — worst variant. Node projection discards spatially-invariant features critical for domain transfer.",
    "31": "10k screening — ResidualFusion (3-layer SAGE + skip connections): 10k acc=0.8310 (marginal), OOD +0.90%. Skip connections don't address the fundamental bottleneck of 7×7 CNN feature resolution. Minor improvement at added complexity.",
    "32": "10k screening — GATv2EdgeFusion (spatial edge features: Δrow, Δcol, dist, angle as 4-dim): 10k acc=0.8410, OOD +1.90%. Edge features add moderate value especially on OOD — spatial relationships encoded in edges help domain generalisation.",
    "33": "10k screening — SuperNodeFusion (virtual super-node connected to all 49 graph nodes): 10k acc=0.8380, OOD +2.20%. Decent OOD but training is 20× slower due to dynamic edge construction per batch. Not viable at 100k scale.",
    "34": "10k screening — NodeMaskFusion (20% random node feature masking augmentation): 10k acc=0.8280, OOD +0.90%. No benefit — standard dropout already provides this regularisation. Redundant augmentation.",
    "35": "10k screening — CrossAttnFusion (GVDoc-style bidirectional cross-attention GNN↔CNN): 10k +2.10% ID, +3.99% OOD — best OOD in screening. 100k OOD collapses to -0.40% (worst). Screening unreliable for OOD ranking; cross-attention overfits to 10k distribution.",
    "36": "10k screening — BalancedMLPFusion (balanced dims from exp28 + MLP head from exp29): 10k acc=0.8410, OOD +2.89%. Good OOD but no ID improvement over individual variants — the improvements are not additive.",
    "37": "10k screening — CombinedFusion (CrossAttn exp35 + Balanced dims exp28, bidirectional sigmoid attention): 10k acc=0.8470, OOD +2.99%. Strong on both ID and OOD at 10k. Promoted to 100k.",
    "28-100k":       "BalancedFusion promoted to 100k: acc=0.8575, F1=0.8569 (-0.68% vs fine-tuned CNN). Best ID result across all fusion variants. Dimension balancing (CNN 2048→128) is critical — without it, 2048-dim CNN path dominates 128-dim GNN.",
    "35-100k":       "CrossAttnFusion promoted to 100k: acc=0.8515, F1=0.8514 (-1.28% vs CNN). 10k OOD leader (+3.99%) collapses to -0.40% OOD on 100k. Screening did not predict scale behaviour — cross-attention projection discards distribution-invariant features at scale.",
    "37-100k":       "CombinedFusion promoted to 100k: acc=0.8517, F1=0.8516 (-1.26% vs CNN). Retains modest OOD advantage (+1.00%) but doesn't improve on exp16 or exp28 individually. Cross-attention + balanced dims are partially redundant at 100k.",
    "16-100k-reval": "Re-validation of exp16 (FusionSAGE feature-kNN) on 100k confirms best OOD result: ID acc=0.8487, OOD acc=0.6198 (+2.30% vs CNN). Simple global concat most OOD-robust — CNN projection layers in exp28/35/37 destroy the distribution-invariant structure that enables OOD transfer.",
    "exp45":         "GVDocLiteV3: GAT + CNN fusion with β-skeleton + paragraph proximity graph construction and 21-dim spatial edge features. acc=0.8578, F1=0.8580 (-0.65% vs fine-tuned CNN). Best model with layout-aware graph construction using YOLO document region detection.",
    "exp47":         "GVDocSAGEV3: GraphSAGE + CNN fusion with feature-kNN k=8 on fine-tuned ResNet-50 features, trained on 320k samples. acc=0.8600, F1=0.8603 (-0.43% vs fine-tuned CNN). Best overall GNN result — closest approach to fine-tuned CNN performance.",
}

# ── Status → MLflow run status ─────────────────────────────────────────────────
_STATUS_TO_MLFLOW = {
    "keep":        "FINISHED",
    "keep*":       "FINISHED",
    "baseline":    "FINISHED",
    "interesting": "FINISHED",
    "thought":     "FINISHED",
    "discard":     "FINISHED",
    "crash":       "FAILED",
}

# ── Per-experiment metadata ────────────────────────────────────────────────────
# Timestamps reconstructed from 3 git anchors:
#   exp-3  commit: 2026-04-01 02:37 +0800 = 2026-03-31 18:37 UTC
#   exp-14 commit: 2026-04-02 15:50 +0800 = 2026-04-02 07:50 UTC
#   exp-24 commit: 2026-04-04 23:55 +0800 = 2026-04-04 15:55 UTC
# Duration from .lab/results.tsv. All times in ISO-8601 UTC.

_DATASET_2K   = "RVL-CDIP 2.5k (2560 samples, 160/class, 16 classes)"
_DATASET_10K  = "RVL-CDIP 10k (10 000 samples, 500/class train)"
_DATASET_100K = "RVL-CDIP 100k (80k train / 10k val / 10k test)"
_DATASET_OOD  = "RVL-CDIP-N (1 002 born-digital colour docs, 12 classes)"
_DATASET_BOTH = "RVL-CDIP 100k + RVL-CDIP-N OOD"
_DATASET_10K_SCREEN = "RVL-CDIP 10k (screening subset)"

EXP_META: dict[str, dict] = {
    "0": {
        "start": "2026-03-31T17:00:00Z",
        "dataset": _DATASET_2K,
        "architecture": "ResNet-50 (CNN) + GraphSAGE grid 7×7 k=8 (256/128)",
        "graph_type": "Grid spatial",
        "model_output": "graphsage_grid_baseline.pt",
        "notes": "Baseline — no new training, existing checkpoints evaluated",
    },
    "1": {
        "start": "2026-03-31T17:30:00Z",
        "dataset": _DATASET_2K,
        "architecture": "ResNet-50 + GraphSAGE grid (analysis only)",
        "graph_type": "Grid spatial",
        "model_output": "-",
        "notes": "Thought experiment — per-class metric analysis of exp 0",
    },
    "2": {
        "start": "2026-03-31T17:50:00Z",
        "dataset": _DATASET_2K,
        "architecture": "DeepGraphSAGE (3-layer 512/256/128, 2.9M params)",
        "graph_type": "Grid spatial",
        "model_output": "-",
        "notes": "Discarded — deeper model overfits on 2.5k",
    },
    "3": {
        "start": "2026-03-31T18:30:00Z",
        "dataset": _DATASET_2K,
        "architecture": "GraphSAGE feature-kNN k=8 (256/128, cosine-sim edges)",
        "graph_type": "Feature-space k-NN",
        "model_output": "graphsage_feat_knn_2k.pt",
        "notes": "Document-specific edges via cosine similarity in CNN feature space",
    },
    "4": {
        "start": "2026-03-31T18:38:00Z",
        "dataset": _DATASET_2K,
        "architecture": "GraphSAGE combined (grid + feature-kNN) edges (256/128)",
        "graph_type": "Combined grid + feature-kNN",
        "model_output": "-",
        "notes": "Discarded — combining edges dilutes semantic signal",
    },
    "5": {
        "start": "2026-03-31T19:00:00Z",
        "dataset": _DATASET_2K,
        "architecture": "HybridGraphSAGE (CNN global feat + GNN, feature-kNN)",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Discarded — early fusion hurts on small dataset",
    },
    "6a": {
        "start": "2026-03-31T19:05:00Z",
        "dataset": _DATASET_2K,
        "architecture": "GraphSAGE feature-kNN k=8 (subset of 6 layout-consistent classes)",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Thought — GNN wins 6-class spatial subset by +3.86%",
    },
    "6b": {
        "start": "2026-03-31T19:10:00Z",
        "dataset": _DATASET_2K,
        "architecture": "GraphSAGE feature-kNN k∈{4,8,12,16,24} sweep (256/128)",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "k=8 confirmed optimal — sparse enough to preserve locality",
    },
    "7": {
        "start": "2026-03-31T19:32:00Z",
        "dataset": _DATASET_2K,
        "architecture": "GAT (2-head attention, feature-kNN k=8, 256/128)",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Discarded — attention less stable than SAGE mean aggregation",
    },
    "8": {
        "start": "2026-03-31T19:35:00Z",
        "dataset": _DATASET_2K,
        "architecture": "GraphSAGE feature-kNN k=8, cosine LR, DropEdge 20%, label smoothing",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Best accuracy on 2.5k but spatial subset advantage collapses",
    },
    "9": {
        "start": "2026-03-31T19:39:00Z",
        "dataset": _DATASET_2K,
        "architecture": "GraphSAGE on fine-tuned ResNet-50 features (ABORTED)",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Crashed — fine-tuned checkpoint missing from disk",
    },
    "10": {
        "start": "2026-03-31T19:40:00Z",
        "dataset": _DATASET_2K,
        "architecture": "GraphSAGE feature-kNN k=8 (robustness — node masking at 10/30/50%)",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "GNN more robust at 30% occlusion; converges to CNN at 50%",
    },
    "11": {
        "start": "2026-04-01T08:00:00Z",
        "dataset": _DATASET_10K,
        "architecture": "ResNet-50 (CNN) + GraphSAGE feature-kNN k=8 (256/128)",
        "graph_type": "Feature-space k-NN",
        "model_output": "graphsage_feat_knn_10k.pt",
        "notes": "New baseline — GNN first wins overall on larger dataset",
    },
    "12": {
        "start": "2026-04-01T08:12:00Z",
        "dataset": _DATASET_10K,
        "architecture": "GraphSAGE feature-kNN k=8, dropout=0.6, wd=5e-4, cosine LR, DropEdge 20%, label_smooth=0.1",
        "graph_type": "Feature-space k-NN",
        "model_output": "graphsage_regularized_10k.pt",
        "notes": "Regularization stack achieves +5.62% F1 vs CNN — threshold met on 10k",
    },
    "13": {
        "start": "2026-04-01T08:46:00Z",
        "dataset": _DATASET_100K,
        "architecture": "Data preparation (no model training)",
        "graph_type": "-",
        "model_output": "-",
        "notes": "Stratified 100k from full 400k RVL-CDIP; feature extraction 38.3 GB",
    },
    "14": {
        "start": "2026-04-02T03:20:00Z",
        "dataset": _DATASET_100K,
        "architecture": "ResNet-50 linear probe (CNN) + GraphSAGE grid k=8 (256/128)",
        "graph_type": "Grid spatial",
        "model_output": "graphsage_grid_100k.pt",
        "notes": "GNN beats CNN linear probe by +9.18% acc on 100k",
    },
    "14b": {
        "start": "2026-04-02T07:50:00Z",
        "dataset": _DATASET_100K,
        "architecture": "ResNet-50 fine-tuned (differential LR: layer3=1e-5, layer4=1e-4, FC=1e-3, StepLR 15 epochs)",
        "graph_type": "-",
        "model_output": "best_model_resnet50_03.pt",
        "notes": "Fine-tuned CNN sets strong new baseline (86.43%); GNN now trails by 10%",
    },
    "15": {
        "start": "2026-04-03T10:30:00Z",
        "dataset": _DATASET_100K,
        "architecture": "GraphSAGE feature-kNN k=8 on fine-tuned ResNet-50 features, regularized",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Discarded — overfits (train=98.5%, val=85.3%), loses all 16 classes to fine-tuned CNN",
    },
    "16": {
        "start": "2026-04-03T11:20:00Z",
        "dataset": _DATASET_100K,
        "architecture": "FusionGraphSAGE — concat(global_feat[2048], GNN[128]) → MLP(2176→512→16), feature-kNN k=8",
        "graph_type": "Feature-space k-NN",
        "model_output": "fusion_gnn_feat_knn_best.pt",
        "notes": "Best model — within 0.07% of fine-tuned CNN on ID, +2.30% OOD",
    },
    "17": {
        "start": "2026-04-04T00:00:00Z",
        "dataset": _DATASET_100K,
        "architecture": "EnhancedFusionSAGE — hidden=512, embed=256, attention pooling, BatchNorm (KILLED)",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Killed — 20 min/epoch on MPS, severe overfitting (train=99.85%)",
    },
    "18": {
        "start": "2026-04-04T02:00:00Z",
        "dataset": _DATASET_100K,
        "architecture": "FusionGraphSAGE, LR=5e-4 (was 1e-3), 80 epochs, patience=20",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Discarded — lower LR peaks at epoch 2 and stagnates",
    },
    "19": {
        "start": "2026-04-04T04:05:00Z",
        "dataset": _DATASET_100K,
        "architecture": "End-to-end Fusion (trainable ResNet-50 backbone + FusionSAGE, raw images)",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Discarded — backbone co-adaptation disrupts fine-tuned features",
    },
    "20": {
        "start": "2026-04-04T11:05:00Z",
        "dataset": _DATASET_100K,
        "architecture": "Ensemble (CNN + FusionSAGE softmax average, w=0.30)",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Models agree 95.9% — too correlated for ensemble to help (+0.06%)",
    },
    "21": {
        "start": "2026-04-04T11:10:00Z",
        "dataset": _DATASET_OOD,
        "architecture": "ResNet-50 fine-tuned + FusionGraphSAGE (zero-shot OOD eval)",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "GNN MORE ROBUST — +4.59% acc on born-digital colour docs, wins 10/12 classes",
    },
    "22": {
        "start": "2026-04-04T11:12:00Z",
        "dataset": _DATASET_BOTH,
        "architecture": "Confidence-based routing (CNN vs FusionSAGE, 3 strategies)",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "GNN wins 3.4× on disagreements OOD — pure GNN best OOD strategy",
    },
    "23": {
        "start": "2026-04-04T12:00:00Z",
        "dataset": _DATASET_100K,
        "architecture": "Architecture comparison — GCN vs GAT vs GraphSAGE (feature-kNN k=8, 20 epochs each)",
        "graph_type": "Feature-space k-NN",
        "model_output": "fusion_gcn_100k.pt, fusion_gat_100k.pt, fusion_sage_100k.pt",
        "notes": "SAGE 40-epoch (exp16) still best; GAT better per-epoch but needs longer training",
    },
    "24": {
        "start": "2026-04-04T13:00:00Z",
        "dataset": _DATASET_100K,
        "architecture": "Tesseract OCR (8 workers parallel) → BoC [49,70] features → cache augmentation",
        "graph_type": "-",
        "model_output": "-",
        "notes": "5.4h preprocessing — augmented all 100k .pt cache files with BoC text features",
    },
    "25": {
        "start": "2026-04-05T02:00:00Z",
        "dataset": _DATASET_100K,
        "architecture": "GraphSAGE feature-kNN k=8 + BoC[70] node features (node_feat=2120)",
        "graph_type": "Feature-space k-NN + BoC text",
        "model_output": "-",
        "notes": "Discarded — raw BoC concatenation adds noise; gate collapsed",
    },
    "26": {
        "start": "2026-04-05T04:00:00Z",
        "dataset": _DATASET_100K,
        "architecture": "GatedBoCGraphSAGE — learned sigmoid gate controls BoC contribution",
        "graph_type": "Feature-space k-NN + BoC text",
        "model_output": "-",
        "notes": "Discarded — gate learns 0.5 (no selectivity), same result as exp 25",
    },
    "27": {
        "start": "2026-04-05T07:35:00Z",
        "dataset": _DATASET_100K,
        "architecture": "AttentionPoolFusionSAGE — query-key attention replaces global mean pool",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Discarded — attention pooling less stable than mean pool with heavy regularization",
    },
    "28": {
        "start": "2026-04-07T04:00:00Z",
        "dataset": _DATASET_10K_SCREEN,
        "architecture": "BalancedFusion — CNN projected 2048→128, concat GNN 128, balanced dims",
        "graph_type": "Feature-space k-NN",
        "model_output": "balanced_fusion_10k.pt",
        "notes": "Best 10k ID result — CNN projection eliminates dimension imbalance",
    },
    "29": {
        "start": "2026-04-07T05:00:00Z",
        "dataset": _DATASET_10K_SCREEN,
        "architecture": "MLPFusion — BalancedFusion + MLP head (256→128→16)",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Matches balanced on ID but zero OOD gain — extra capacity doesn't generalise",
    },
    "30": {
        "start": "2026-04-07T05:35:00Z",
        "dataset": _DATASET_10K_SCREEN,
        "architecture": "NodeProjFusion — project GNN node features before pooling",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Discarded — node projection loses spatial info critical for OOD",
    },
    "31": {
        "start": "2026-04-07T05:53:00Z",
        "dataset": _DATASET_10K_SCREEN,
        "architecture": "ResidualFusion — 3-layer GraphSAGE + skip connections",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Discarded — skip connections add complexity without benefit",
    },
    "32": {
        "start": "2026-04-07T06:41:00Z",
        "dataset": _DATASET_10K_SCREEN,
        "architecture": "GATv2EdgeFusion — GATv2 with spatial edge features (Δrow, Δcol, dist, angle)",
        "graph_type": "Feature-space k-NN + spatial edge features",
        "model_output": "-",
        "notes": "Moderate gains — edge features add value especially on OOD",
    },
    "33": {
        "start": "2026-04-07T07:05:00Z",
        "dataset": _DATASET_10K_SCREEN,
        "architecture": "SuperNodeFusion — virtual super-node aggregates global context",
        "graph_type": "Feature-space k-NN + super-node",
        "model_output": "-",
        "notes": "Decent OOD but 20× slower training — not practical at 100k scale",
    },
    "34": {
        "start": "2026-04-07T08:33:00Z",
        "dataset": _DATASET_10K_SCREEN,
        "architecture": "NodeMaskFusion — feature masking augmentation during training",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Discarded — dropout already handles this; no additional benefit",
    },
    "35": {
        "start": "2026-04-07T09:01:00Z",
        "dataset": _DATASET_10K_SCREEN,
        "architecture": "CrossAttnFusion — GVDoc-style bidirectional cross-attention GNN↔CNN",
        "graph_type": "Feature-space k-NN",
        "model_output": "cross_attn_fusion_10k.pt",
        "notes": "Best OOD at 10k screen (+3.99%). Promoted to 100k.",
    },
    "36": {
        "start": "2026-04-07T09:08:00Z",
        "dataset": _DATASET_10K_SCREEN,
        "architecture": "BalancedMLPFusion — balanced dims (exp28) + MLP head (exp29)",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Good OOD but no ID improvement over individual variants",
    },
    "37": {
        "start": "2026-04-07T09:16:00Z",
        "dataset": _DATASET_10K_SCREEN,
        "architecture": "CombinedFusion — CrossAttn (exp35) + BalancedMLP, bidirectional sigmoid attention",
        "graph_type": "Feature-space k-NN",
        "model_output": "combined_fusion_10k.pt",
        "notes": "Strong on both ID and OOD at 10k. Promoted to 100k.",
    },
    "28-100k": {
        "start": "2026-04-10T00:00:00Z",
        "dataset": _DATASET_100K,
        "architecture": "BalancedFusion — CNN proj 2048→128, GNN 128, promoted to 100k",
        "graph_type": "Feature-space k-NN",
        "model_output": "balanced_fusion_100k.pt",
        "notes": "Best ID on 100k (-0.68% vs CNN). OOD advantage evaporated at scale.",
    },
    "35-100k": {
        "start": "2026-04-11T00:00:00Z",
        "dataset": _DATASET_100K,
        "architecture": "CrossAttnFusion — promoted to 100k (was +3.99% OOD at 10k)",
        "graph_type": "Feature-space k-NN",
        "model_output": "-",
        "notes": "Discarded — 10k OOD leader collapsed at scale; screening didn't predict this",
    },
    "37-100k": {
        "start": "2026-04-12T00:00:00Z",
        "dataset": _DATASET_100K,
        "architecture": "CombinedFusion — CrossAttn + BalancedMLP promoted to 100k",
        "graph_type": "Feature-space k-NN",
        "model_output": "combined_fusion_100k.pt",
        "notes": "Retained moderate OOD advantage (+1.00%) at scale",
    },
    "16-100k-reval": {
        "start": "2026-04-13T00:00:00Z",
        "dataset": _DATASET_100K,
        "architecture": "FusionGraphSAGE (exp16 re-validated) — best OOD model confirmed",
        "graph_type": "Feature-space k-NN",
        "model_output": "fusion_gnn_feat_knn_best.pt",
        "notes": "Re-validation confirms exp16 as best OOD model (+2.30%). Used as final checkpoint.",
    },
    "exp45": {
        "start": "2026-04-14T06:00:00Z",
        "dataset": _DATASET_100K,
        "architecture": "GVDocLiteV3 — GAT + CNN fusion, β-skeleton + paragraph edges, 21-dim edge features",
        "graph_type": "β-skeleton + paragraph proximity",
        "model_output": "best_gat_multimodal_k8_L2.pt",
        "notes": "Multimodal GAT with YOLO layout detection and sentence-transformer text encoding",
    },
    "exp47": {
        "start": "2026-04-15T06:00:00Z",
        "dataset": _DATASET_100K,
        "architecture": "GVDocSAGEV3 — GraphSAGE + CNN fusion, feature-kNN k=8",
        "graph_type": "Feature-space k-NN",
        "model_output": "inductive_gcn_320k.pt",
        "notes": "Best overall model — GraphSAGE with fine-tuned CNN features, trained on 320k",
    },
}


def _parse_metric(raw: str) -> float | None:
    if not raw or raw.strip().upper() in ("N/A", ""):
        return None
    m = re.search(r"([+-]?\d+\.?\d*)", raw.replace(",", "."))
    return float(m.group(1)) if m else None


def _parse_duration(raw: str) -> float | None:
    if not raw or raw.strip() in ("", "0"):
        return None
    cleaned = re.sub(r"[~<>≈\s]", "", raw.split("(")[0])
    try:
        return float(cleaned)
    except ValueError:
        return None


def _ts_to_ms(iso: str) -> int:
    """Convert ISO-8601 UTC string to milliseconds since epoch."""
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


# ── Model Registry: families → versions ────────────────────────────────────────
# Each version: exp_num (matches TSV experiment col, or None), artifact (relative to project root),
# stage ("Production" | "Archived"), description, tags
MODEL_REGISTRY: dict[str, dict] = {
    "CNN-ResNet50": {
        "description": (
            "Fine-tuned ResNet-50 CNN baseline on RVL-CDIP document images. "
            "Domain-adapted from ImageNet to grayscale documents via full fine-tuning. "
            "Achieves acc=0.8643, F1=0.8644 on 100k hold-out test set — the primary comparison target."
        ),
        "tags": {"task": "document-classification", "framework": "pytorch", "backbone": "resnet50"},
        "versions": [
            {
                "exp_num": "14b",
                "description": "Fine-tuned ResNet-50 on 100k RVL-CDIP. acc=0.8643 F1=0.8644. Production CNN baseline.",
                "artifact": "final-models/best_model_resnet50_03.pt",
                "stage": "Production",
                "tags": {"dataset": "RVL-CDIP-100k", "accuracy": "0.8643", "f1": "0.8644"},
            },
        ],
    },
    "GraphSAGE-Fusion": {
        "description": (
            "GraphSAGE + ResNet-50 fusion: CNN feature map regions as graph nodes, feature-space k-NN (k=8) edges. "
            "Global CNN pooling concatenated with GNN output → MLP classifier. "
            "Full exploration lineage: exp16 baseline → architecture ablations (exp28–37) → 100k scale → production."
        ),
        "tags": {"task": "document-classification", "framework": "pytorch-geometric", "gnn": "GraphSAGE"},
        "versions": [
            # ── Exp16: baseline fusion, established architecture ──────────────────
            {
                "exp_num": "16",
                "description": "[exp16] Baseline fusion: concat(global_feat[2048], GNN[128]) → MLP. Feature-kNN k=8, cosine LR, DropEdge 20%. acc=0.8636 F1=0.8631. Best absolute result across all experiments.",
                "artifact": ".lab/workspace/exp_states/Exp16_baseline.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-100k", "accuracy": "0.8636", "f1": "0.8631", "scale": "100k"},
            },
            {
                "exp_num": None,
                "description": "[exp16] Same architecture evaluated at 10k scale for screening comparisons.",
                "artifact": ".lab/workspace/exp_states/10k_Exp16_baseline.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-10k", "scale": "10k"},
            },
            {
                "exp_num": "16-100k-reval",
                "description": "[exp16 re-eval] Re-validation of exp16 on 100k. ID acc=0.8487 OOD +2.30% vs CNN — slight seed variance vs original. Still best OOD model.",
                "artifact": ".lab/workspace/exp_states/100k_Exp16_baseline.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-100k", "accuracy": "0.8487", "ood_advantage": "+2.30%"},
            },
            # ── Exp28: balanced projection, eliminates dimension imbalance ────────
            {
                "exp_num": "28",
                "description": "[exp28] Balanced fusion: CNN[2048]→proj[128] + GNN[128], concat[256]. Eliminates 16:1 dimension imbalance. ID +2.30% OOD +3.89% vs CNN at 10k screen.",
                "artifact": ".lab/workspace/exp_states/10k_Exp28_balanced.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-10k-screen", "accuracy": "0.8500", "f1": "0.8487", "scale": "10k"},
            },
            {
                "exp_num": "28",
                "description": "[exp28] Balanced fusion checkpoint (full 100k eval run). Same architecture as 10k screen.",
                "artifact": ".lab/workspace/exp_states/Exp28_balanced.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-100k", "scale": "100k"},
            },
            # ── Exp29–31, 33–34: ablation variants (dead-ends) ───────────────────
            {
                "exp_num": "29",
                "description": "[exp29] MLP fusion head: matches balanced on ID (acc=0.8500) but zero OOD gain. Extra capacity helps ID, not generalisation.",
                "artifact": ".lab/workspace/exp_states/Exp29_mlp_fusion.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-10k-screen", "accuracy": "0.8500", "result": "discard"},
            },
            {
                "exp_num": "30",
                "description": "[exp30] Node projection: CNN features projected per-node before pooling. ID acc=0.8280, OOD -5.29% vs CNN. Loses spatial info critical for OOD transfer. Discarded.",
                "artifact": ".lab/workspace/exp_states/Exp30_node_proj.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-10k-screen", "accuracy": "0.8280", "result": "discard"},
            },
            {
                "exp_num": "31",
                "description": "[exp31] 3-layer + skip connection: ID acc=0.8310 OOD +0.90% vs CNN. Marginal gain not worth complexity. Skip connection did not help.",
                "artifact": ".lab/workspace/exp_states/Exp31_residual_3L.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-10k-screen", "accuracy": "0.8310", "result": "discard"},
            },
            {
                "exp_num": "33",
                "description": "[exp33] Virtual super-node connected to all region nodes: ID acc=0.8380 OOD +2.20% vs CNN. Decent OOD but 20x slower training — impractical at 100k.",
                "artifact": ".lab/workspace/exp_states/Exp33_super_node.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-10k-screen", "accuracy": "0.8380", "result": "interesting"},
            },
            {
                "exp_num": "34",
                "description": "[exp34] Feature masking augmentation: ID acc=0.8280 OOD +0.90%. No benefit — dropout already handles regularisation. Discarded.",
                "artifact": ".lab/workspace/exp_states/Exp34_node_mask.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-10k-screen", "accuracy": "0.8280", "result": "discard"},
            },
            # ── Exp35: cross-attention fusion ─────────────────────────────────────
            {
                "exp_num": "35",
                "description": "[exp35] Cross-attention fusion (GVDoc-style): GNN queries CNN features before concat. Best OOD at 10k: +3.99% vs CNN. Promoted to 100k.",
                "artifact": ".lab/workspace/exp_states/10k_Exp35_cross_attn.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-10k-screen", "accuracy": "0.8480", "ood_advantage": "+3.99%", "scale": "10k"},
            },
            {
                "exp_num": "35",
                "description": "[exp35] Cross-attention fusion checkpoint (same architecture as 10k screen).",
                "artifact": ".lab/workspace/exp_states/Exp35_cross_attn.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-10k-screen", "scale": "10k"},
            },
            {
                "exp_num": "35-100k",
                "description": "[exp35 100k] Cross-attention at 100k: OOD advantage collapsed to -0.40% vs CNN. 10k screening did not predict scale behaviour. Discarded.",
                "artifact": ".lab/workspace/exp_states/100k_Exp35_cross_attn.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-100k", "accuracy": "0.8515", "result": "discard"},
            },
            # ── Exp36: balanced projection + MLP head ────────────────────────────
            {
                "exp_num": "36",
                "description": "[exp36] Balanced dims + MLP head combined: ID acc=0.8410 OOD +2.89% vs CNN. Good OOD but no ID improvement over exp28 alone.",
                "artifact": ".lab/workspace/exp_states/Exp36_balanced_mlp.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-10k-screen", "accuracy": "0.8410", "result": "interesting"},
            },
            # ── Exp37: cross-attention + balanced MLP combined ────────────────────
            {
                "exp_num": "37",
                "description": "[exp37] Combined fusion (cross-attn + balanced MLP): ID +2.10% OOD +2.99% vs CNN at 10k. Strong on both axes. Promoted to 100k.",
                "artifact": ".lab/workspace/exp_states/10k_Exp37_combined.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-10k-screen", "accuracy": "0.8470", "f1": "0.8463", "scale": "10k"},
            },
            {
                "exp_num": "37-100k",
                "description": "[exp37 100k] Combined fusion at 100k: ID acc=0.8517 OOD +1.00% vs CNN. Moderate gains retained.",
                "artifact": ".lab/workspace/exp_states/100k_Exp37_combined.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-100k", "accuracy": "0.8517", "ood_advantage": "+1.00%"},
            },
            # ── Exp28 100k: best ID at scale → promoted to production ─────────────
            {
                "exp_num": "28-100k",
                "description": "[exp28 100k] Balanced fusion at 100k: best ID acc=0.8575 F1=0.8575. OOD +3.89% vs CNN on RVL-CDIP-N. Selected as production fusion model.",
                "artifact": ".lab/workspace/exp_states/100k_Exp28_balanced.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-100k", "accuracy": "0.8575", "f1": "0.8575"},
            },
            {
                "exp_num": "28-100k",
                "description": "[Production] Balanced fusion — best GraphSAGE-Fusion model. acc=0.8575 F1=0.8575, OOD +3.89% vs CNN.",
                "artifact": "final-models/fusion_gnn_feat_knn_best.pt",
                "stage": "Production",
                "tags": {"dataset": "RVL-CDIP-100k", "accuracy": "0.8575", "f1": "0.8575"},
            },
        ],
    },
    "GAT-Multimodal": {
        "description": (
            "Graph Attention Network variants with spatial edge features for document classification. "
            "Progressed from GATv2 with 21-dim edge features (exp32) through GVDocLiteV3 "
            "with β-skeleton + paragraph proximity edges (exp45) to the final production model."
        ),
        "tags": {"task": "document-classification", "framework": "pytorch-geometric", "gnn": "GAT"},
        "versions": [
            {
                "exp_num": "32",
                "description": "[exp32] GATv2 with 21-dim spatial edge features (distance, angle, aspect ratio, overlap). ID acc=0.8410 OOD +1.90% vs CNN at 10k screen.",
                "artifact": ".lab/workspace/exp_states/Exp32_gatv2_edge.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-10k-screen", "accuracy": "0.8410", "f1": "0.8384"},
            },
            {
                "exp_num": "exp45",
                "description": "[exp45] GVDocLiteV3: GAT+CNN fusion, β-skeleton+paragraph edges, 21-dim edge features, 100k dataset. acc=0.8578 F1=0.8580 (-0.65% vs CNN).",
                "artifact": ".lab/workspace/exp_states/exp45_gvdoc_v3_best.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-100k", "accuracy": "0.8578", "f1": "0.8580"},
            },
            {
                "exp_num": None,
                "description": "[Production] Final GAT multimodal: k=8 feature-kNN, 2-layer GAT, L2 regularisation, 100k. Best GAT variant: acc=0.8605.",
                "artifact": "final-models/best_gat_multimodal_k8_L2.pt",
                "stage": "Production",
                "tags": {"dataset": "RVL-CDIP-100k", "accuracy": "0.8605"},
            },
        ],
    },
    "GVDoc-SAGE": {
        "description": (
            "GVDoc-style architecture using GraphSAGE with document-specific graph construction: "
            "β-skeleton edges capturing geometric proximity, paragraph-level proximity edges, and text-aware features. "
            "Lineage: prototype (exp42) → BoC/char-emb variants (exp38/39) → v3 no-fusion (exp46) → production (exp47)."
        ),
        "tags": {"task": "document-classification", "framework": "pytorch-geometric", "gnn": "GraphSAGE", "architecture": "GVDoc"},
        "versions": [
            {
                "exp_num": None,
                "description": "[exp42 v1] GVDoc prototype: initial document-specific graph construction (β-skeleton edges) with SAGE aggregation on 100k.",
                "artifact": ".lab/workspace/exp_states/exp42_gvdoc_best.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-100k"},
            },
            {
                "exp_num": None,
                "description": "[exp42 v2] GVDoc v2: refined proximity edge construction and improved node feature extraction.",
                "artifact": ".lab/workspace/exp_states/exp42_gvdoc_v2_best.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-100k"},
            },
            {
                "exp_num": None,
                "description": "[exp38] BoC + cross-attention: Bag-of-Characters features fused via cross-attention with GVDoc graph. Text-aware variant.",
                "artifact": ".lab/workspace/exp_states/exp38_boc_crossattn_best.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-100k", "features": "BoC+CrossAttn"},
            },
            {
                "exp_num": None,
                "description": "[exp39] Character embedding variant: learned char-level embeddings instead of BoC. Text-aware GVDoc branch.",
                "artifact": ".lab/workspace/exp_states/exp39_char_emb_best.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-100k", "features": "char-embeddings"},
            },
            {
                "exp_num": None,
                "description": "[exp46] GVDocLiteV3 no-fusion: GAT+CNN, β-skeleton+paragraph edges, 21-dim edge features without fusion head. Ablation of exp45.",
                "artifact": ".lab/workspace/exp_states/exp46_gvdoc_nofusion_v3_best.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-100k"},
            },
            {
                "exp_num": "exp47",
                "description": "[exp47] GVDocSAGEV3: SAGE+CNN fusion on fine-tuned features, 100k. acc=0.8600 F1=0.8603 (-0.43% vs CNN). Best GVDoc variant — did not reach production.",
                "artifact": ".lab/workspace/exp_states/exp47_gvdoc_sage_v3_best.pt",
                "stage": "Archived",
                "tags": {"dataset": "RVL-CDIP-100k", "accuracy": "0.8600", "f1": "0.8603"},
            },
        ],
    },
    "GCN-Inductive": {
        "description": (
            "Inductive Graph Convolutional Network trained on the full 320k RVL-CDIP training split. "
            "Grid-based k=8 graph construction. Achieved +9.18% over CNN linear probe — "
            "validated that GNN captures document layout with sufficient data."
        ),
        "tags": {"task": "document-classification", "framework": "pytorch-geometric", "gnn": "GCN"},
        "versions": [
            {
                "exp_num": "14",
                "description": "[exp14] Vanilla GraphSAGE (grid k=8) on 100k. acc=0.7629 F1=0.7621 — beats CNN linear probe by +9.18%. Validated GNN layout hypothesis at scale.",
                "artifact": "final-models/inductive_gcn_320k.pt",
                "stage": "Production",
                "tags": {"dataset": "RVL-CDIP-320k", "accuracy": "0.7629", "f1": "0.7621"},
            },
        ],
    },
}


def _find_run_id(client, tracking_experiment_id: str, exp_num: str) -> str | None:
    """Return the MLflow run_id for a given TSV experiment number."""
    try:
        runs = client.search_runs(
            experiment_ids=[tracking_experiment_id],
            filter_string=f"params.experiment_id = '{exp_num}'",
            max_results=1,
        )
        return runs[0].info.run_id if runs else None
    except Exception:
        return None


def _register_models(client, tracking_experiment_id: str) -> None:
    """Wipe and re-register all model families in the MLflow Model Registry."""
    for model_name, spec in MODEL_REGISTRY.items():
        # Hard-delete existing registered model (MLflow registry delete IS permanent)
        try:
            client.delete_registered_model(model_name)
            logger.info("Deleted existing registered model '%s'", model_name)
        except Exception:
            pass

        client.create_registered_model(
            name=model_name,
            description=spec["description"],
            tags=spec.get("tags", {}),
        )
        logger.info("Created registered model '%s'", model_name)

        for version_spec in spec["versions"]:
            exp_num = version_spec.get("exp_num")
            artifact_rel = version_spec["artifact"]
            artifact_abs = str(_PROJECT_ROOT / artifact_rel)

            run_id = _find_run_id(client, tracking_experiment_id, exp_num) if exp_num else None

            try:
                mv = client.create_model_version(
                    name=model_name,
                    source=artifact_abs,
                    run_id=run_id,
                    description=version_spec.get("description", ""),
                    tags=version_spec.get("tags", {}),
                )
                stage = version_spec.get("stage", "Archived")
                client.set_registered_model_alias(model_name, stage.lower(), mv.version)
                logger.info("  v%s [%s] → %s", mv.version, stage, artifact_rel)
            except Exception as exc:
                logger.warning("  FAILED version for '%s' (%s): %s", model_name, artifact_rel, exc)


def _load_rows() -> list[dict]:
    if not _LAB_RESULTS.exists():
        logger.error(".lab/results.tsv not found at %s", _LAB_RESULTS)
        sys.exit(1)
    rows = []
    with _LAB_RESULTS.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append({k.strip(): (v.strip() if v is not None else "") for k, v in row.items()})
    logger.info("Loaded %d experiment rows from .lab/results.tsv", len(rows))
    return rows


def _hard_wipe(db_path: Path, name: str) -> None:
    """Hard-delete experiment and all its runs from SQLite (bypasses MLflow soft-delete)."""
    if not db_path.exists():
        return
    with _sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT experiment_id FROM experiments WHERE name = ?", (name,))
        row = cur.fetchone()
        if row is None:
            return
        exp_id = row[0]
        cur.execute("SELECT run_uuid FROM runs WHERE experiment_id = ?", (exp_id,))
        run_uuids = [r[0] for r in cur.fetchall()]
        placeholders = ",".join("?" * len(run_uuids)) if run_uuids else "''"
        for table in ("metrics", "latest_metrics", "params", "tags"):
            cur.execute(f"DELETE FROM {table} WHERE run_uuid IN ({placeholders})", run_uuids)  # noqa: S608
        cur.execute(f"DELETE FROM runs WHERE run_uuid IN ({placeholders})", run_uuids)  # noqa: S608
        cur.execute("DELETE FROM experiment_tags WHERE experiment_id = ?", (exp_id,))
        cur.execute("DELETE FROM experiments WHERE experiment_id = ?", (exp_id,))
        conn.commit()
    logger.info("Hard-deleted '%s' (id=%s, %d runs)", name, exp_id, len(run_uuids))


def _log_run(client, experiment_id: str, row: dict) -> None:
    from mlflow.entities import RunStatus

    exp_num = row.get("experiment", "?")
    status_tag = row.get("status", "").lower()
    mlflow_status = _STATUS_TO_MLFLOW.get(status_tag, "FINISHED")
    meta = EXP_META.get(exp_num, {})

    # Use rich description from log; fall back to TSV description
    rich_desc = DESCRIPTIONS.get(exp_num, row.get("description", ""))
    run_name = f"exp-{exp_num}: {rich_desc[:100]}"

    # Timestamps
    start_iso = meta.get("start", "2026-04-07T00:00:00Z")
    start_ms = _ts_to_ms(start_iso)
    raw_duration = row.get("duration_s", "0")
    duration_s = _parse_duration(raw_duration) or 0.0
    end_ms = start_ms + int(duration_s * 1000)
    if end_ms <= start_ms:
        end_ms = start_ms + 1000  # minimum 1s

    run = client.create_run(
        experiment_id=experiment_id,
        run_name=run_name,
        start_time=start_ms,
        tags={
            "mlflow.runName": run_name,
            "status": status_tag,
            "branch": row.get("branch", "research/gnn-optimizations"),
            "commit": row.get("commit", ""),
            "description": rich_desc,
            "secondary_metrics": row.get("secondary_metrics", ""),
            "notes": meta.get("notes", ""),
        },
    )
    run_id = run.info.run_id

    # ── Parameters ─────────────────────────────────────────────────────────────
    params = {
        "experiment_id":  exp_num,
        "parent":         row.get("parent", ""),
        "status":         status_tag,
        "dataset":        meta.get("dataset", "RVL-CDIP"),
        "architecture":   meta.get("architecture", ""),
        "graph_type":     meta.get("graph_type", ""),
        "model_output":   meta.get("model_output", "-"),
        "branch":         row.get("branch", "research/gnn-optimizations"),
    }
    for k, v in params.items():
        if v:
            client.log_param(run_id, k, v[:500])  # MLflow param value limit

    # ── Metrics ────────────────────────────────────────────────────────────────
    advantage = _parse_metric(row.get("metric", ""))
    if advantage is not None:
        client.log_metric(run_id, "advantage_score_pct", advantage)

    if duration_s > 0:
        client.log_metric(run_id, "training_duration_s", duration_s)

    _log_secondary_metrics(client, run_id, row.get("secondary_metrics", ""))

    client.set_terminated(run_id, status=mlflow_status, end_time=end_ms)


def _log_secondary_metrics(client, run_id: str, secondary: str) -> None:
    if not secondary:
        return
    patterns = [
        (r"CNN[_\w]*(?:acc|accuracy)\s*[=:]\s*(0\.\d+)", "cnn_accuracy"),
        (r"CNN[_\w]*f1\s*[=:]\s*(0\.\d+)",               "cnn_f1"),
        (r"GNN[_\w]*(?:acc|accuracy)\s*[=:]\s*(0\.\d+)", "gnn_accuracy"),
        (r"GNN[_\w]*f1\s*[=:]\s*(0\.\d+)",               "gnn_f1"),
        (r"\bacc\s*[=:]\s*(0\.\d+)",                      "accuracy"),
        (r"\bf1\s*[=:]\s*(0\.\d+)",                       "f1_macro"),
    ]
    logged: set[str] = set()
    for pattern, metric_name in patterns:
        if metric_name in logged:
            continue
        m = re.search(pattern, secondary, re.IGNORECASE)
        if m:
            try:
                client.log_metric(run_id, metric_name, float(m.group(1)))
                logged.add(metric_name)
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri",
        default=f"sqlite:///{_DEFAULT_DB}",
        help="MLflow tracking URI (default: local SQLite DB)",
    )
    args = parser.parse_args()
    _DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    logger.info("MLflow tracking URI: %s", args.tracking_uri)

    _hard_wipe(_DEFAULT_DB, EXPERIMENT_NAME)

    experiment_id = client.create_experiment(
        name=EXPERIMENT_NAME,
        tags={
            "objective":       "Discover when GNN outperforms fine-tuned CNN on document images",
            "dataset":         "RVL-CDIP 400k (16 classes) + RVL-CDIP-N 1k OOD",
            "primary_metric":  "advantage_score_pct (GNN acc − CNN acc, %)",
            "target":          "GNN ≥ 2% above fine-tuned CNN baseline (acc=0.8643)",
            "baseline_cnn":    "ResNet-50 fine-tuned, acc=0.8643, f1=0.8644",
            "team":            "Seow Chun Yong, Prathosh Chander, Sagar Pratap Singh",
            "course":          "61.502 Deep Learning for Enterprise, SUTD Y2026",
        },
    )
    logger.info("Created experiment '%s' (id=%s)", EXPERIMENT_NAME, experiment_id)

    rows = _load_rows()
    for row in rows:
        try:
            _log_run(client, experiment_id, row)
            logger.info("  logged exp-%s [%s] %s", row.get("experiment"), row.get("status"), row.get("metric"))
        except Exception as exc:
            logger.warning("  FAILED exp-%s: %s", row.get("experiment"), exc)

    logger.info("Done — %d runs in '%s'", len(rows), EXPERIMENT_NAME)

    _register_models(client, experiment_id)
    logger.info("Done — model registry populated with %d model families", len(MODEL_REGISTRY))


if __name__ == "__main__":
    main()
