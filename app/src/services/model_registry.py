"""Model registry: ModelSpec definitions and checkpoint loading for all 4 final models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from app.src.config import MODEL_DIR
from app.src.model import (
    DocumentGAT,
    FusionGraphSAGE,
    InductiveGCN,
)


@dataclass
class ModelSpec:
    name: str
    display_name: str
    model_type: str  # "CNN" | "GNN" | "GNN+OCR"
    model_class: Optional[Type[nn.Module]]
    constructor_kwargs: dict = field(default_factory=dict)
    checkpoint_path: str = ""
    graph_type: str = "none"  # "none" | "feature_knn" | "doc_knn" | "gated_boc"
    needs_global_feat: bool = False
    needs_pe: bool = False
    needs_boc: bool = False
    needs_text_density: bool = False


MODEL_SPECS = [
    ModelSpec(
        name="cnn_baseline",
        display_name="CNN Baseline (ResNet-50)",
        model_type="CNN",
        model_class=None,
        checkpoint_path="best_model_resnet50_03.pt",
        graph_type="none",
    ),
    ModelSpec(
        name="fusion_graphsage",
        display_name="Fusion GraphSAGE",
        model_type="GNN",
        model_class=FusionGraphSAGE,
        constructor_kwargs={"in_channels": 2048},
        checkpoint_path="fusion_gnn_feat_knn_best.pt",
        graph_type="feature_knn",
        needs_global_feat=True,
    ),
    ModelSpec(
        name="inductive_gcn",
        display_name="Inductive GCN",
        model_type="GNN",
        model_class=InductiveGCN,
        constructor_kwargs={},
        checkpoint_path="inductive_gcn_320k.pt",
        graph_type="doc_knn",
        needs_global_feat=False,
    ),
]


GCN_FEATURE_BANK_PATH = Path(__file__).resolve().parent.parent / "data" / "gcn_feature_bank.pt"


def _load_gcn_feature_bank(device: torch.device) -> Optional[dict]:
    """Load the document feature bank used by the Inductive GCN for single-doc inference."""
    if not GCN_FEATURE_BANK_PATH.exists():
        print(f"Warning: GCN feature bank not found at {GCN_FEATURE_BANK_PATH}")
        print("         Build it with: poetry run python scripts/build_gcn_feature_bank.py")
        return None

    bundle = torch.load(GCN_FEATURE_BANK_PATH, map_location="cpu", weights_only=False)
    bank = {
        "features": bundle["features"].to(device),         # [N, 2048] L2-normalized
        "labels": bundle["labels"].to(device),             # [N] int64
        "doc_ids": bundle["doc_ids"],                      # [N] int64 (HF dataset indices)
        "thumb_dir": bundle["thumb_dir"],
        "thumb_paths": bundle["thumb_paths"],
        "num_classes": bundle["num_classes"],
        "per_class": bundle["per_class"],
    }
    print(f"Loaded GCN feature bank: {tuple(bank['features'].shape)} on {device}")
    return bank


def _unwrap_resnet_checkpoint(checkpoint) -> dict:
    """Normalize checkpoint formats to a plain ResNet-50 state dict."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")

    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_all_models(device: torch.device) -> dict:
    """Load all model checkpoints and return a structured registry dict.

    Returns:
        {
            "feature_extractor": torchvision feature extractor (layer4 + avgpool),
            "resnet50_classifier": full ResNet-50 with fc head (for CNN baseline),
            "models": {spec.name: (spec, loaded_model) for each GNN},
        }
    """
    # Load CNN baseline checkpoint (ResNet-50 with fc head)
    cnn_checkpoint_path = MODEL_DIR / "best_model_resnet50_03.pt"
    checkpoint = torch.load(cnn_checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_resnet_checkpoint(checkpoint)

    num_classes = state_dict["fc.weight"].shape[0]

    # Build full ResNet-50 classifier (CNN baseline)
    resnet = resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet.load_state_dict(state_dict)
    resnet.eval()
    resnet = resnet.to(device)

    # Build feature extractor from the same fine-tuned weights
    feature_extractor = create_feature_extractor(
        resnet,
        return_nodes={"layer4": "layer4", "avgpool": "avgpool"},
    )
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor = feature_extractor.to(device)

    # Load GNN models
    gnn_models = {}
    for spec in MODEL_SPECS:
        if spec.model_class is None:
            continue  # CNN baseline handled above

        ckpt_path = MODEL_DIR / spec.checkpoint_path
        if not ckpt_path.exists():
            print(f"Warning: checkpoint not found for {spec.display_name}: {ckpt_path}")
            continue

        model = spec.model_class(**spec.constructor_kwargs)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt)
        model.eval()
        model = model.to(device)

        gnn_models[spec.name] = (spec, model)

    # Load multimodal GAT
    multimodal_gat_model = None
    multimodal_gat_path = MODEL_DIR / "best_gat_multimodal_k8_L2.pt"
    if multimodal_gat_path.exists():
        multimodal_gat_model = DocumentGAT(
            in_dim=2447,
            hidden_dim=256,
            out_dim=num_classes,
            heads=4,
            num_layers=2,
            dropout=0.2,
            pooling="mean",
        )
        ckpt = torch.load(multimodal_gat_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        multimodal_gat_model.load_state_dict(state_dict)
        multimodal_gat_model.eval()
        multimodal_gat_model = multimodal_gat_model.to(device)
        print(f"Loaded DocumentGAT (multimodal) from {multimodal_gat_path.name}")
    else:
        print(f"Warning: multimodal GAT checkpoint not found at {multimodal_gat_path}")

    print(f"Loaded CNN baseline + {len(gnn_models)} GNN models on {device}")

    # Feature bank for Inductive GCN single-doc inference (query connects into bank via kNN).
    gcn_feature_bank = _load_gcn_feature_bank(device)

    # Pre-load doctr text density detector once at startup.
    # Loading it per-request causes a hang (model weights fetched/loaded on every call).
    text_detector = None
    try:
        from src.text_features import create_text_detector
        text_detector = create_text_detector(type("Config", (), {"device": device})())
        print("Loaded doctr text density detector")
    except Exception as exc:
        print(f"Text density detector unavailable (will skip): {exc}")

    return {
        "feature_extractor": feature_extractor,
        "resnet50_classifier": resnet,
        "models": gnn_models,
        "multimodal_gat_model": multimodal_gat_model,
        "text_detector": text_detector,
        "gcn_feature_bank": gcn_feature_bank,
    }
