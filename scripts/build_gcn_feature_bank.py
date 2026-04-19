"""Build the Inductive GCN document-feature bank used by the demo app.

The Inductive GCN was trained on a single k-NN graph of ~320k document-level
ResNet-50 avgpool features. Single-document inference requires a reference
bank of documents to connect the query into — this script builds that bank.

Features are re-extracted through the current ResNet-50 checkpoint with the
app's exact preprocessing pipeline (Grayscale(3)+Resize(224)+ToTensor+
Normalize) to guarantee bank features live in the same space as the live
query feature extractor — NOT copied from .hf-cache-100k/cached_features/
(which has unknown provenance).

Pipeline:
  1. Read label column from the local HF dataset (fast).
  2. Class-balanced sample: PER_CLASS indices per class × 16 classes.
  3. Load each picked image, preprocess, run through ResNet-50 to get
     2048-d avgpool feature; save a 96×96 JPEG thumbnail.
  4. L2-normalize the stacked features.

Output bundle: app/src/data/gcn_feature_bank.pt
Output thumbs: app/src/data/gcn_thumbs/<0000…NNNN>.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet50

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.src.config import MODEL_DIR, RVL_CDIP_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

NUM_CLASSES = 16
DEFAULT_PER_CLASS = 125
THUMB_SIZE = 96
BATCH_SIZE = 32

CACHE_DATASET = _PROJECT_ROOT / ".hf-cache-100k" / "rvl-cdip-100k"
RESNET_CKPT = MODEL_DIR / "best_model_resnet50_03.pt"

OUTPUT_DIR = _PROJECT_ROOT / "app" / "src" / "data"
OUTPUT_BUNDLE = OUTPUT_DIR / "gcn_feature_bank.pt"
OUTPUT_THUMBS = OUTPUT_DIR / "gcn_thumbs"

# Must match app/src/services/inference.py preprocessing exactly
FEATURE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_feature_extractor(device: torch.device) -> nn.Module:
    """Build a ResNet-50 backbone-to-avgpool extractor using the app's checkpoint."""
    log.info("Loading ResNet-50 checkpoint: %s", RESNET_CKPT)
    checkpoint = torch.load(RESNET_CKPT, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    num_classes = state_dict["fc.weight"].shape[0]

    resnet = resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet.load_state_dict(state_dict)
    resnet.eval()

    # Keep backbone + avgpool, drop fc — mirrors the app's feature_extractor usage
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone.to(device)


def _class_balanced_sample(labels: list[int], per_class: int, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    picked_indices: list[int] = []
    picked_labels: list[int] = []

    for cls in range(NUM_CLASSES):
        class_indices = [i for i, lbl in enumerate(labels) if lbl == cls]
        if len(class_indices) < per_class:
            raise RuntimeError(
                f"Class {cls} ({RVL_CDIP_LABELS[cls]}) has only {len(class_indices)} "
                f"samples, need {per_class}."
            )
        perm = torch.randperm(len(class_indices), generator=generator)[:per_class]
        picks = [class_indices[i] for i in perm.tolist()]
        picked_indices.extend(picks)
        picked_labels.extend([cls] * per_class)
        log.info("  class %2d %-25s : picked %d of %d available",
                 cls, RVL_CDIP_LABELS[cls], per_class, len(class_indices))

    return picked_indices, picked_labels


def _save_thumbnail(img: Image.Image, path: Path) -> None:
    thumb = img.convert("RGB") if img.mode != "RGB" else img.copy()
    thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
    thumb.save(path, "JPEG", quality=82, optimize=True)


def _class_slug(label_idx: int) -> str:
    return RVL_CDIP_LABELS[label_idx].replace(" ", "_")


def _thumb_filename(bank_idx: int, label_idx: int) -> str:
    return f"{bank_idx:04d}_{_class_slug(label_idx)}.jpg"


@torch.no_grad()
def _extract_features_and_thumbs(
    ds,
    picked_indices: list[int],
    picked_labels: list[int],
    backbone: nn.Module,
    device: torch.device,
) -> list[torch.Tensor]:
    total = len(picked_indices)
    feats: list[torch.Tensor] = []
    batch_tensors: list[torch.Tensor] = []
    t0 = time.perf_counter()

    def flush():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device, non_blocking=True)
        feat = backbone(x).flatten(1).cpu()  # [B, 2048]
        for i in range(feat.shape[0]):
            feats.append(feat[i])

    for bank_idx, (doc_idx, label_idx) in enumerate(zip(picked_indices, picked_labels)):
        img = ds[doc_idx]["image"]
        rgb = img.convert("RGB") if img.mode != "RGB" else img

        batch_tensors.append(FEATURE_TRANSFORM(rgb))

        _save_thumbnail(img, OUTPUT_THUMBS / _thumb_filename(bank_idx, label_idx))

        if len(batch_tensors) == BATCH_SIZE:
            flush()
            batch_tensors = []

        if (bank_idx + 1) % 200 == 0:
            log.info("  ...%d/%d (%.0fs)", bank_idx + 1, total, time.perf_counter() - t0)

    flush()
    return feats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not CACHE_DATASET.exists():
        log.error("HF dataset cache not found: %s", CACHE_DATASET)
        sys.exit(1)
    if not RESNET_CKPT.exists():
        log.error("ResNet checkpoint not found: %s", RESNET_CKPT)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_THUMBS.mkdir(parents=True, exist_ok=True)

    # Clear any stale thumbnails
    for old in OUTPUT_THUMBS.glob("*.jpg"):
        old.unlink()

    # Load HF dataset (label column is cheap, images fetched lazily per-index)
    log.info("Loading HF dataset from %s", CACHE_DATASET)
    ds = load_from_disk(str(CACHE_DATASET))["train"]
    log.info("  %d training items", len(ds))

    # Pull labels from HF for class-balanced sampling (cached column, fast)
    log.info("Reading label column from HF dataset...")
    all_labels = list(ds["label"])

    log.info("Class-balanced sampling: %d per class × %d classes = %d docs",
             args.per_class, NUM_CLASSES, args.per_class * NUM_CLASSES)
    picked_indices, picked_labels = _class_balanced_sample(all_labels, args.per_class, args.seed)

    # Live feature extractor — matches app's inference pipeline exactly
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    log.info("Feature extractor device: %s", device)
    backbone = _load_feature_extractor(device)

    total = len(picked_indices)
    log.info("Extracting %d features + thumbnails through the live pipeline...", total)

    features_list = _extract_features_and_thumbs(ds, picked_indices, picked_labels, backbone, device)
    assert len(features_list) == total, f"Got {len(features_list)} features, expected {total}"

    thumb_paths = [_thumb_filename(i, picked_labels[i]) for i in range(total)]

    features = torch.stack(features_list, dim=0)                     # [N, 2048]
    features = F.normalize(features, p=2, dim=1)

    bundle = {
        "features": features,
        "labels": torch.tensor(picked_labels, dtype=torch.long),
        "doc_ids": torch.tensor(picked_indices, dtype=torch.long),
        "thumb_dir": "gcn_thumbs",
        "thumb_paths": thumb_paths,
        "per_class": args.per_class,
        "num_classes": NUM_CLASSES,
        "seed": args.seed,
    }
    torch.save(bundle, OUTPUT_BUNDLE)

    size_mb = OUTPUT_BUNDLE.stat().st_size / 1e6
    log.info("Wrote bundle: %s (%.2f MB)", OUTPUT_BUNDLE, size_mb)
    log.info("  features: %s  labels: %s", tuple(features.shape), tuple(bundle["labels"].shape))
    log.info("  thumbnails: %d × %dpx in %s", total, THUMB_SIZE, OUTPUT_THUMBS)


if __name__ == "__main__":
    main()
