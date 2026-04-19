"""Multimodal GAT inference pipeline: YOLO region detection + ResNet + OCR + MiniLM."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pytesseract
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet50_Weights
from torchvision import transforms as T

from app.src.config import (
    GAT_K_NEIGHBORS,
    GAT_MAX_REGIONS,
    TEXT_ENCODER_MODEL,
    YOLO_CONF,
    YOLO_FILENAME,
    YOLO_REPO_ID,
)

TEXT_EMBED_DIM = 384
NUM_LAYOUT_CLASSES = 11

LAYOUT_CLASS_NAMES: dict[int, str] = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
}

LAYOUT_CLASS_COLORS: dict[int, str] = {
    0: "#e6194b",   # Caption
    1: "#3cb44b",   # Footnote
    2: "#ffe119",   # Formula
    3: "#4363d8",   # List-item
    4: "#f58231",   # Page-footer
    5: "#911eb4",   # Page-header
    6: "#42d4f4",   # Picture
    7: "#f032e6",   # Section-header
    8: "#bfef45",   # Table
    9: "#c8a882",   # Text (darkened for readability)
    10: "#469990",  # Title
}

_yoloModel = None
_textEncoder = None


def _loadYolo():
    global _yoloModel
    if _yoloModel is None:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        modelPath = hf_hub_download(repo_id=YOLO_REPO_ID, filename=YOLO_FILENAME)
        _yoloModel = YOLO(modelPath)
    return _yoloModel


def _loadTextEncoder():
    global _textEncoder
    if _textEncoder is None:
        from sentence_transformers import SentenceTransformer

        _textEncoder = SentenceTransformer(TEXT_ENCODER_MODEL)
        _textEncoder.max_seq_length = 128
    return _textEncoder


def detectRegions(image: Image.Image) -> tuple[list, int, int]:
    """Run YOLO DocLayNet on image and return detected region boxes.

    Returns:
        boxes: [(x, y, w, h, class_id), ...] in pixel coordinates, capped at GAT_MAX_REGIONS
        img_w, img_h: image dimensions
    """
    yolo = _loadYolo()
    imgRgb = image.convert("RGB") if image.mode != "RGB" else image
    imgW, imgH = imgRgb.size

    results = yolo(imgRgb, conf=YOLO_CONF, verbose=False)
    boxes = []
    for box in results[0].boxes:
        x, y, x2, y2 = box.xyxy[0].tolist()
        w, h = x2 - x, y2 - y
        clsId = int(box.cls[0].item())
        if w > 5 and h > 5:
            boxes.append((int(x), int(y), int(w), int(h), clsId))

    return boxes[:GAT_MAX_REGIONS], imgW, imgH


def _cropRegions(image: Image.Image, boxes: list) -> list[Image.Image]:
    """Crop each detected region. Last crop is the full image (global node)."""
    imgRgb = image.convert("RGB") if image.mode != "RGB" else image
    imgW, imgH = imgRgb.size
    crops = []

    for x, y, w, h, _ in boxes:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(imgW, x + w)
        y2 = min(imgH, y + h)
        crops.append(imgRgb.crop((x1, y1, x2, y2)))

    crops.append(imgRgb)  # global node = full image
    return crops


def _runResNetOnCrops(
    crops: list[Image.Image],
    featureExtractor,
    device: torch.device,
) -> torch.Tensor:
    """Run ResNet avgpool on each crop. Returns [N, 2048]."""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensors = [transform(crop) for crop in crops]
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        featOut = featureExtractor(batch)
        avgpool = featOut["avgpool"].squeeze(-1).squeeze(-1)  # [N, 2048]

    return avgpool.cpu()


def _runOcr(crops: list[Image.Image]) -> list[str]:
    """Run Tesseract OCR on each crop."""
    texts = []
    for crop in crops:
        try:
            text = pytesseract.image_to_string(crop, timeout=10).strip()
        except Exception:
            text = ""
        texts.append(text)
    return texts


def _encodeTexts(texts: list[str]) -> torch.Tensor:
    """Encode texts via MiniLM. Returns [N, 384]."""
    encoder = _loadTextEncoder()
    embeddings = np.zeros((len(texts), TEXT_EMBED_DIM), dtype=np.float32)
    nonEmptyMask = [bool(t) for t in texts]
    nonEmptyTexts = [t for t in texts if t]

    if nonEmptyTexts:
        encoded = encoder.encode(nonEmptyTexts, batch_size=64, show_progress_bar=False)
        idx = 0
        for i, hasText in enumerate(nonEmptyMask):
            if hasText:
                embeddings[i] = encoded[idx]
                idx += 1

    return torch.tensor(embeddings, dtype=torch.float32)


def _buildNodeFeatures(
    resnetFeats: torch.Tensor,
    textEmbeds: torch.Tensor,
    boxes: list,
    imgW: int,
    imgH: int,
) -> torch.Tensor:
    """Concatenate per-node features into [N+1, 2447] tensor.

    Layout per node: [ResNet 2048 | MiniLM 384 | bbox 4 | layout_onehot 11]
    """
    bboxFeats = []
    classOneHots = []

    for x, y, w, h, clsId in boxes:
        bboxFeats.append([x / imgW, y / imgH, w / imgW, h / imgH])
        onehot = [0.0] * NUM_LAYOUT_CLASSES
        if 0 <= clsId < NUM_LAYOUT_CLASSES:
            onehot[clsId] = 1.0
        classOneHots.append(onehot)

    # Global node: bbox = full image, no layout class
    bboxFeats.append([0.0, 0.0, 1.0, 1.0])
    classOneHots.append([0.0] * NUM_LAYOUT_CLASSES)

    bboxTensor = torch.tensor(bboxFeats, dtype=torch.float32)
    onehotTensor = torch.tensor(classOneHots, dtype=torch.float32)

    return torch.cat([resnetFeats, textEmbeds, bboxTensor, onehotTensor], dim=1)


def _buildKnnEdges(boxes: list, k: int, imgW: int, imgH: int) -> torch.Tensor:
    """k-NN edges from region centroids + global node connected to all.

    Returns undirected edge_index [2, E].
    """
    nRegions = len(boxes)
    globalIdx = nRegions

    if nRegions == 0:
        return torch.tensor([[0], [0]], dtype=torch.long)

    centroids = np.array([
        [(x + w / 2) / imgW, (y + h / 2) / imgH]
        for x, y, w, h, _ in boxes
    ])

    kActual = min(k, nRegions - 1)
    edgeSet: set[tuple[int, int]] = set()

    if kActual > 0:
        diffs = centroids[:, None, :] - centroids[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)

        for i in range(nRegions):
            for j in np.argsort(dists[i])[:kActual]:
                s, d = int(i), int(j)
                edgeSet.add((min(s, d), max(s, d)))

    # Global node ↔ all region nodes
    for i in range(nRegions):
        edgeSet.add((min(i, globalIdx), max(i, globalIdx)))

    srcList, dstList = [], []
    for s, d in edgeSet:
        srcList.extend([s, d])
        dstList.extend([d, s])

    if not srcList:
        return torch.tensor([[0], [0]], dtype=torch.long)

    return torch.tensor([srcList, dstList], dtype=torch.long)


@torch.no_grad()
def runMultimodalGatInference(
    image: Image.Image,
    model: torch.nn.Module,
    featureExtractor,
    device: torch.device,
) -> tuple[torch.Tensor, dict, float]:
    """Full multimodal GAT inference pipeline.

    Returns:
        probs: [16] softmax probabilities
        graphMeta: dict with boxes, layout_classes, edge_index, ocr_texts, img_w, img_h
        elapsedMs: total inference time in milliseconds
    """
    t0 = time.perf_counter()

    boxes, imgW, imgH = detectRegions(image)
    crops = _cropRegions(image, boxes)
    resnetFeats = _runResNetOnCrops(crops, featureExtractor, device)
    ocrTexts = _runOcr(crops)
    textEmbeds = _encodeTexts(ocrTexts)
    nodeFeats = _buildNodeFeatures(resnetFeats, textEmbeds, boxes, imgW, imgH)
    edgeIndex = _buildKnnEdges(boxes, GAT_K_NEIGHBORS, imgW, imgH)

    nTotal = nodeFeats.shape[0]
    batchVec = torch.zeros(nTotal, dtype=torch.long)

    logits = model(nodeFeats.to(device), edgeIndex.to(device), batchVec.to(device))
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    elapsedMs = (time.perf_counter() - t0) * 1000

    graphMeta = {
        "boxes": boxes,
        "layout_classes": [clsId for _, _, _, _, clsId in boxes],
        "edge_index": edgeIndex,
        "ocr_texts": ocrTexts[:len(boxes)],  # region nodes only
        "img_w": imgW,
        "img_h": imgH,
    }

    return probs, graphMeta, elapsedMs
