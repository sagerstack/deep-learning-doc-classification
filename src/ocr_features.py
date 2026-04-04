"""OCR feature extraction using Tesseract for document image classification.

Extracts word-level OCR from document images using Tesseract, maps words
to a 7x7 spatial grid matching ResNet-50 layer4 dimensions, and computes
Bag-of-Characters (BoC) feature vectors per grid cell.

Pipeline:
    1. run_tesseract_single() — OCR one image → list of (text, bbox) tuples
    2. compute_boc_features() — map words to 7x7 grid → [49, NUM_CHARS] tensor
    3. augment_cache_with_ocr() — add ocr_boc key to cached .pt files
"""

import string
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# Character vocabulary: a-z (26) + A-Z (26) + 0-9 (10) + punctuation (8) = 70
_LOWERCASE = string.ascii_lowercase          # 26
_UPPERCASE = string.ascii_uppercase          # 26
_DIGITS = string.digits                      # 10
_PUNCTUATION = ".,;:!?-/"                    # 8
CHAR_VOCAB = _LOWERCASE + _UPPERCASE + _DIGITS + _PUNCTUATION
NUM_CHARS = len(CHAR_VOCAB)  # 70
_CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_VOCAB)}


def run_tesseract_single(image: Image.Image) -> List[Dict]:
    """Run Tesseract OCR on a single PIL image.

    Returns a list of word dicts with keys: text, confidence, x, y, w, h.
    Bounding boxes are in pixel coordinates. Filters to confidence > 0
    and non-empty text.
    """
    import pytesseract

    if image.mode != "RGB":
        image = image.convert("RGB")

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words = []
    for i in range(len(data["text"])):
        conf = int(data["conf"][i])
        text = data["text"][i].strip()
        if conf > 0 and text:
            words.append({
                "text": text,
                "confidence": conf,
                "x": data["left"][i],
                "y": data["top"][i],
                "w": data["width"][i],
                "h": data["height"][i],
            })
    return words


def _word_to_grid_cell(word: Dict, img_w: int, img_h: int, grid_h: int = 7, grid_w: int = 7) -> Tuple[int, int]:
    """Map a word's bounding box center to a grid cell (row, col)."""
    cx = word["x"] + word["w"] / 2
    cy = word["y"] + word["h"] / 2

    col = int(cx / img_w * grid_w)
    row = int(cy / img_h * grid_h)

    col = min(max(col, 0), grid_w - 1)
    row = min(max(row, 0), grid_h - 1)
    return row, col


def compute_boc_features(
    words: List[Dict],
    img_w: int,
    img_h: int,
    grid_h: int = 7,
    grid_w: int = 7,
) -> torch.Tensor:
    """Compute Bag-of-Characters feature tensor from OCR words.

    Maps each word to a 7x7 grid cell based on bounding box center,
    then counts character frequencies per cell.

    Args:
        words: List of word dicts from run_tesseract_single()
        img_w: Image width in pixels
        img_h: Image height in pixels
        grid_h: Grid height (default 7)
        grid_w: Grid width (default 7)

    Returns:
        Tensor [grid_h * grid_w, NUM_CHARS] (i.e., [49, 70]), float32, L1-normalized per cell.
        Cells with no text have all-zero vectors.
    """
    num_nodes = grid_h * grid_w
    boc = np.zeros((num_nodes, NUM_CHARS), dtype=np.float32)

    for word in words:
        row, col = _word_to_grid_cell(word, img_w, img_h, grid_h, grid_w)
        node_idx = row * grid_w + col

        for char in word["text"]:
            idx = _CHAR_TO_IDX.get(char)
            if idx is not None:
                boc[node_idx, idx] += 1

    # L1-normalize per cell (sum of counts → 1.0, or 0 if empty)
    row_sums = boc.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    boc /= row_sums

    return torch.from_numpy(boc)


def process_single_image(args: Tuple) -> Tuple[int, torch.Tensor]:
    """Process a single image: OCR → BoC features. Used by thread pool.

    Args:
        args: Tuple of (index, image_path_or_pil, img_size)
              where img_size is (width, height)

    Returns:
        Tuple of (index, boc_tensor [49, 70])
    """
    idx, image, img_size = args
    img_w, img_h = img_size

    try:
        words = run_tesseract_single(image)
        boc = compute_boc_features(words, img_w, img_h)
    except Exception:
        # On failure, return zeros (graceful degradation)
        boc = torch.zeros(49, NUM_CHARS)

    return idx, boc


def augment_cache_with_ocr(
    cache_dir: Path,
    split: str,
    dataset,
    num_workers: int = 8,
    batch_size: int = 200,
) -> Dict[str, Any]:
    """Augment cached .pt feature files with ocr_boc key.

    Processes images through Tesseract OCR in parallel using ThreadPoolExecutor
    (Tesseract is a subprocess, so threads release the GIL), computes BoC features,
    and saves them into existing cache files.

    Args:
        cache_dir: Path to cached features (e.g., .hf-cache-100k/cached_features)
        split: Dataset split name (train, validation, test)
        dataset: HuggingFace dataset split with image column
        num_workers: Number of parallel Tesseract workers
        batch_size: Process and save in batches to limit memory

    Returns:
        Dict with stats: total, augmented, skipped, errors
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    split_dir = cache_dir / split
    pt_files = sorted(split_dir.glob("*.pt"))

    if len(pt_files) != len(dataset):
        raise ValueError(
            f"Mismatch: {len(pt_files)} cached files but {len(dataset)} samples "
            f"in split '{split}'"
        )

    stats = {"total": len(pt_files), "augmented": 0, "skipped": 0, "errors": 0}

    # Process in batches to manage memory
    for batch_start in tqdm(
        range(0, len(pt_files), batch_size),
        desc=f"OCR {split}",
        total=(len(pt_files) + batch_size - 1) // batch_size,
    ):
        batch_end = min(batch_start + batch_size, len(pt_files))
        batch_indices = list(range(batch_start, batch_end))

        # Check which files need augmentation
        indices_to_process = []
        for i in batch_indices:
            data = torch.load(pt_files[i], weights_only=False)
            if "ocr_boc" not in data:
                indices_to_process.append(i)
            else:
                stats["skipped"] += 1

        if not indices_to_process:
            continue

        # Prepare args for parallel processing
        args_list = []
        for i in indices_to_process:
            sample = dataset[i]
            image = sample["image"]
            img_size = image.size  # (width, height)
            args_list.append((i, image, img_size))

        # Run Tesseract in parallel via threads
        results = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_single_image, args): args[0]
                for args in args_list
            }
            for future in as_completed(futures):
                try:
                    idx, boc = future.result()
                    results[idx] = boc
                except Exception:
                    idx = futures[future]
                    results[idx] = torch.zeros(49, NUM_CHARS)
                    stats["errors"] += 1

        # Save results to cache files
        for i in indices_to_process:
            boc = results.get(i, torch.zeros(49, NUM_CHARS))
            data = torch.load(pt_files[i], weights_only=False)
            data["ocr_boc"] = boc
            torch.save(data, pt_files[i])
            stats["augmented"] += 1

    return stats
