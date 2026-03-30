"""Text density extraction from document images using doctr DBNet."""

import ssl
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def _get_detection_device(config_device: torch.device) -> torch.device:
    """Resolve the device for doctr text detection.

    doctr's DBNet does not support MPS. Fall back to CPU when MPS is selected.

    Args:
        config_device: Project-level device (may be MPS, CUDA, or CPU)

    Returns:
        Device safe for doctr inference
    """
    if config_device.type == "mps":
        return torch.device("cpu")
    return config_device


def create_text_detector(config: Any):
    """Create and configure a doctr DBNet text detection predictor.

    Uses db_mobilenet_v3_large: lightweight architecture suitable for
    per-image cache augmentation on a Mac with MPS.

    Args:
        config: Config object with .device attribute

    Returns:
        Frozen doctr DetectionPredictor in eval mode, placed on detection device
    """
    # SSL workaround for environments with corporate/self-signed certs
    ssl._create_default_https_context = ssl._create_unverified_context

    from doctr.models import detection_predictor

    detection_device = _get_detection_device(config.device)

    predictor = detection_predictor(
        arch="db_mobilenet_v3_large",
        pretrained=True,
        assume_straight_pages=True,
    )

    predictor = predictor.to(detection_device)
    predictor.eval()

    for param in predictor.parameters():
        param.requires_grad_(False)

    return predictor


def extract_text_density(
    image: Image.Image,
    detector,
    device: torch.device,
) -> torch.Tensor:
    """Extract a [7, 7] text density heatmap from a PIL image.

    Runs the doctr DBNet detector in raw probability-map mode
    (return_model_output=True) to obtain the per-pixel text probability,
    then downsamples to a 7x7 grid matching ResNet-50 layer4 spatial dimensions.

    Args:
        image: PIL Image (grayscale or RGB), any size
        detector: doctr DetectionPredictor (from create_text_detector)
        device: torch.device used by the project (used only for output placement)

    Returns:
        Tensor [7, 7], float32, values in [0, 1]
    """
    detection_device = _get_detection_device(device)

    # Convert PIL image to uint8 RGB numpy array for doctr
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_np = np.array(image, dtype=np.uint8)

    with torch.no_grad():
        # Pre-process image -> [1, 3, H', W'] batch tensor
        processed = detector.pre_processor([img_np])
        batch_tensor = processed[0].to(detection_device)

        # Run underlying DBNet model with raw probability map output
        raw_out = detector.model(
            batch_tensor,
            return_model_output=True,
            return_preds=False,
        )
        # out_map shape: [1, 1, H', W'] — sigmoid probability of text
        prob_map = raw_out["out_map"]  # [1, 1, H', W']

    # Downsample to [1, 1, 7, 7] then squeeze to [7, 7]
    density = F.adaptive_avg_pool2d(prob_map, output_size=(7, 7))
    density = density.squeeze(0).squeeze(0).float()  # [7, 7]

    # Move to project device
    return density.to(device)


def augment_cache_with_text_density(
    config: Any,
    split: str,
    detector,
    samples: List[dict],
) -> None:
    """Augment cached .pt feature files with a text_density key.

    Loads each .pt file in config.cache_dir / split, checks for an existing
    text_density key, and if missing extracts it from the corresponding PIL
    image in samples and re-saves the file.

    Args:
        config: Config object with .cache_dir and .device attributes
        split: Dataset split name (e.g., "train", "validation", "test")
        detector: doctr DetectionPredictor (from create_text_detector)
        samples: List of dicts indexed identically to the cached .pt files.
                 Each dict must contain an "image" key with a PIL Image value.
    """
    split_dir = Path(config.cache_dir) / split
    pt_files = sorted(split_dir.glob("*.pt"))

    if len(pt_files) != len(samples):
        raise ValueError(
            f"Mismatch: {len(pt_files)} cached files but {len(samples)} samples "
            f"in split '{split}'"
        )

    for i, pt_path in enumerate(tqdm(pt_files, desc=f"Augmenting {split} with text density")):
        data = torch.load(pt_path, weights_only=False)

        if "text_density" in data:
            continue

        pil_image = samples[i]["image"]
        density = extract_text_density(pil_image, detector, config.device)
        data["text_density"] = density

        torch.save(data, pt_path)
