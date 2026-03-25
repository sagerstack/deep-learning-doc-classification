"""Data loading and preprocessing for RVL-CDIP dataset."""

from typing import Dict, List

import matplotlib.pyplot as plt
from datasets import load_dataset
from torchvision import transforms

from src.config import Config


# Populated in-place by load_rvl_cdip from dataset features.
RVL_CDIP_LABELS: List[str] = []


def get_transform():
    """Get preprocessing transform pipeline.

    Returns:
        Compose: Torchvision transform pipeline that:
            - Resizes to 224x224
            - Converts grayscale to 3 channels for ResNet-50
            - Converts to tensor
            - Normalizes with ImageNet stats
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def load_rvl_cdip(config: Config) -> Dict[str, List]:
    """Load RVL-CDIP dataset based on config mode.

    Args:
        config: Config object with mode, sample_size, dataset_name, etc.

    Returns:
        Dict with keys "train", "validation", and optionally "test",
        each containing either:
        - List of dicts (sample mode): [{"image": PIL.Image, "label": int}, ...]
        - DatasetDict splits (full mode): HuggingFace Dataset objects

    Modes:
        - "sample": Loads RVL-CDIP Small-200 (3,200 images, balanced 200/class).
          Splits: train (2,560) + validation (640). No test split.
        - "full": Full RVL-CDIP dataset download to config.hf_cache_dir.
          Splits: train (320k) + validation (40k) + test (40k).
    """
    if config.mode == "sample":
        print(f"Loading sample dataset: {config.sample_dataset_name}...")

        dataset = load_dataset(
            config.sample_dataset_name,
            cache_dir=str(config.hf_cache_dir),
        )

        RVL_CDIP_LABELS.clear()
        RVL_CDIP_LABELS.extend(dataset["train"].features["label"].names)

        data = {}
        for split_name in dataset.keys():
            key = "validation" if split_name == "val" else split_name
            data[key] = list(dataset[split_name])
            print(f"Loaded {key}: {len(data[key])} samples")

        return data

    elif config.mode == "full":
        print(f"Loading full dataset to cache: {config.hf_cache_dir}...")

        dataset = load_dataset(
            config.dataset_name,
            cache_dir=str(config.hf_cache_dir),
        )

        RVL_CDIP_LABELS.clear()
        RVL_CDIP_LABELS.extend(dataset["train"].features["label"].names)

        data = {}
        for split_name in dataset.keys():
            key = "validation" if split_name == "val" else split_name
            data[key] = dataset[split_name]
            print(f"Loaded {key}: {len(dataset[split_name])} samples")

        return data

    else:
        raise ValueError(f"Unknown mode: {config.mode}. Use 'sample' or 'full'.")


def display_samples(data: List[Dict], n: int = 10):
    """Display sample images in a grid.

    Args:
        data: List of dicts with "image" (PIL.Image) and "label" (int)
        n: Number of samples to display (default 10)

    Returns:
        matplotlib.figure.Figure: The figure object for notebook display
    """
    n = min(n, len(data))

    cols = (n + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(cols * 3, 6))
    axes = axes.flatten()

    for i in range(n):
        sample = data[i]
        image = sample["image"]
        label_idx = sample["label"]
        label_name = RVL_CDIP_LABELS[label_idx] if RVL_CDIP_LABELS else str(label_idx)

        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(label_name)
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig


def apply_transform(samples: List[Dict], transform) -> List[Dict]:
    """Apply preprocessing transform to samples.

    Args:
        samples: List of dicts with "image" (PIL.Image) and "label" (int)
        transform: Torchvision transform to apply

    Returns:
        List of dicts with "image" replaced by tensor [3, 224, 224], "label" kept as int
    """
    transformed = []
    for sample in samples:
        transformed.append({
            "image": transform(sample["image"]),
            "label": sample["label"],
        })
    return transformed
