"""Data loading and preprocessing for RVL-CDIP dataset."""

from typing import Dict, List

import matplotlib.pyplot as plt
from datasets import load_dataset
from torchvision import transforms

from src.config import Config


# 16 RVL-CDIP class labels in order
RVL_CDIP_LABELS = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific_report",
    "scientific_publication",
    "specification",
    "file_folder",
    "news_article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]


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
        Dict with keys "train", "validation", "test", each containing either:
        - List of dicts (sample mode): [{"image": PIL.Image, "label": int}, ...]
        - DatasetDict splits (full mode): HuggingFace DatasetDict

    Modes:
        - "sample": Streaming mode, shuffled sample of size config.sample_size
        - "full": Full dataset download to config.hf_cache_dir
    """
    if config.mode == "sample":
        print(f"Loading dataset in streaming mode (sample_size={config.sample_size})...")

        # Load with streaming
        dataset = load_dataset(config.dataset_name, streaming=True)

        # For each split, shuffle and take sample_size items
        data = {}
        for split in ["train", "val", "test"]:
            # Shuffle before taking to get random sample
            shuffled = dataset[split].shuffle(
                seed=config.seed,
                buffer_size=10000
            )
            # Take sample and convert to list
            samples = list(shuffled.take(config.sample_size))
            # Store with consistent key name
            key = "validation" if split == "val" else split
            data[key] = samples
            print(f"Loaded {key}: {len(samples)} samples")

        return data

    elif config.mode == "full":
        print(f"Loading full dataset to cache: {config.hf_cache_dir}...")

        # Load full dataset
        dataset = load_dataset(
            config.dataset_name,
            cache_dir=str(config.hf_cache_dir)
        )

        # Print split sizes and normalize keys
        data = {}
        for split in ["train", "val", "test"]:
            key = "validation" if split == "val" else split
            data[key] = dataset[split]
            print(f"Loaded {key}: {len(dataset[split])} samples")

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
    # Limit to available samples
    n = min(n, len(data))

    # Create 2-row grid
    cols = (n + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(cols * 3, 6))
    axes = axes.flatten()

    for i in range(n):
        sample = data[i]
        image = sample["image"]
        label_idx = sample["label"]
        label_name = RVL_CDIP_LABELS[label_idx]

        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(label_name)
        axes[i].axis("off")

    # Hide unused subplots
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
