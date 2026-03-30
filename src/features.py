"""ResNet-50 feature extraction and disk caching for RVL-CDIP."""

from pathlib import Path
from typing import Dict, List

import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from src.config import Config


def create_resnet_extractor(config: Config):
    """Create frozen ResNet-50 feature extractor outputting layer4 and avgpool.

    Args:
        config: Config object with device

    Returns:
        Feature extractor model that returns layer4 [B, 2048, 7, 7] + avgpool [B, 2048, 1, 1]
    """
    # Load ResNet-50 with modern weights API (NOT deprecated pretrained=True)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Extract layer4 output (spatial) and avgpool (global context)
    extractor = create_feature_extractor(
        model,
        return_nodes={"layer4": "layer4", "avgpool": "avgpool"}
    )

    # Set to eval mode (critical for BatchNorm/Dropout) and freeze all parameters
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False

    # Move to device
    extractor = extractor.to(config.device)

    # Print model info
    num_params = sum(p.numel() for p in extractor.parameters())
    print(f"ResNet-50 feature extractor loaded")
    print(f"Parameters: {num_params:,} (all frozen)")
    print(f"Output: layer4 [B, 2048, 7, 7] + avgpool [B, 2048, 1, 1]")
    print(f"Device: {config.device}")

    return extractor


def create_extractor_from_trained(model, config: Config):
    """Create layer4 + avgpool feature extractor from an already-trained ResNet-50.

    Unlike create_resnet_extractor which loads fresh pretrained weights,
    this wraps an existing fine-tuned model to extract its adapted features.

    Args:
        model: Trained ResNet-50 model (with fine-tuned backbone)
        config: Config object with device

    Returns:
        Feature extractor that returns layer4 [B, 2048, 7, 7] + avgpool [B, 2048, 1, 1]
    """
    extractor = create_feature_extractor(
        model,
        return_nodes={"layer4": "layer4", "avgpool": "avgpool"}
    )
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False
    extractor = extractor.to(config.device)

    num_params = sum(p.numel() for p in extractor.parameters())
    print(f"Fine-tuned feature extractor loaded")
    print(f"Parameters: {num_params:,} (all frozen for extraction)")
    print(f"Output: layer4 [B, 2048, 7, 7] + avgpool [B, 2048, 1, 1]")
    print(f"Device: {config.device}")

    return extractor


def extract_features_batch(extractor, images: torch.Tensor, device):
    """Extract layer4 spatial features and avgpool global features from a batch.

    Args:
        extractor: Feature extractor model from create_resnet_extractor
        images: Tensor [B, 3, 224, 224] of preprocessed images
        device: Device to run extraction on

    Returns:
        Tuple of (layer4_features [B, 2048, 7, 7], avgpool_features [B, 2048])
        Both on CPU for device-agnostic caching
    """
    images = images.to(device)

    with torch.no_grad():
        output = extractor(images)
        layer4_features = output["layer4"]
        avgpool_features = output["avgpool"].squeeze(-1).squeeze(-1)  # [B, 2048, 1, 1] -> [B, 2048]

    # Move to CPU for device-agnostic caching
    return layer4_features.cpu(), avgpool_features.cpu()


def cache_features(samples: List[Dict], extractor, transform, config: Config, split: str):
    """Extract and cache features for all samples to disk.

    Args:
        samples: List of dicts with "image" (PIL.Image) and "label" (int)
        extractor: Feature extractor model
        transform: Preprocessing transform to apply
        config: Config object with cache_dir, batch_size, device
        split: Split name (e.g., "train", "validation", "test")

    Returns:
        Path to the cache directory
    """
    cache_dir = config.cache_dir / split
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Caching {len(samples)} samples to {cache_dir}...")

    # Process in batches
    for batch_start in tqdm(range(0, len(samples), config.batch_size)):
        batch_end = min(batch_start + config.batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]

        # Apply transform and stack into tensor [B, 3, 224, 224]
        batch_images = torch.stack([
            transform(sample["image"]) for sample in batch_samples
        ])

        # Extract features: layer4 [B, 2048, 7, 7] + avgpool [B, 2048]
        layer4_features, avgpool_features = extract_features_batch(
            extractor, batch_images, config.device
        )

        # Save each feature individually with both spatial and global features
        for i, sample in enumerate(batch_samples):
            idx = batch_start + i
            cache_path = cache_dir / f"{idx:06d}.pt"

            torch.save(
                {
                    "features": layer4_features[i],      # [2048, 7, 7]
                    "global_feat": avgpool_features[i],  # [2048]
                    "label": sample["label"],            # int
                },
                cache_path
            )

    print(f"Cached {len(samples)} features to {cache_dir}")
    return cache_dir


def load_cached_features(config: Config, split: str) -> List[Dict]:
    """Load all cached features from disk.

    Args:
        config: Config object with cache_dir
        split: Split name (e.g., "train", "validation", "test")

    Returns:
        List of dicts with "features" (tensor [2048, 7, 7]) and "label" (int)
    """
    cache_dir = config.cache_dir / split
    cache_files = sorted(cache_dir.glob("*.pt"))

    features_list = []
    for cache_path in cache_files:
        # weights_only=False because we save dict with tensor + int label
        # This is safe because we control the cache files
        data = torch.load(cache_path, weights_only=False)
        features_list.append(data)

    print(f"Loaded {len(features_list)} cached features from {cache_dir}")
    return features_list
