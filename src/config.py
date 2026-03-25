"""Project configuration with device detection and environment-driven settings."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Config:
    """Project configuration loaded from environment variables with sensible defaults."""

    device: torch.device = None
    seed: int = 42
    sample_size: int = None
    mode: str = None
    cache_dir: Path = None
    hf_cache_dir: Path = None
    batch_size: int = None
    num_workers: int = None
    dataset_name: str = "aharley/rvl_cdip"

    def __post_init__(self):
        """Initialize device detection and load environment variables."""
        # Device detection: MPS -> CUDA -> CPU fallback
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        # Load from environment with defaults
        self.sample_size = int(os.environ.get("SAMPLE_SIZE", "100"))
        self.mode = os.environ.get("DATA_MODE", "sample")
        self.cache_dir = Path(os.environ.get("CACHE_DIR", "./cached_features"))
        self.hf_cache_dir = Path(os.environ.get("HF_CACHE_DIR", "./.hf_cache"))
        self.batch_size = int(os.environ.get("BATCH_SIZE", "32"))
        self.num_workers = int(os.environ.get("NUM_WORKERS", "0"))

        # Print device info
        print(f"Device: {self.device}")
        print(f"Mode: {self.mode}")
        print(f"Sample size: {self.sample_size}")

    def seed_everything(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
