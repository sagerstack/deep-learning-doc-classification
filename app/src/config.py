import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env.local from project root (if present)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_env_path = _PROJECT_ROOT / ".env.local"
if _env_path.exists():
    load_dotenv(_env_path)

PROJECT_ROOT = _PROJECT_ROOT

STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
SAMPLES_DIR = STATIC_DIR / "samples"

# Server
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")

# Device
FORCE_DEVICE = os.environ.get("FORCE_DEVICE", "")

# CORS
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

# Models
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
if not MODEL_DIR.is_absolute():
    MODEL_DIR = PROJECT_ROOT / MODEL_DIR

RVL_CDIP_LABELS = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]

MODEL_CHECKPOINTS = {
    "CNN Baseline (ResNet-50)": MODEL_DIR / "exp14b_finetuned_resnet50_cnn.pt",
    "Fusion GraphSAGE": MODEL_DIR / "exp16_fusion_featknn_graphsage.pt",
    "Fusion GAT": MODEL_DIR / "exp23_gat_fusion.pt",
    "BoC GraphSAGE": MODEL_DIR / "exp25_boc_sage.pt",
    "Gated BoC GraphSAGE": MODEL_DIR / "exp26_gated_boc.pt",
    "Attention Pool GraphSAGE": MODEL_DIR / "exp27_attn_pool.pt",
}
