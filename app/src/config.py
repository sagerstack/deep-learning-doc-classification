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

# Monitoring
_monitoring_db_raw = os.environ.get("MONITORING_DB_PATH", "monitoring/data/inference_events.sqlite3")
MONITORING_DB_PATH = Path(_monitoring_db_raw)
if not MONITORING_DB_PATH.is_absolute():
    MONITORING_DB_PATH = PROJECT_ROOT / MONITORING_DB_PATH

EVIDENTLY_DASHBOARD_URL = os.environ.get("EVIDENTLY_DASHBOARD_URL", "")

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

# GAT Multimodal
YOLO_REPO_ID = os.environ.get("YOLO_REPO_ID", "hantian/yolo-doclaynet")
YOLO_FILENAME = os.environ.get("YOLO_FILENAME", "yolov8n-doclaynet.pt")
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.15"))
GAT_MAX_REGIONS = int(os.environ.get("GAT_MAX_REGIONS", "20"))
GAT_K_NEIGHBORS = int(os.environ.get("GAT_K_NEIGHBORS", "8"))
TEXT_ENCODER_MODEL = os.environ.get("TEXT_ENCODER_MODEL", "all-MiniLM-L6-v2")

MODEL_CHECKPOINTS = {
    "CNN Baseline (ResNet-50)": MODEL_DIR / "exp14b_finetuned_resnet50_cnn.pt",
    "Fusion GraphSAGE": MODEL_DIR / "exp16_fusion_featknn_graphsage.pt",
    "Fusion GAT": MODEL_DIR / "exp23_gat_fusion.pt",
    "BoC GraphSAGE": MODEL_DIR / "exp25_boc_sage.pt",
    "Gated BoC GraphSAGE": MODEL_DIR / "exp26_gated_boc.pt",
    "Attention Pool GraphSAGE": MODEL_DIR / "exp27_attn_pool.pt",
}
