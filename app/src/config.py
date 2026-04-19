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

# Seq structured logging
SEQ_SERVER_URL = os.environ.get("SEQ_SERVER_URL", "")    # empty disables Seq ingestion (non-fatal)
SEQ_API_KEY = os.environ.get("SEQ_API_KEY", "")          # blank = no auth (local dev)
SEQ_UI_URL = os.environ.get("SEQ_UI_URL", "http://localhost:5341")
MLFLOW_UI_URL = os.environ.get("MLFLOW_UI_URL", "http://localhost:5050")
MLFLOW_DB_PATH = os.environ.get("MLFLOW_DB_PATH", str(_PROJECT_ROOT / "monitoring" / "mlflow" / "mlflow.db"))
MLFLOW_EXPERIMENT_NAME = "GNN vs CNN — Document Classification"
ENVIRONMENT = os.environ.get("ENVIRONMENT", "local")

RVL_CDIP_LABELS = [
    "advertisement",
    "budget",
    "email",
    "file folder",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo",
    "news article",
    "presentation",
    "questionnaire",
    "resume",
    "scientific publication",
    "scientific report",
    "specification",
]

# GAT Multimodal
YOLO_REPO_ID = os.environ.get("YOLO_REPO_ID", "hantian/yolo-doclaynet")
YOLO_FILENAME = os.environ.get("YOLO_FILENAME", "yolov8n-doclaynet.pt")
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
GAT_MAX_REGIONS = int(os.environ.get("GAT_MAX_REGIONS", "50"))
GAT_K_NEIGHBORS = int(os.environ.get("GAT_K_NEIGHBORS", "8"))
TEXT_ENCODER_MODEL = os.environ.get("TEXT_ENCODER_MODEL", "all-MiniLM-L6-v2")

MODEL_CHECKPOINTS = {
    "CNN Baseline (ResNet-50)": MODEL_DIR / "best_model_resnet50_03.pt",
    "Fusion GraphSAGE": MODEL_DIR / "fusion_gnn_feat_knn_best.pt",
    "Inductive GCN": MODEL_DIR / "inductive_gcn_320k.pt",
    "Multimodal GAT": MODEL_DIR / "best_gat_multimodal_k8_L2.pt",
}
