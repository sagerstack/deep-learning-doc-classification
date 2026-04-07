from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
SAMPLES_DIR = STATIC_DIR / "samples"

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
    "CNN Baseline (ResNet-50)": PROJECT_ROOT / "models" / "exp14b_finetuned_resnet50_cnn.pt",
    "Fusion GraphSAGE": PROJECT_ROOT / "models" / "exp16_fusion_featknn_graphsage.pt",
    "Fusion GAT": PROJECT_ROOT / "models" / "exp23_gat_fusion.pt",
    "BoC GraphSAGE": PROJECT_ROOT / "models" / "exp25_boc_sage.pt",
    "Gated BoC GraphSAGE": PROJECT_ROOT / "models" / "exp26_gated_boc.pt",
    "Attention Pool GraphSAGE": PROJECT_ROOT / "models" / "exp27_attn_pool.pt",
}
