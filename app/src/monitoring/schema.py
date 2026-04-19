"""Monitoring schema: dataclass + helpers for per-model inference events."""

from dataclasses import dataclass, field
from typing import Optional

from app.src.config import RVL_CDIP_LABELS

# Stable, SQL-safe column names for per-class probabilities
PROB_COLUMN_NAMES: list[str] = [
    f"prob_{label.replace(' ', '_').replace('-', '_')}"
    for label in RVL_CDIP_LABELS
]


@dataclass
class InferenceEvent:
    """One row per model prediction per classify request."""

    # Request context
    request_id: str
    timestamp: str                         # ISO-8601 UTC

    # Input metadata
    sample_type: str                       # "sample" | "upload"
    sample_name: str                       # filename or "upload"
    image_width: int
    image_height: int
    image_mode: str                        # "RGB" | "L" | ...

    # Model identity
    model_id: str                          # e.g. "cnn_baseline"
    model_display_name: str                # e.g. "CNN Baseline (ResNet-50)"
    model_version: str                     # reserved; use "1.0" by default

    # Prediction
    predicted_label: str
    predicted_index: int
    confidence: float

    # Per-class probabilities (16 values aligned to PROB_COLUMN_NAMES)
    probabilities: list[float]             # len == 16

    # Timing (milliseconds; 0.0 = not available)
    total_time_ms: float
    feature_time_ms: float
    graph_time_ms: float
    model_time_ms: float

    # Optional feature availability flags
    ocr_available: bool = False
    text_density_available: bool = False

    # Error capture (None on success)
    error_type: Optional[str] = None

    # Ground-truth label for labeled quality monitoring (backfilled post-hoc)
    target: Optional[str] = None


def build_inference_events(
    *,
    request_id: str,
    timestamp: str,
    sample_type: str,
    sample_name: str,
    image_width: int,
    image_height: int,
    image_mode: str,
    results: list,                 # list[InferenceResult] from inference.py
    total_time_ms: float,
    feature_time_ms: float,
    graph_time_ms: float,
    ocr_available: bool,
    text_density_available: bool,
) -> list[InferenceEvent]:
    """Build one InferenceEvent per model result in a pipeline output."""
    events: list[InferenceEvent] = []

    for r in results:
        # Normalise probabilities: may be all-zero for OCR-unavailable placeholder
        probs = list(r.probabilities) if r.probabilities else [0.0] * len(RVL_CDIP_LABELS)
        if len(probs) != len(RVL_CDIP_LABELS):
            probs = (probs + [0.0] * len(RVL_CDIP_LABELS))[: len(RVL_CDIP_LABELS)]

        events.append(
            InferenceEvent(
                request_id=request_id,
                timestamp=timestamp,
                sample_type=sample_type,
                sample_name=sample_name,
                image_width=image_width,
                image_height=image_height,
                image_mode=image_mode,
                model_id=r.model_name,
                model_display_name=r.display_name,
                model_version="1.0",
                predicted_label=r.predicted_class,
                predicted_index=r.predicted_index,
                confidence=r.confidence,
                probabilities=probs,
                total_time_ms=total_time_ms,
                feature_time_ms=feature_time_ms,
                graph_time_ms=graph_time_ms,
                model_time_ms=r.inference_time_ms,
                ocr_available=ocr_available,
                text_density_available=text_density_available,
                error_type=None,
            )
        )

    return events
