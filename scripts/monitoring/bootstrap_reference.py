"""Bootstrap the Evidently reference dataset from curated inference events.

The reference dataset establishes the baseline distribution that all future
monitoring windows are compared against.  It should represent normal, healthy
traffic — i.e. a window of real inference events when the model was known good.

Usage (export the last 168 hours of events as reference):
    python scripts/monitoring/bootstrap_reference.py --window-hours 168

Usage (export from a specific time range):
    python scripts/monitoring/bootstrap_reference.py \\
        --since 2026-04-01T00:00:00Z \\
        --until 2026-04-07T23:59:59Z

Usage (seed synthetic reference rows covering all 16 labels):
    python scripts/monitoring/bootstrap_reference.py --synthetic

Outputs a parquet file at monitoring/reference/reference_dataset.parquet.
Overwrites any existing file (refresh is intentional).
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ─── Project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

_env_path = _PROJECT_ROOT / ".env.local"
if _env_path.exists():
    load_dotenv(_env_path)

import os

import pandas as pd

from app.src.monitoring.schema import PROB_COLUMN_NAMES, RVL_CDIP_LABELS
from app.src.monitoring.store import fetch_events_as_dataframe, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_REFERENCE_DIR = _PROJECT_ROOT / "monitoring" / "reference"
_REFERENCE_FILE = _REFERENCE_DIR / "reference_dataset.parquet"

# ─── Builders ─────────────────────────────────────────────────────────────────


def bootstrap_reference_dataset(
    *,
    db_path: Path,
    since: str | None = None,
    until: str | None = None,
    window_hours: int | None = None,
    output_path: Path = _REFERENCE_FILE,
) -> Path:
    """Export real inference events as a reference baseline dataset.

    Args:
        db_path: Path to the SQLite monitoring event store.
        since: ISO-8601 lower bound (overrides window_hours).
        until: ISO-8601 upper bound (defaults to now).
        window_hours: How far back from now to query if since is not set.
        output_path: Parquet file path to write (overwrites existing).

    Returns:
        Path to the written parquet file.
    """
    if since is None:
        hours = window_hours or 168
        since = (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).isoformat()

    if until is None:
        until = datetime.now(tz=timezone.utc).isoformat()

    logger.info("Fetching events since=%s until=%s", since[:19], until[:19])
    df = fetch_events_as_dataframe(db_path, since=since, until=until, limit=50_000)

    if df.empty:
        logger.error("No events found in the specified window. Cannot create reference dataset.")
        logger.error("Submit some classifications via the app first, then re-run.")
        sys.exit(1)

    logger.info("Fetched %d events covering %d model(s)", len(df), df["model_id"].nunique())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Reference dataset written to %s", output_path)
    logger.info("  Rows: %d", len(df))
    logger.info("  Models: %s", sorted(df["model_id"].unique().tolist()))
    logger.info("  Labels: %s", sorted(df["predicted_label"].unique().tolist()))
    return output_path


def bootstrap_synthetic_reference_dataset(
    *,
    rows_per_model: int = 100,
    output_path: Path = _REFERENCE_FILE,
) -> Path:
    """Generate a synthetic reference dataset that covers all 16 RVL-CDIP classes.

    Useful when no real traffic has been captured yet (first-time setup, testing).
    Rows are randomly sampled but balanced across labels and model IDs.

    Args:
        rows_per_model: Number of synthetic rows to generate per model.
        output_path: Parquet file path to write (overwrites existing).

    Returns:
        Path to the written parquet file.
    """
    rng = random.Random(2026)
    now = datetime.now(tz=timezone.utc)

    MODELS = [
        ("cnn_baseline", "CNN Baseline (ResNet-50)", "1.0"),
        ("graphsage_fusion", "Fusion GraphSAGE", "1.0"),
        ("fusion_gat", "Fusion GAT", "1.0"),
        ("boc_graphsage", "BoC GraphSAGE", "1.0"),
        ("gated_boc_graphsage", "Gated BoC GraphSAGE", "1.0"),
        ("attention_pool_graphsage", "Attention Pool GraphSAGE", "1.0"),
    ]

    records: list[dict] = []
    for model_id, display_name, version in MODELS:
        for i in range(rows_per_model):
            label_idx = i % len(RVL_CDIP_LABELS)
            probs = [rng.uniform(0.01, 0.08) for _ in range(16)]
            probs[label_idx] = rng.uniform(0.55, 0.95)
            total = sum(probs)
            probs = [p / total for p in probs]
            ts = (now - timedelta(hours=rng.uniform(72, 240))).isoformat()

            row: dict = {
                "request_id": str(uuid.uuid4()),
                "timestamp": ts,
                "sample_type": rng.choice(["sample", "upload"]),
                "sample_name": f"ref_doc_{i}.png",
                "image_width": rng.randint(600, 1400),
                "image_height": rng.randint(800, 1800),
                "image_mode": rng.choice(["L", "RGB"]),
                "model_id": model_id,
                "model_display_name": display_name,
                "model_version": version,
                "predicted_label": RVL_CDIP_LABELS[label_idx],
                "predicted_index": label_idx,
                "confidence": probs[label_idx],
                "total_time_ms": rng.uniform(80, 400),
                "feature_time_ms": rng.uniform(30, 150),
                "graph_time_ms": rng.uniform(10, 80),
                "model_time_ms": rng.uniform(15, 100),
                "ocr_available": rng.choice([0, 1]),
                "text_density_available": rng.choice([0, 1]),
                "error_type": None,
            }
            for col, val in zip(PROB_COLUMN_NAMES, probs):
                row[col] = val

            records.append(row)

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Synthetic reference dataset written to %s", output_path)
    logger.info("  Rows: %d  Models: %d  Labels: %d", len(df), len(MODELS), len(RVL_CDIP_LABELS))
    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap the Evidently reference dataset for drift monitoring."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--window-hours",
        type=int,
        default=None,
        help="Export real events from the last N hours (default: 168 = 7 days)",
    )
    mode.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a synthetic balanced reference instead of using real events",
    )
    parser.add_argument("--since", type=str, default=None, help="ISO-8601 lower bound (overrides --window-hours)")
    parser.add_argument("--until", type=str, default=None, help="ISO-8601 upper bound (default: now)")
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override MONITORING_DB_PATH env var",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_REFERENCE_FILE),
        help=f"Output parquet path (default: {_REFERENCE_FILE})",
    )
    parser.add_argument(
        "--rows-per-model",
        type=int,
        default=100,
        help="Rows per model in synthetic mode (default: 100)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)

    if args.synthetic:
        bootstrap_synthetic_reference_dataset(
            rows_per_model=args.rows_per_model,
            output_path=output_path,
        )
        return

    raw_db = args.db_path or os.environ.get(
        "MONITORING_DB_PATH", "monitoring/data/inference_events.sqlite3"
    )
    db_path = Path(raw_db) if Path(raw_db).is_absolute() else _PROJECT_ROOT / raw_db

    if not db_path.exists():
        logger.error("Monitoring DB not found: %s", db_path)
        logger.error("Use --synthetic to generate a synthetic reference instead.")
        sys.exit(1)

    bootstrap_reference_dataset(
        db_path=db_path,
        since=args.since,
        until=args.until,
        window_hours=args.window_hours,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
