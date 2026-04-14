"""Evidently batch monitoring job.

Generates per-model monitoring reports for a chosen time window by querying
inference events logged by the app and comparing against a reference dataset.

Usage (offline — no cloud credentials needed):
    python scripts/monitoring/run_evidently.py --window-hours 24 --offline

Usage (cloud publish):
    python scripts/monitoring/run_evidently.py --window-hours 24

Env vars consumed (from .env.local or shell):
    MONITORING_DB_PATH         — SQLite event store (default: monitoring/data/inference_events.sqlite3)
    EVIDENTLY_API_URL          — Evidently Cloud URL
    EVIDENTLY_API_KEY          — API key for cloud publishing
    EVIDENTLY_PROJECT_ID       — Project ID to publish into
    EVIDENTLY_OFFLINE_OUTPUT_DIR — Directory for local HTML/JSON output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ─── Project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

_env_path = _PROJECT_ROOT / ".env.local"
if _env_path.exists():
    load_dotenv(_env_path)

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.core.datasets import MulticlassClassification
from evidently.presets import ClassificationPreset, DataDriftPreset, DataSummaryPreset

from app.src.monitoring.schema import PROB_COLUMN_NAMES, RVL_CDIP_LABELS
from app.src.monitoring.store import fetch_events_as_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_REFERENCE_DIR = _PROJECT_ROOT / "monitoring" / "reference"
_DEFAULT_REFERENCE_FILENAME = "reference_dataset.parquet"

_METADATA_NUMERICAL_COLS = [
    "image_width", "image_height",
    "total_time_ms", "feature_time_ms", "graph_time_ms", "model_time_ms",
    "confidence",
]
_METADATA_CATEGORICAL_COLS = [
    "sample_type", "image_mode", "predicted_label",
    "ocr_available", "text_density_available",
]


# ─── Report builder ───────────────────────────────────────────────────────────

def build_reports_for_window(
    *,
    window_hours: int,
    offline: bool,
    db_path: Path,
    reference_path: Path,
    output_dir: Path,
    env: str = "local",
) -> dict[str, list[Path]]:
    """Generate Evidently reports for all model_ids found in the time window.

    Args:
        window_hours: How far back from now to query inference events.
        offline: When True, write HTML/JSON artifacts locally instead of publishing.
        db_path: Path to the SQLite monitoring event store.
        reference_path: Path to the reference dataset parquet file.
        output_dir: Directory for local report artifacts (used when offline=True).
        env: Deployment environment tag (e.g. "local", "prod").

    Returns:
        Dict mapping model_id to list of output file paths produced.
    """
    since_dt = _utcnow_minus_hours(window_hours)
    since_str = since_dt.isoformat()
    until_str = datetime.now(tz=timezone.utc).isoformat()
    batch_window = f"{since_str[:19]}Z/{until_str[:19]}Z"

    logger.info("Querying events: %s → now (window=%dh)", since_str[:19], window_hours)
    current_df = fetch_events_as_dataframe(db_path, since=since_str, until=until_str)

    if current_df.empty:
        logger.warning("No inference events found in the window. Nothing to report.")
        return {}

    logger.info("Loaded %d events across %d model(s)", len(current_df), current_df["model_id"].nunique())

    reference_df = _load_reference(reference_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, list[Path]] = {}

    for model_id, model_df in current_df.groupby("model_id"):
        logger.info("Processing model_id=%s  rows=%d", model_id, len(model_df))
        model_version = model_df["model_version"].iloc[0]
        has_labels = "target" in model_df.columns and model_df["target"].notna().any()

        ref_model_df = (
            reference_df[reference_df["model_id"] == model_id].copy()
            if reference_df is not None and not reference_df.empty
            else None
        )

        paths = _run_model_reports(
            model_id=model_id,
            model_version=model_version,
            current_df=model_df.copy(),
            reference_df=ref_model_df,
            has_labels=has_labels,
            batch_window=batch_window,
            env=env,
            offline=offline,
            output_dir=output_dir,
        )
        produced[model_id] = paths

    return produced


def _run_model_reports(
    *,
    model_id: str,
    model_version: str,
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame | None,
    has_labels: bool,
    batch_window: str,
    env: str,
    offline: bool,
    output_dir: Path,
) -> list[Path]:
    """Build unlabeled (and optionally labeled) reports for a single model."""
    safe_model_id = model_id.replace(" ", "_").replace("/", "_")
    timestamp_tag = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    paths: list[Path] = []

    # ── Unlabeled monitoring report ────────────────────────────────────────────
    unlabeled_current = _build_evidently_dataset(current_df, include_target=False)
    unlabeled_reference = (
        _build_evidently_dataset(reference_df, include_target=False)
        if reference_df is not None and len(reference_df) > 0
        else None
    )

    # DataDriftPreset requires a reference dataset; fall back to summary-only when absent.
    has_reference = unlabeled_reference is not None
    unlabeled_metrics = [DataSummaryPreset()]
    if has_reference:
        unlabeled_metrics.append(DataDriftPreset())

    unlabeled_report = Report(
        metrics=unlabeled_metrics,
        tags=[env, model_id, "unlabeled"],
        metadata={
            "model_id": model_id,
            "model_version": model_version,
            "batch_window": batch_window,
            "report_type": "unlabeled",
            "env": env,
        },
    )

    unlabeled_snapshot = unlabeled_report.run(
        current_data=unlabeled_current,
        reference_data=unlabeled_reference,
        name=f"{model_id} | unlabeled | {batch_window}",
    )

    paths.extend(
        _output_snapshot(
            snapshot=unlabeled_snapshot,
            report_type="unlabeled",
            safe_model_id=safe_model_id,
            timestamp_tag=timestamp_tag,
            offline=offline,
            output_dir=output_dir,
        )
    )

    # ── Labeled quality report (only when ground-truth labels are present) ─────
    if has_labels:
        labeled_current = _build_evidently_dataset(current_df, include_target=True)
        labeled_reference = (
            _build_evidently_dataset(reference_df, include_target=True)
            if reference_df is not None and len(reference_df) > 0
            else None
        )

        labeled_metrics = [
            ClassificationPreset(),
        ]
        labeled_report = Report(
            metrics=labeled_metrics,
            tags=[env, model_id, "labeled"],
            metadata={
                "model_id": model_id,
                "model_version": model_version,
                "batch_window": batch_window,
                "report_type": "labeled",
                "env": env,
            },
        )

        labeled_snapshot = labeled_report.run(
            current_data=labeled_current,
            reference_data=labeled_reference,
            name=f"{model_id} | labeled | {batch_window}",
        )

        paths.extend(
            _output_snapshot(
                snapshot=labeled_snapshot,
                report_type="labeled",
                safe_model_id=safe_model_id,
                timestamp_tag=timestamp_tag,
                offline=offline,
                output_dir=output_dir,
            )
        )

    return paths


def _build_evidently_dataset(df: pd.DataFrame, *, include_target: bool) -> Dataset:
    """Wrap a DataFrame in an Evidently Dataset with explicit column roles.

    For unlabeled monitoring, no classification block is added (MulticlassClassification
    requires both target and prediction columns). Classification-quality metrics are only
    included when ground-truth labels are available (include_target=True).
    """
    has_labels = include_target and "target" in df.columns and df["target"].notna().any()

    if has_labels:
        classification = [
            MulticlassClassification(
                name="document_class",
                target="target",
                prediction_labels="predicted_label",
                prediction_probas=PROB_COLUMN_NAMES,
            )
        ]
    else:
        classification = None

    data_def = DataDefinition(
        timestamp="timestamp",
        numerical_columns=_METADATA_NUMERICAL_COLS,
        categorical_columns=_METADATA_CATEGORICAL_COLS,
        classification=classification,
    )
    return Dataset.from_pandas(df, data_definition=data_def)


def _output_snapshot(
    *,
    snapshot,
    report_type: str,
    safe_model_id: str,
    timestamp_tag: str,
    offline: bool,
    output_dir: Path,
) -> list[Path]:
    """Write snapshot to local files or publish to Evidently Cloud."""
    written: list[Path] = []

    if offline:
        html_path = output_dir / f"{safe_model_id}__{report_type}__{timestamp_tag}.html"
        json_path = output_dir / f"{safe_model_id}__{report_type}__{timestamp_tag}.json"
        snapshot.save_html(str(html_path))
        json_path.write_text(snapshot.json())
        logger.info("  [offline] wrote %s", html_path.name)
        logger.info("  [offline] wrote %s", json_path.name)
        written.extend([html_path, json_path])
    else:
        _publish_to_cloud(snapshot, report_type=report_type, safe_model_id=safe_model_id)

    return written


def _publish_to_cloud(snapshot, *, report_type: str, safe_model_id: str) -> None:
    """Publish snapshot to Evidently Cloud using env-var credentials."""
    api_key = os.environ.get("EVIDENTLY_API_KEY", "")
    project_id = os.environ.get("EVIDENTLY_PROJECT_ID", "")
    api_url = os.environ.get("EVIDENTLY_API_URL", "https://app.evidently.cloud")

    if not api_key or not project_id:
        logger.warning(
            "EVIDENTLY_API_KEY or EVIDENTLY_PROJECT_ID not set — "
            "falling back to offline output for %s/%s",
            safe_model_id,
            report_type,
        )
        return

    try:
        from evidently.ui.workspace.cloud import CloudWorkspace  # noqa: PLC0415

        ws = CloudWorkspace(token=api_key, url=api_url)
        project = ws.get_project(project_id)
        project.add_run(snapshot)
        logger.info("  [cloud] published %s/%s", safe_model_id, report_type)
    except Exception as exc:
        logger.error("Cloud publish failed (%s/%s): %s", safe_model_id, report_type, exc)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _utcnow_minus_hours(hours: int) -> datetime:
    from datetime import timedelta  # noqa: PLC0415

    return datetime.now(tz=timezone.utc) - timedelta(hours=hours)


def _load_reference(reference_path: Path) -> pd.DataFrame | None:
    if not reference_path.exists():
        logger.warning("Reference dataset not found at %s — running without reference comparison", reference_path)
        return None
    df = pd.read_parquet(reference_path)
    logger.info("Loaded reference dataset: %d rows from %s", len(df), reference_path.name)
    return df


# ─── CLI entrypoint ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Evidently monitoring reports from inference event log."
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=24,
        help="Hours to look back from now (default: 24)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Write HTML/JSON locally instead of publishing to Evidently Cloud",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override MONITORING_DB_PATH env var",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help=f"Path to reference parquet (default: {_REFERENCE_DIR / _DEFAULT_REFERENCE_FILENAME})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override EVIDENTLY_OFFLINE_OUTPUT_DIR env var",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="local",
        help="Environment tag attached to each report (default: local)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve paths (CLI > env > defaults)
    raw_db = args.db_path or os.environ.get(
        "MONITORING_DB_PATH", "monitoring/data/inference_events.sqlite3"
    )
    db_path = Path(raw_db) if Path(raw_db).is_absolute() else _PROJECT_ROOT / raw_db

    reference_path = (
        Path(args.reference) if args.reference
        else _REFERENCE_DIR / _DEFAULT_REFERENCE_FILENAME
    )

    raw_output = args.output_dir or os.environ.get(
        "EVIDENTLY_OFFLINE_OUTPUT_DIR", "monitoring/output"
    )
    output_dir = Path(raw_output) if Path(raw_output).is_absolute() else _PROJECT_ROOT / raw_output

    if not db_path.exists():
        logger.error("Monitoring DB not found: %s", db_path)
        logger.error("Run the app and submit at least one classification to seed events.")
        sys.exit(1)

    produced = build_reports_for_window(
        window_hours=args.window_hours,
        offline=args.offline,
        db_path=db_path,
        reference_path=reference_path,
        output_dir=output_dir,
        env=args.env,
    )

    if not produced:
        logger.info("No reports generated.")
        return

    logger.info("Reports generated for %d model(s):", len(produced))
    for model_id, paths in produced.items():
        logger.info("  %s → %d artifact(s)", model_id, len(paths))
        for p in paths:
            logger.info("    %s", p)


if __name__ == "__main__":
    main()
