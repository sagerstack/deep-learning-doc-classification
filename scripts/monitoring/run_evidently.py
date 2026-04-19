"""Evidently batch monitoring job.

Generates per-model monitoring reports, writes HTML/JSON artifacts locally, and
optionally pushes snapshots into an Evidently UI workspace so they appear in the
live dashboard served by `evidently ui`.

Usage (local dashboard — recommended):
    # Terminal 1: run the UI server
    poetry run evidently ui --workspace monitoring/evidently_workspace --port 8080

    # Terminal 2: generate reports and push to workspace
    python scripts/monitoring/run_evidently.py --window-hours 24

Usage (offline HTML only):
    python scripts/monitoring/run_evidently.py --window-hours 24 --no-workspace

Env vars consumed (from .env.local or shell):
    MONITORING_DB_PATH          — SQLite event store (default: monitoring/data/inference_events.sqlite3)
    EVIDENTLY_WORKSPACE_PATH    — Workspace dir for live UI (default: monitoring/evidently_workspace)
    EVIDENTLY_DASHBOARD_URL     — Base URL of the running `evidently ui` server (default: http://localhost:8080)
    EVIDENTLY_OFFLINE_OUTPUT_DIR — Directory for local HTML/JSON output (default: monitoring/output)
    EVIDENTLY_API_KEY           — API key for Evidently Cloud publishing (optional)
    EVIDENTLY_API_URL           — Evidently Cloud URL (optional)
    EVIDENTLY_PROJECT_ID        — Evidently Cloud project ID (optional)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

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
from evidently.ui.workspace import Workspace
from evidently.sdk.models import DashboardPanelPlot, PanelMetric

from app.src.monitoring.schema import PROB_COLUMN_NAMES, RVL_CDIP_LABELS
from app.src.monitoring.store import fetch_events_as_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_REFERENCE_DIR = _PROJECT_ROOT / "monitoring" / "reference"
_DEFAULT_REFERENCE_FILENAME = "reference_dataset.parquet"
_DEFAULT_WORKSPACE_DIR = _PROJECT_ROOT / "monitoring" / "evidently_workspace"
_DEFAULT_DASHBOARD_URL = "http://localhost:8080"

_METADATA_NUMERICAL_COLS = [
    "image_width", "image_height",
    "total_time_ms", "feature_time_ms", "graph_time_ms", "model_time_ms",
    "confidence",
]
_METADATA_CATEGORICAL_COLS = [
    "sample_type", "image_mode", "predicted_label",
    "ocr_available", "text_density_available",
]


# ─── Workspace helpers ────────────────────────────────────────────────────────

def _get_or_create_workspace(workspace_path: Path) -> Workspace:
    """Load or create an Evidently UI workspace."""
    workspace_path.mkdir(parents=True, exist_ok=True)
    ws = Workspace.create(str(workspace_path))
    logger.info("Evidently workspace ready at %s", workspace_path)
    return ws


def _get_or_create_project(ws: Workspace, model_id: str) -> object:
    """Get or create one Evidently project per model."""
    project_name = f"doc_classification__{model_id}"
    for proj in ws.list_projects():
        if proj.name == project_name:
            logger.info("  Using existing project: %s (%s)", project_name, proj.id)
            return ws.get_project(proj.id)

    proj = ws.create_project(project_name)
    proj.description = f"RVL-CDIP document classification monitoring — {model_id}"
    _configure_dashboard(proj, model_id)
    proj.save()
    logger.info("  Created project: %s (%s)", project_name, proj.id)
    return proj


def _configure_dashboard(project, model_id: str) -> None:
    """Configure monitoring dashboard panels for a single model.

    Layout:
      Row 1 (2 cols): Latest Confidence (counter) | Drifted Features % (counter)
      Row 2 (2 cols): Drifted Features # (counter) | Prediction Drift p-value (counter)
      Row 3 (full):   Confidence Trend over time (line)
      Row 4 (full):   Drift Trends over time (line)
    """
    try:
        # Row 1 — Key counters
        project.dashboard.add_panel(
            DashboardPanelPlot(
                title="Avg Confidence (latest)",
                size="half",
                values=[PanelMetric(
                    metric="evidently:metric_v2:MeanValue",
                    metric_labels={"column": "confidence"},
                    legend="Mean Confidence",
                )],
                plot_params={"plot_type": "counter"},
            )
        )
        project.dashboard.add_panel(
            DashboardPanelPlot(
                title="Drifted Features (%)",
                size="half",
                values=[PanelMetric(
                    metric="evidently:metric_v2:DriftedColumnsCount",
                    metric_labels={"value_type": "share"},
                    legend="Drift Share",
                )],
                plot_params={"plot_type": "counter"},
            )
        )

        # Row 2 — More counters
        project.dashboard.add_panel(
            DashboardPanelPlot(
                title="Drifted Features (#)",
                size="half",
                values=[PanelMetric(
                    metric="evidently:metric_v2:DriftedColumnsCount",
                    metric_labels={"value_type": "count"},
                    legend="Drift Count",
                )],
                plot_params={"plot_type": "counter"},
            )
        )
        project.dashboard.add_panel(
            DashboardPanelPlot(
                title="Prediction Drift (p-value)",
                size="half",
                values=[PanelMetric(
                    metric="evidently:metric_v2:ValueDrift",
                    metric_labels={"column": "predicted_label"},
                    legend="Pred Drift",
                )],
                plot_params={"plot_type": "counter"},
            )
        )

        # Row 3 — Confidence trend (full width)
        project.dashboard.add_panel(
            DashboardPanelPlot(
                title="Confidence Over Time",
                size="full",
                values=[PanelMetric(
                    metric="evidently:metric_v2:MeanValue",
                    metric_labels={"column": "confidence"},
                    legend="Mean Confidence",
                )],
                plot_params={"plot_type": "line"},
            )
        )

        # Row 4 — Drift trends (full width)
        project.dashboard.add_panel(
            DashboardPanelPlot(
                title="Drift Trends Over Time",
                size="full",
                values=[
                    PanelMetric(
                        metric="evidently:metric_v2:DriftedColumnsCount",
                        metric_labels={"value_type": "share"},
                        legend="Feature Drift %",
                    ),
                    PanelMetric(
                        metric="evidently:metric_v2:ValueDrift",
                        metric_labels={"column": "predicted_label"},
                        legend="Prediction Drift",
                    ),
                    PanelMetric(
                        metric="evidently:metric_v2:ValueDrift",
                        metric_labels={"column": "confidence"},
                        legend="Confidence Drift",
                    ),
                ],
                plot_params={"plot_type": "line"},
            )
        )

        logger.info("  Dashboard configured: 6 panels for %s", model_id)
    except Exception as exc:
        logger.warning("  Failed to configure dashboard panels: %s", exc)


def _push_to_workspace(ws: Workspace, project, snapshot, run_name: str) -> None:
    """Add snapshot to workspace and reload the UI server cache."""
    try:
        ws.add_run(project_id=project.id, run=snapshot, name=run_name)
        logger.info("  [workspace] pushed snapshot → project %s", project.name)
    except Exception as exc:
        logger.warning("  [workspace] failed to push snapshot: %s", exc)


def _trigger_reload(dashboard_url: str, project_id: str) -> None:
    """Tell the running Evidently UI server to reload snapshots for this project."""
    try:
        url = f"{dashboard_url.rstrip('/')}/api/projects/{project_id}/reload"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            logger.info("  [dashboard] reloaded project %s (refresh browser to see updates)", project_id)
        else:
            logger.warning("  [dashboard] reload returned HTTP %d", resp.status_code)
    except requests.RequestException as exc:
        logger.warning("  [dashboard] could not reach UI server at %s: %s", dashboard_url, exc)


# ─── Report builder ───────────────────────────────────────────────────────────

def build_reports_for_window(
    *,
    window_hours: int,
    db_path: Path,
    reference_path: Path,
    output_dir: Path,
    workspace_path: Path | None,
    dashboard_url: str,
    env: str = "local",
) -> dict[str, list[Path]]:
    """Generate Evidently reports for all model_ids found in the time window.

    Always writes HTML/JSON artifacts locally. If workspace_path is set, also
    pushes snapshots into the Evidently UI workspace for the live dashboard.
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

    ws = _get_or_create_workspace(workspace_path) if workspace_path else None

    produced: dict[str, list[Path]] = {}

    for model_id, model_df in current_df.groupby("model_id"):
        logger.info("Processing model_id=%s  rows=%d", model_id, len(model_df))
        model_version = model_df["model_version"].iloc[0]
        labeled_count = model_df["target"].notna().sum() if "target" in model_df.columns else 0
        # Require at least 5 labeled rows — fewer can't support a 16-class confusion matrix.
        has_labels = labeled_count >= 5

        ref_model_df = (
            reference_df[reference_df["model_id"] == model_id].copy()
            if reference_df is not None and not reference_df.empty
            else None
        )

        project = _get_or_create_project(ws, model_id) if ws else None

        paths = _run_model_reports(
            model_id=model_id,
            model_version=model_version,
            current_df=model_df.copy(),
            reference_df=ref_model_df,
            has_labels=has_labels,
            batch_window=batch_window,
            env=env,
            output_dir=output_dir,
            ws=ws,
            project=project,
            dashboard_url=dashboard_url,
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
    output_dir: Path,
    ws,
    project,
    dashboard_url: str,
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

    paths.extend(_write_snapshot(
        snapshot=unlabeled_snapshot,
        report_type="unlabeled",
        safe_model_id=safe_model_id,
        timestamp_tag=timestamp_tag,
        output_dir=output_dir,
    ))

    if ws and project:
        _push_to_workspace(ws, project, unlabeled_snapshot, f"{safe_model_id}__unlabeled__{timestamp_tag}")
        _trigger_reload(dashboard_url, str(project.id))

    # ── Labeled quality report (only when sufficient ground-truth labels exist) ─
    if has_labels:
        labeled_df = current_df[current_df["target"].notna()].copy()
        labeled_current = _build_evidently_dataset(labeled_df, include_target=True)
        # Synthetic reference has no target column — run labeled report without reference.
        labeled_reference = None

        labeled_report = Report(
            metrics=[ClassificationPreset(classification_name="document_class")],
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

        paths.extend(_write_snapshot(
            snapshot=labeled_snapshot,
            report_type="labeled",
            safe_model_id=safe_model_id,
            timestamp_tag=timestamp_tag,
            output_dir=output_dir,
        ))

        if ws and project:
            _push_to_workspace(ws, project, labeled_snapshot, f"{safe_model_id}__labeled__{timestamp_tag}")
            _trigger_reload(dashboard_url, str(project.id))

    return paths


def _build_evidently_dataset(df: pd.DataFrame, *, include_target: bool) -> Dataset:
    """Wrap a DataFrame in an Evidently Dataset with explicit column roles."""
    df = df.copy()

    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    has_labels = include_target and "target" in df.columns and df["target"].notna().any()

    if not include_target and "target" in df.columns:
        # Drop target for unlabeled datasets — Evidently flags "partially present"
        # when current has the column but reference does not.
        df = df.drop(columns=["target"])

    classification = (
        [MulticlassClassification(
            name="document_class",
            target="target",
            prediction_labels="predicted_label",
            # Omit prediction_probas: prob column names use underscores (prob_scientific_report)
            # but target/predicted_label use spaces ('scientific report'). Evidently derives
            # class labels from the prob column names, producing a mismatch that causes
            # sklearn's confusion_matrix to raise ValueError. Labels are inferred correctly
            # from target and predicted_label when probas are omitted.
        )]
        if has_labels else None
    )

    data_def = DataDefinition(
        timestamp="timestamp",
        numerical_columns=_METADATA_NUMERICAL_COLS,
        categorical_columns=_METADATA_CATEGORICAL_COLS,
        classification=classification,
    )
    return Dataset.from_pandas(df, data_definition=data_def)


def _write_snapshot(
    *,
    snapshot,
    report_type: str,
    safe_model_id: str,
    timestamp_tag: str,
    output_dir: Path,
) -> list[Path]:
    """Write snapshot HTML and JSON to the local output directory."""
    html_path = output_dir / f"{safe_model_id}__{report_type}__{timestamp_tag}.html"
    json_path = output_dir / f"{safe_model_id}__{report_type}__{timestamp_tag}.json"
    snapshot.save_html(str(html_path))
    json_path.write_text(snapshot.json())
    logger.info("  [html] wrote %s", html_path.name)
    logger.info("  [json] wrote %s", json_path.name)
    return [html_path, json_path]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _utcnow_minus_hours(hours: int) -> datetime:
    from datetime import timedelta
    return datetime.now(tz=timezone.utc) - timedelta(hours=hours)


def _load_reference(reference_path: Path) -> pd.DataFrame | None:
    if not reference_path.exists():
        logger.warning("Reference dataset not found at %s — running without drift comparison", reference_path)
        return None
    df = pd.read_parquet(reference_path)
    logger.info("Loaded reference dataset: %d rows from %s", len(df), reference_path.name)
    return df


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    default_workspace = os.environ.get("EVIDENTLY_WORKSPACE_PATH", str(_DEFAULT_WORKSPACE_DIR))
    default_dashboard = os.environ.get("EVIDENTLY_DASHBOARD_URL", _DEFAULT_DASHBOARD_URL)

    parser = argparse.ArgumentParser(
        description="Generate Evidently monitoring reports from inference event log."
    )
    parser.add_argument(
        "--window-hours", type=int, default=24,
        help="Hours to look back from now (default: 24)",
    )
    parser.add_argument(
        "--no-workspace", action="store_true",
        help="Skip workspace push — write HTML/JSON only (no live dashboard update)",
    )
    parser.add_argument(
        "--workspace", type=str, default=default_workspace,
        help=f"Evidently workspace directory (default: {default_workspace})",
    )
    parser.add_argument(
        "--dashboard-url", type=str, default=default_dashboard,
        help=f"Base URL of running `evidently ui` server (default: {default_dashboard})",
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Override MONITORING_DB_PATH env var",
    )
    parser.add_argument(
        "--reference", type=str, default=None,
        help=f"Path to reference parquet (default: {_REFERENCE_DIR / _DEFAULT_REFERENCE_FILENAME})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override EVIDENTLY_OFFLINE_OUTPUT_DIR env var",
    )
    parser.add_argument(
        "--env", type=str, default="local",
        help="Environment tag attached to each report (default: local)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

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

    workspace_path = None if args.no_workspace else Path(args.workspace)

    if not db_path.exists():
        logger.error("Monitoring DB not found: %s", db_path)
        logger.error("Run the app and submit at least one classification to seed events.")
        sys.exit(1)

    produced = build_reports_for_window(
        window_hours=args.window_hours,
        db_path=db_path,
        reference_path=reference_path,
        output_dir=output_dir,
        workspace_path=workspace_path,
        dashboard_url=args.dashboard_url,
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

    if workspace_path:
        logger.info("")
        logger.info("Live dashboard: %s", args.dashboard_url)
        logger.info("If the UI server is not running, start it with:")
        logger.info("  poetry run evidently ui --workspace %s --port 8080", workspace_path)


if __name__ == "__main__":
    main()
