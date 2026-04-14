# Model Performance Monitoring

## Overview

The monitoring stack logs every inference event to a local SQLite database and
provides a batch job that turns those events into Evidently reports.  The
reports can be written locally (offline mode) or published to Evidently Cloud.

The `/model-performance` route in the app links to the Evidently Cloud dashboard
(or shows a "not configured" page when `EVIDENTLY_DASHBOARD_URL` is empty).

---

## Event Schema

Each inference produces one row per model in `inference_events`:

| Column | Type | Description |
|---|---|---|
| `request_id` | TEXT | UUID shared across all models in one request |
| `timestamp` | TEXT | ISO-8601 UTC |
| `sample_type` | TEXT | `"sample"` (preloaded) or `"upload"` |
| `sample_name` | TEXT | Filename or `"upload"` |
| `image_width` | INTEGER | |
| `image_height` | INTEGER | |
| `image_mode` | TEXT | `"L"` (grayscale), `"RGB"`, etc. |
| `model_id` | TEXT | Matches model registry key |
| `model_display_name` | TEXT | Human-readable model name |
| `model_version` | TEXT | Reserved; `"1.0"` |
| `predicted_label` | TEXT | Winning class label |
| `predicted_index` | INTEGER | Index 0–15 |
| `confidence` | REAL | Max softmax probability |
| `prob_letter` … `prob_memo` | REAL × 16 | Per-class probabilities |
| `total_time_ms` | REAL | End-to-end request latency |
| `feature_time_ms` | REAL | CNN feature extraction |
| `graph_time_ms` | REAL | Graph construction |
| `model_time_ms` | REAL | GNN/model forward pass |
| `ocr_available` | INTEGER | 1 when tesseract OCR succeeded |
| `text_density_available` | INTEGER | 1 when text density map present |
| `error_type` | TEXT | NULL on success |
| `target` | TEXT | Backfilled ground-truth label (optional) |

---

## Multiple-Model Strategy

One monitoring run processes **all** `model_id` values found in the query window.
A separate Evidently report is generated per model so that drift in one model
does not mask or conflate drift in another.

The `model_id` values come directly from the model registry:

| `model_id` | Display name |
|---|---|
| `cnn_baseline` | CNN Baseline (ResNet-50) |
| `graphsage_fusion` | Fusion GraphSAGE |
| `fusion_gat` | Fusion GAT |
| `boc_graphsage` | BoC GraphSAGE |
| `gated_boc_graphsage` | Gated BoC GraphSAGE |
| `attention_pool_graphsage` | Attention Pool GraphSAGE |

---

## Local / Offline Usage

No Evidently Cloud account required.

### Step 1 — Bootstrap the reference dataset

First-time setup (generates synthetic balanced baseline):

```bash
python scripts/monitoring/bootstrap_reference.py --synthetic
```

After collecting real traffic (7-day export):

```bash
python scripts/monitoring/bootstrap_reference.py --window-hours 168
```

### Step 2 — Run the monitoring job

```bash
# 24-hour window, offline HTML/JSON output
python scripts/monitoring/run_evidently.py --window-hours 24 --offline

# 7-day window
python scripts/monitoring/run_evidently.py --window-hours 168 --offline

# Custom output directory
python scripts/monitoring/run_evidently.py \
    --window-hours 24 \
    --offline \
    --output-dir /tmp/evidently-reports
```

Reports are written to `monitoring/output/` (or `EVIDENTLY_OFFLINE_OUTPUT_DIR`):

```
monitoring/output/
  cnn_baseline__unlabeled__20260414T013843.html
  cnn_baseline__unlabeled__20260414T013843.json
  graphsage_fusion__unlabeled__20260414T013847.html
  graphsage_fusion__unlabeled__20260414T013847.json
```

Open any `.html` file in a browser to explore the interactive Evidently report.

---

## Cloud Publishing Usage

Set credentials in `.env.local`:

```bash
EVIDENTLY_API_URL=https://app.evidently.cloud
EVIDENTLY_API_KEY=your-api-key
EVIDENTLY_PROJECT_ID=your-project-uuid
EVIDENTLY_DASHBOARD_URL=https://app.evidently.cloud/projects/your-project-uuid
```

Then run without `--offline`:

```bash
python scripts/monitoring/run_evidently.py --window-hours 24
```

Reports are published directly to the Evidently Cloud project.  The
`/model-performance` route will redirect to `EVIDENTLY_DASHBOARD_URL`.

---

## Report Types

### Unlabeled monitoring (always available)

Generated for every model whenever at least one event exists in the window.

Includes:
- **DataSummaryPreset** — distribution statistics for all columns
- **DataDriftPreset** — column-level drift vs reference (requires reference dataset)

Tracked signals:
- Predicted class distribution (label drift)
- Confidence distribution
- Latency distribution (total, feature, graph, model)
- Input metadata drift (image dimensions, sample type)
- OCR/text-density availability rates

### Labeled quality report (when ground truth is available)

Generated only when a `target` column is present and populated.  Includes
**ClassificationPreset**: accuracy, precision, recall, F1 per class, confusion
matrix trends.

---

## Labeled Data Backfill Strategy

The schema reserves a `target` column for ground-truth labels.  To enable
quality monitoring:

1. Collect human annotations for a set of `request_id` values.
2. Write a backfill script that updates the SQLite `inference_events` table:
   ```sql
   UPDATE inference_events SET target = 'letter' WHERE request_id = '...';
   ```
3. Re-run `run_evidently.py` — it will automatically detect the `target` column
   and produce a labeled quality report alongside the unlabeled one.

No schema changes are needed; `target` is already supported in the event store.

---

## Docker Compose — Manual Monitoring Job

An optional `monitoring` profile is available in `docker-compose.yml` for
running the monitoring job inside the same container environment:

```bash
# Run the monitoring job (one-shot, offline)
docker compose --profile monitoring run --rm monitoring
```

The service exits after generating reports.  It does not run on a schedule.
Scheduling (cron, Airflow, GitHub Actions) is left to the operator.

---

## How `/model-performance` Relates to the Dashboard

The app route at `/model-performance`:

- If `EVIDENTLY_DASHBOARD_URL` is set → 302 redirect to the Evidently Cloud dashboard
- If empty → serves a local HTML page explaining that monitoring is not yet configured

The route is read-only and does not trigger any monitoring job.  The batch job
(`run_evidently.py`) is a separate process that must be invoked manually or via
a scheduler.
