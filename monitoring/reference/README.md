# Monitoring Reference Dataset

## What this is

`reference_dataset.parquet` is the baseline that Evidently compares all future
monitoring windows against.  It captures what "normal" traffic looks like for
each deployed model — the distribution of predictions, confidences, latencies,
and input metadata when the model was in a known-good state.

Without a reference dataset, Evidently can still report distribution summaries
for the current window, but it cannot detect **drift** (the `DataDriftPreset`
preset requires a reference).

## What it contains

One row per inference event, same schema as the SQLite `inference_events` table:

| Column group | Columns |
|---|---|
| Request context | `request_id`, `timestamp` |
| Input metadata | `sample_type`, `sample_name`, `image_width`, `image_height`, `image_mode` |
| Model identity | `model_id`, `model_display_name`, `model_version` |
| Prediction | `predicted_label`, `predicted_index`, `confidence` |
| Per-class probabilities | `prob_letter` … `prob_memo` (16 columns) |
| Timing | `total_time_ms`, `feature_time_ms`, `graph_time_ms`, `model_time_ms` |
| Feature flags | `ocr_available`, `text_density_available` |
| Optional label | `target` (backfilled when ground truth is available) |

## How it was generated

### First-time setup (synthetic)

When no real traffic exists yet, generate a balanced synthetic baseline:

```bash
python scripts/monitoring/bootstrap_reference.py --synthetic
```

This produces 100 rows per deployed model, balanced across all 16 RVL-CDIP
document classes, and writes `monitoring/reference/reference_dataset.parquet`.

### From real traffic

Once the app has been running and collecting events, export a stable window:

```bash
# Last 7 days of events
python scripts/monitoring/bootstrap_reference.py --window-hours 168

# Specific date range
python scripts/monitoring/bootstrap_reference.py \
    --since 2026-04-01T00:00:00Z \
    --until 2026-04-07T23:59:59Z
```

## When to refresh

Refresh the reference dataset when:

1. **A new model is deployed** — the reference should include rows for the new
   `model_id` so it can be monitored from day one.  Easiest: run one week of
   traffic, then re-export.
2. **Known distribution shift was intentional** — e.g. the document types being
   classified genuinely changed.  Keeping a stale reference would fire
   false-positive drift alerts.
3. **The reference is more than 90 days old** — distributions drift slowly over
   time; refresh quarterly as a hygiene practice.
4. **Model retraining** — always refresh after deploying a retrained model
   checkpoint, because the new model's output distribution may differ.

## How to verify the file

```bash
python - <<'EOF'
import pandas as pd
df = pd.read_parquet("monitoring/reference/reference_dataset.parquet")
print(f"Rows: {len(df)}")
print(f"Models: {sorted(df['model_id'].unique())}")
print(f"Labels: {sorted(df['predicted_label'].unique())}")
print(df.dtypes)
EOF
```
