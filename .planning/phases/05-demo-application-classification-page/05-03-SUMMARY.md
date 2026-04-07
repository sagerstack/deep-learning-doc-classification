---
phase: 05-demo-application-classification-page
plan: 03
subsystem: inference
tags: [pytorch, torchvision, pyg, resnet50, graphsage, gat, inference-pipeline]

requires:
  - phase: 05-01
    provides: model architectures promoted to src/model.py
  - phase: 05-02
    provides: app skeleton with config, routes, templates
provides:
  - ModelSpec dataclass and MODEL_SPECS list for all 6 models
  - load_all_models() for checkpoint loading with device placement
  - run_inference_pipeline() for single-image multi-model classification
affects: [05-04, 05-05, demo-templates, classification-results-display]

tech-stack:
  added: []
  patterns:
    - "ModelSpec registry pattern for multi-model management"
    - "needs_pe flag for correct node feature dimension routing"
    - "Graceful OCR fallback: skip BoC-dependent models if tesseract unavailable"

key-files:
  created:
    - app/src/services/model_registry.py
    - app/src/services/inference.py
  modified: []

key-decisions:
  - "needs_pe flag on ModelSpec to distinguish raw CNN (2048) vs PE-augmented (2050) node features"
  - "Feature extractor built from fine-tuned ResNet-50 checkpoint (not pretrained ImageNet weights)"
  - "OCR-dependent models return placeholder result when tesseract unavailable"

patterns-established:
  - "ModelSpec-driven dispatch: graph_type + needs_pe + needs_boc determine feature routing"
  - "PipelineResult carries intermediate features (layer4, edges) for downstream visualization"

duration: 6min
completed: 2026-04-07
---

# Phase 5 Plan 3: Model Registry and Inference Pipeline Summary

**ModelSpec registry loading 6 checkpoints + single-image inference pipeline producing multi-model predictions with timing and graph stats in ~5s on CPU**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-07T07:44:13Z
- **Completed:** 2026-04-07T07:50:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Model registry with ModelSpec dataclass defining all 6 models (1 CNN + 5 GNN) with checkpoint paths, constructor kwargs, and feature requirements
- Single-image inference pipeline that preprocesses image, extracts ResNet-50 features, constructs kNN/grid graphs, and runs all models
- Structured PipelineResult with per-model predictions, intermediate features for visualization, graph statistics, and per-step timing

## Task Commits

Each task was committed atomically:

1. **Task 1: Create model registry with ModelSpec definitions and checkpoint loading** - `0dadbb5` (feat)
2. **Task 2: Create single-image inference pipeline** - `82a9450` (feat)

## Files Created/Modified
- `app/src/services/model_registry.py` - ModelSpec dataclass, MODEL_SPECS list, load_all_models()
- `app/src/services/inference.py` - InferenceResult/PipelineResult dataclasses, run_inference_pipeline()

## Decisions Made
- Added `needs_pe` boolean flag to ModelSpec to correctly route node features: exp16/exp23 use raw 2048-dim CNN features, while exp25/exp26/exp27 use 2050-dim (CNN + 2D positional encoding)
- Feature extractor is created from the fine-tuned ResNet-50 checkpoint (exp14b), not from pretrained ImageNet weights, ensuring features match what GNN models were trained on
- OCR-dependent models (exp25 BoC GraphSAGE, exp26 Gated BoC) gracefully degrade to placeholder results when tesseract is unavailable

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed node feature dimension mismatch for AttentionPoolFusionSAGE**
- **Found during:** Task 2 (inference pipeline verification)
- **Issue:** Plan's feature routing sent raw 2048-dim features to exp27 (AttentionPoolFusionSAGE), but its conv1 expects 2050-dim (CNN + PE). RuntimeError on mat1/mat2 shape mismatch.
- **Fix:** Added `needs_pe` flag to ModelSpec; set True for exp25/exp26/exp27. Inference pipeline routes PE-augmented features to models that need them.
- **Files modified:** app/src/services/model_registry.py, app/src/services/inference.py
- **Verification:** All 6 models produce predictions on dummy image without errors
- **Committed in:** 82a9450 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix for correctness. Without it, 3 of 5 GNN models would crash on inference.

## Issues Encountered
None beyond the dimension mismatch caught during verification.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Inference pipeline ready for integration with classification route handler
- PipelineResult provides all data needed for results template (predictions, features, graph stats)
- Text density visualization available when doctr is installed (optional)

---
*Phase: 05-demo-application-classification-page*
*Completed: 2026-04-07*
