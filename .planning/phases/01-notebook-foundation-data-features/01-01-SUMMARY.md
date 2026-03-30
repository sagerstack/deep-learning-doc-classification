---
phase: 01-notebook-foundation-data-features
plan: 01
subsystem: data
tags: [pytorch, huggingface-datasets, rvl-cdip, torchvision, preprocessing, config]

# Dependency graph
requires:
  - phase: none
    provides: "Initial project setup"
provides:
  - "src/ Python package with Config dataclass for env-driven configuration"
  - "RVL-CDIP data loading with sample/full modes via HuggingFace datasets"
  - "Preprocessing pipeline: grayscale PIL -> [3, 224, 224] tensor with ImageNet normalization"
  - "16-class label constants and visualization utilities"
affects: [02-feature-extraction, 03-graph-construction, notebook-implementation]

# Tech tracking
tech-stack:
  added: [torch, torchvision, datasets, pillow, tqdm, torch-geometric, numpy, matplotlib]
  patterns: ["Environment-driven config with dataclasses", "Streaming vs full dataset modes", "Split name normalization (val -> validation)"]

key-files:
  created:
    - "src/__init__.py"
    - "src/config.py"
    - "src/data.py"
  modified:
    - "pyproject.toml"
    - "poetry.lock"

key-decisions:
  - "Used chainyo/rvl-cdip dataset instead of aharley/rvl_cdip (modern parquet format, no loading scripts)"
  - "Normalized split names to train/validation/test (maps val -> validation internally)"
  - "MPS -> CUDA -> CPU device auto-detection pattern"
  - "Sample mode uses streaming with shuffle before take for random sampling"

patterns-established:
  - "All config values loaded from env vars with defaults (no hardcoded values per user values)"
  - "dataclasses for config management instead of pydantic"
  - "Consistent dict structure: {'image': PIL.Image, 'label': int} across sample/full modes"

# Metrics
duration: 12min
completed: 2026-03-25
---

# Phase 01 Plan 01: Foundation - Data & Config Summary

**Importable src/ package with env-driven Config and RVL-CDIP data loading supporting streaming sample mode (100 images) and full download mode (320k/40k/40k splits) with grayscale-to-RGB preprocessing pipeline**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-25T07:24:28Z
- **Completed:** 2026-03-25T07:36:32Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Created importable src/ Python package with __init__.py marker
- Config dataclass with device auto-detection (MPS/CUDA/CPU) and environment variable loading
- RVL-CDIP data loader supporting both sample mode (streaming with shuffle) and full mode (download)
- Preprocessing transform pipeline converting grayscale PIL images to [3, 224, 224] tensors with ImageNet normalization
- 16-class label constants (RVL_CDIP_LABELS) and visualization utilities (display_samples, apply_transform)

## Task Commits

Each task was committed atomically:

1. **Task 1: Install dependencies and create src/ package with config module** - `6feff18` (feat)
2. **Task 2: Create data loading and preprocessing module** - `5f273f4` (feat)

## Files Created/Modified
- `pyproject.toml` - Added torch, torchvision, datasets, pillow, tqdm, torch-geometric dependencies; fixed Python version constraint
- `poetry.lock` - Generated lock file with 48 new packages
- `src/__init__.py` - Package marker (empty file)
- `src/config.py` - Config dataclass with device detection, env var loading, and seed management
- `src/data.py` - RVL-CDIP loading (sample/full modes), preprocessing transforms, visualization utilities, and 16-class labels

## Decisions Made

**1. Dataset source: chainyo/rvl-cdip instead of aharley/rvl_cdip**
- Rationale: HuggingFace datasets v4.x dropped support for loading scripts. The aharley/rvl_cdip dataset uses legacy loading scripts that are no longer supported. chainyo/rvl-cdip is a modern parquet-based format that works with current datasets library.

**2. Split name normalization: val -> validation**
- Rationale: The chainyo/rvl-cdip dataset uses "val" split name, but for consistency across the codebase we normalize to "validation" at the data loading layer. This provides a consistent API regardless of dataset source.

**3. Device detection pattern: MPS -> CUDA -> CPU**
- Rationale: Prioritizes Apple Silicon GPU (MPS) when available, falls back to NVIDIA CUDA, then CPU. This supports development on MacBook (MPS) and cluster training (CUDA) without config changes.

**4. Sample mode streaming with shuffle**
- Rationale: Streaming mode with `.shuffle(buffer_size=10000).take(N)` provides random sampling without downloading full dataset. Essential for local development with subset-first approach.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed Python version constraint for torchvision compatibility**
- **Found during:** Task 1 (poetry add dependencies)
- **Issue:** pyproject.toml had `requires-python = ">=3.13"` which conflicted with torchvision requirement of `!=3.14.1,>=3.10`
- **Fix:** Changed to `requires-python = ">=3.13,<3.14.1"` to satisfy torchvision constraints
- **Files modified:** pyproject.toml
- **Verification:** poetry add succeeded after fix
- **Committed in:** 6feff18 (Task 1 commit)

**2. [Rule 1 - Bug] Used modern dataset source to avoid deprecated loading scripts**
- **Found during:** Task 2 (load_dataset verification)
- **Issue:** aharley/rvl_cdip uses deprecated loading script format that errors with "Dataset scripts are no longer supported" in datasets v4.x
- **Fix:** Changed dataset_name from "aharley/rvl_cdip" to "chainyo/rvl-cdip" which uses modern parquet format
- **Files modified:** src/config.py, src/data.py
- **Verification:** load_dataset succeeded, sample data loaded correctly
- **Committed in:** 5f273f4 (Task 2 commit)

**3. [Rule 1 - Bug] Normalized dataset split names**
- **Found during:** Task 2 (load_dataset verification)
- **Issue:** chainyo/rvl-cdip uses "val" split name instead of "validation", causing KeyError when accessing dataset["validation"]
- **Fix:** Added split name normalization logic to map "val" -> "validation" at data loading layer for consistent API
- **Files modified:** src/data.py
- **Verification:** All three splits (train/validation/test) accessible with normalized names
- **Committed in:** 5f273f4 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking)
**Impact on plan:** All auto-fixes necessary for correctness and compatibility with current library versions. No scope creep.

## Issues Encountered
- Python version incompatibility with torchvision - resolved by constraining Python version
- Legacy dataset loading script deprecated - resolved by switching to modern parquet-based dataset source
- Split name inconsistency - resolved with normalization layer

## User Setup Required

None - no external service configuration required. All configuration is environment-driven with sensible defaults.

## Next Phase Readiness

Ready for Phase 01 Plan 02 (Feature Extraction):
- Data loading infrastructure complete
- Preprocessing pipeline tested and working
- Config system supports both local (sample mode) and cluster (full mode) workflows
- Device auto-detection enables seamless local development and cluster training

No blockers. Ready to proceed with ResNet-50 feature extraction.

---
*Phase: 01-notebook-foundation-data-features*
*Completed: 2026-03-25*
