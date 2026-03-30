# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-25)

**Core value:** The GraphSAGE model must demonstrate whether graph-based spatial reasoning improves document classification over CNN-only baselines — with clear, reproducible evidence.
**Current focus:** Phase 2.1 - Hybrid Fusion + Positional Encoding

## Current Position

Phase: 2.1 of 4 (Hybrid Fusion + Positional Encoding)
Plan: 2 of TBD (paused at checkpoint)
Status: In progress — awaiting human verification
Last activity: 2026-03-30 — Completed Task 1 of 02.1-02-PLAN.md, paused at human-verify checkpoint

Progress: [████░░░░░░] ~35%

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 8min
- Total execution time: 0.3 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-notebook-foundation-data-features | 1 | 12min | 12min |
| 02.1-hybrid-fusion-positional-encoding | 2 | 6min | 3min |

**Recent Trend:**
- Last 5 plans: 12min, 4min
- Trend: Accelerating (4min vs 12min avg)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Subset-first development: Iterate on architecture locally (5-10% data) before full training on cluster
- CNN baseline from reference notebook: Avoid duplicating work, use existing baseline for comparison
- Graph construction as experimentation: Finding the right approach is part of the contribution
- Incremental notebook building: Each phase adds sections to a single Jupyter notebook that grows phase by phase
- **[01-01]** Dataset source: chainyo/rvl-cdip instead of aharley/rvl_cdip (modern parquet format compatible with datasets v4.x)
- **[01-01]** Device detection pattern: MPS -> CUDA -> CPU (supports local Mac dev and cluster training)
- **[01-01]** Sample mode with streaming + shuffle for random sampling without full download
- **[02.1-01]** Single cache format for all notebooks (hybrid notebooks use global_feat, plain GraphSAGE ignores it)
- **[02.1-01]** Normalized 2D coordinates (2 dims) over sinusoidal PE (64+ dims) to minimize overfitting
- **[02.1-01]** Store global_feat as [1, 2048] for correct PyG batching to [batch_size, 2048]
- **[02.1-01]** Direct concatenation fusion without CNN projection (parameter-efficient on 2560 samples)

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-03-30 06:19 UTC
Stopped at: 02.1-02-PLAN.md Task 1 complete (477d36d), paused at Task 2 (human-verify checkpoint)
Resume file: None

**Note:** Baseline notebook must be re-run to regenerate cache with global_feat before hybrid notebook can be executed end-to-end. This is a prerequisite for the human verification checkpoint.
