# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-25)

**Core value:** The GraphSAGE model must demonstrate whether graph-based spatial reasoning improves document classification over CNN-only baselines — with clear, reproducible evidence.
**Current focus:** Phase 2.2 - Text-Aware Hybrid GNN

## Current Position

Phase: 2.2 of 4 (Text-Aware Hybrid GNN)
Plan: 1 of TBD (completed)
Status: In progress
Last activity: 2026-03-30 — Completed 02.2-01-PLAN.md (text density extraction + text-aware graph + TextAwareGraphSAGE)

Progress: [█████░░░░░] ~45%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 8min
- Total execution time: 0.4 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-notebook-foundation-data-features | 1 | 12min | 12min |
| 02.1-hybrid-fusion-positional-encoding | 2 | 6min | 3min |
| 02.2-text-aware-hybrid-gnn | 1 | 7min | 7min |

**Recent Trend:**
- Last 5 plans: 12min, 4min, 7min
- Trend: Stable ~7min

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
- **[02.2-01]** Raw DBNet probability map (return_model_output=True) over bounding-box reconstruction for text density
- **[02.2-01]** MPS falls back to CPU for doctr inference (doctr lacks MPS support)
- **[02.2-01]** TextAwareGraphSAGE as independent class (not subclass of HybridGraphSAGE) for independent evolution

### Pending Todos

- poetry.lock not regenerated via `poetry add` (SSL cert issue). Package installed in venv and declared in pyproject.toml. Run `poetry lock --no-update` in a stable network environment.

### Blockers/Concerns

- SSL certificate verification fails in current environment (corporate/self-signed cert). Workaround in place (ssl bypass + curl --insecure for model weights). Does not affect runtime.

## Session Continuity

Last session: 2026-03-30 14:31 UTC
Stopped at: 02.2-01-PLAN.md complete (1e31ebb)
Resume file: None

**Note:** Baseline notebook must be re-run to regenerate cache with global_feat before hybrid notebook can be executed end-to-end. This is a prerequisite for the 02.1-02 human verification checkpoint. Phase 02.2-02 (notebook integration) is the next plan to execute.
