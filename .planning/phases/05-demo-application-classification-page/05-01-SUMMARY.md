---
phase: 05-demo-application-classification-page
plan: 01
subsystem: model-architecture
tags: [graphsage, gat, gnn, pytorch-geometric, attention-pooling, bag-of-characters]

# Dependency graph
requires:
  - phase: 02.1-hybrid-fusion-positional-encoding
    provides: HybridGraphSAGE class and graph construction functions
  - phase: 02.2-text-aware-hybrid-gnn
    provides: TextAwareGraphSAGE class and text-aware graph construction
provides:
  - FusionGraphSAGE, FusionGAT, GatedBoCGraphSAGE, AttentionPoolFusionSAGE model classes
  - feature_map_to_graph_gated_boc graph construction function
  - All 6 GNN model checkpoints loadable from src/model.py
affects: [05-demo-application-classification-page, model-registry, inference-pipeline]

# Tech tracking
tech-stack:
  added: [GATConv from torch_geometric, torch_geometric.utils.scatter, torch_geometric.utils.softmax]
  patterns: [gated-text-fusion, attention-pooling, sequential-classifier-fusion]

key-files:
  created: []
  modified:
    - src/model.py
    - src/graph.py

key-decisions:
  - "exp26 text_gate is plain nn.Linear with sigmoid in forward (not nn.Sequential with Sigmoid) per checkpoint keys"
  - "exp27 attn_key uses bias=False per checkpoint lacking attn_key.bias key"
  - "exp25 HybridGraphSAGE loads with node_dim=2120 (2048 CNN + 2 PE + 70 BoC concatenated)"
  - "Used torch_geometric.utils.scatter instead of torch_scatter (not installed, same API)"

patterns-established:
  - "Checkpoint-first architecture: inspect state_dict keys before writing model class"
  - "Fusion classifier patterns: Sequential (exp16/23) vs BatchNorm+Linear (exp25/26/27)"

# Metrics
duration: 8min
completed: 2026-04-07
---

# Phase 5 Plan 1: Promote Model Architectures Summary

**4 GNN model classes (FusionGraphSAGE, FusionGAT, GatedBoCGraphSAGE, AttentionPoolFusionSAGE) promoted from experiment notebooks to src/model.py with verified checkpoint loading**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-07T07:35:30Z
- **Completed:** 2026-04-07T07:43:30Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- All 4 missing model architecture classes added to src/model.py with exact state_dict key compatibility
- feature_map_to_graph_gated_boc added to src/graph.py for separate BoC attribute graph construction
- All 6 model checkpoints (exp14b CNN, exp16, exp23, exp25, exp26, exp27) verified loadable from src/ modules

## Task Commits

Each task was committed atomically:

1. **Task 1: Add FusionGraphSAGE, FusionGAT, GatedBoCGraphSAGE, AttentionPoolFusionSAGE** - `c38974c` (feat)
2. **Task 2: Add feature_map_to_graph_gated_boc** - `9efe807` (feat)

## Files Created/Modified
- `src/model.py` - Added 4 model classes: FusionGraphSAGE (Sequential fusion), FusionGAT (multi-head attention), GatedBoCGraphSAGE (learned text gating), AttentionPoolFusionSAGE (query-key attention pooling)
- `src/graph.py` - Added feature_map_to_graph_gated_boc (feature-kNN graph with separate x_boc attribute)

## Decisions Made
- exp26 text_gate implemented as plain nn.Linear with sigmoid applied in forward(), not nn.Sequential -- confirmed by checkpoint keys showing text_gate.weight/bias (not text_gate.0.weight)
- exp27 attn_key created with bias=False -- checkpoint only has attn_key.weight, no attn_key.bias
- Used torch_geometric.utils.scatter instead of torch_scatter package (not installed, equivalent API)
- exp25 checkpoint uses node_dim=2120 (BoC concatenated into node features), loads via existing HybridGraphSAGE with different constructor args

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Replaced torch_scatter with torch_geometric.utils.scatter**
- **Found during:** Task 1 (model class implementation)
- **Issue:** torch_scatter package not installed, import failed
- **Fix:** Used torch_geometric.utils.scatter which provides the same API and is already available
- **Files modified:** src/model.py
- **Verification:** All model imports and checkpoint loading succeed
- **Committed in:** c38974c (Task 1 commit)

**2. [Rule 1 - Bug] Fixed attn_key bias mismatch in AttentionPoolFusionSAGE**
- **Found during:** Task 1 verification
- **Issue:** nn.Linear defaults to bias=True but checkpoint has no attn_key.bias key
- **Fix:** Set bias=False on attn_key Linear layer
- **Files modified:** src/model.py
- **Verification:** model.load_state_dict succeeds for exp27 checkpoint
- **Committed in:** c38974c (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes required for correct operation. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All model architecture classes available in src/model.py for the demo app's model registry
- Graph construction functions complete in src/graph.py for inference pipeline
- Demo app can now import and instantiate any of the 7 model classes (3 existing + 4 new)

---
*Phase: 05-demo-application-classification-page*
*Completed: 2026-04-07*
