# Phase 5: Demo Application - Classification Page - Research

**Researched:** 2026-04-06
**Domain:** FastAPI + Jinja2 + HTMX web application with PyTorch inference
**Confidence:** HIGH

## Summary

This phase builds a FastAPI web application that serves a classification demo page. The stack is FastAPI + Jinja2 + HTMX + Alpine.js + Tailwind CSS, decomposed from an existing Stitch-generated HTML prototype at `/tmp/stitch-demo.html`. The application loads 7 model checkpoints at startup, accepts document image uploads (or preloaded samples), runs CNN feature extraction + graph construction + multi-model inference, and returns results as HTMX partial HTML fragments.

The critical discovery is that 4 of 7 model architectures (exp16/FusionGraphSAGE, exp23/FusionGAT, exp25/BocFusion, exp26/GatedBoCGraphSAGE, exp27/AttnPoolGraphSAGE) exist ONLY in notebook experiment code, not in `src/model.py`. Before the demo can load these checkpoints, their model classes must be promoted to `src/model.py`. Similarly, `feature_map_to_graph_gated_boc` from exp26 is referenced but not in `src/graph.py`. This is a prerequisite blocker.

**Primary recommendation:** First promote all model classes to `src/model.py`, then build the FastAPI app using jinja2-fragments for HTMX partial rendering, with models loaded once at startup via FastAPI lifespan events.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| fastapi | latest | Web framework, routing, request handling | Specified in requirements |
| jinja2 | latest | Server-side HTML templating | FastAPI's native template engine |
| jinja2-fragments | >=1.10 | Render individual Jinja2 blocks as HTMX partials | Enables Locality of Behavior -- one template file serves both full page and fragments |
| uvicorn | latest | ASGI server | Standard FastAPI server |
| python-multipart | latest | File upload support (UploadFile) | Required by FastAPI for form/file endpoints |
| htmx | 2.0 (CDN) | Client-side partial DOM swaps | Specified in requirements, no JS framework needed |
| alpinejs | 3.x (CDN) | Accordion state, small client interactions | Specified in requirements |
| tailwindcss | CDN play | Styling with MD3 tokens | Prototype already uses cdn.tailwindcss.com with custom config |

### Supporting (already in project)
| Library | Purpose | When to Use |
|---------|---------|-------------|
| torch | Model inference | Loading checkpoints, running forward pass |
| torch-geometric | GNN inference | SAGEConv, GATConv, global_mean_pool |
| torchvision | ResNet-50 feature extraction | create_feature_extractor, transforms |
| matplotlib | Heatmap generation | Activation heatmaps, text density maps |
| Pillow | Image processing | Upload handling, resize, convert |
| numpy | Array ops | Heatmap overlays |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| jinja2-fragments | Separate partial template files | Fragments keep full page + partials in one file (better locality); separate files are more traditional but fragment logic across files |
| Tailwind CDN | Built Tailwind CSS | CDN is faster to set up for a demo app; built CSS is better for production but overkill for academic demo |
| matplotlib heatmaps | Raw canvas/SVG | matplotlib generates server-side images easily; client-side rendering adds complexity |

**Installation (new dependencies only):**
```bash
poetry add fastapi uvicorn jinja2 jinja2-fragments python-multipart
```

## Architecture Patterns

### Recommended Project Structure
```
app/
  src/
    __init__.py
    main.py              # FastAPI app factory, lifespan, mount static/templates
    config.py             # App config (model paths, sample paths, env vars)
    routes/
      __init__.py
      classify.py         # GET / (full page), POST /classify (HTMX partials)
    services/
      __init__.py
      model_registry.py   # Load all models at startup, store in app.state
      inference.py        # Single-image pipeline: preprocess -> extract -> graph -> predict
      visualization.py    # Heatmap generation, graph SVG, probability bars
    templates/
      base.html           # Full page layout (header, left panel, right panel)
      partials/
        cnn_features.html     # Section 1: activation heatmap + text density
        graph_construction.html # Section 2: grid vs kNN SVG + stats
        model_predictions.html  # Section 3: comparison table
        detailed_analysis.html  # Section 4: accordion per model
        upload_zone.html        # Upload area + sample thumbnails
    static/
      css/
        app.css           # Tailwind config + MD3 tokens (from Stitch prototype)
      samples/            # 5 preloaded RVL-CDIP sample images
      js/
        app.js            # Minimal JS (if needed beyond HTMX/Alpine)
  tests/
    __init__.py
    conftest.py           # TestClient fixture, mock model registry
    test_routes.py
    test_inference.py
    test_visualization.py
```

### Pattern 1: FastAPI Lifespan for Model Loading
**What:** Load all model checkpoints once at startup using FastAPI's lifespan context manager, store in `app.state`.
**When to use:** Always -- models are expensive to load, must persist across requests.
**Example:**
```python
# Source: FastAPI official docs (lifespan events)
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup
    app.state.models = load_all_models()
    app.state.feature_extractor = load_feature_extractor()
    yield
    # Cleanup (optional)

app = FastAPI(lifespan=lifespan)
```

### Pattern 2: HTMX Partial Rendering with jinja2-fragments
**What:** Single template file with named blocks; endpoint renders either full page or specific block based on HX-Request header.
**When to use:** Every endpoint that serves both initial page load and HTMX updates.
**Example:**
```python
# Source: jinja2-fragments docs + FastAPI templates docs
from jinja2_fragments.fastapi import Jinja2Blocks

templates = Jinja2Blocks(directory="app/src/templates")

@router.post("/classify")
async def classify(request: Request):
    # ... run inference pipeline ...
    context = {"results": results, "request": request}

    # Return only the results blocks for HTMX requests
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            "base.html",
            context,
            block_name="results_section",  # renders just this block
        )
    # Full page for non-HTMX requests
    return templates.TemplateResponse("base.html", context)
```

### Pattern 3: Base64-Encoded Visualization Images
**What:** Generate matplotlib figures server-side, encode as base64 data URIs, embed directly in HTML.
**When to use:** Activation heatmaps, text density maps -- any server-generated visualization.
**Example:**
```python
import base64
import io
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend -- critical for server
import matplotlib.pyplot as plt

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)  # Prevent memory leak
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# In template: <img src="data:image/png;base64,{{ heatmap_b64 }}">
```

### Pattern 4: Alpine.js Accordion for Detailed Analysis
**What:** Alpine.js manages which accordion item is expanded, with x-show for content visibility.
**When to use:** The "Detailed Model Analysis" section with per-model expandable panels.
**Example:**
```html
<!-- Source: Alpine.js docs + Stitch prototype pattern -->
<div x-data="{ open: null }">
  {% for model in models %}
  <div>
    <button @click="open = open === {{ loop.index }} ? null : {{ loop.index }}">
      {{ model.name }}
      <span x-text="open === {{ loop.index }} ? 'expand_less' : 'expand_more'"
            class="material-symbols-outlined"></span>
    </button>
    <div x-show="open === {{ loop.index }}" x-collapse>
      <!-- 16-class bar chart + node importance + metadata -->
    </div>
  </div>
  {% endfor %}
</div>
```

### Pattern 5: SVG Graph Visualization (Server-Generated)
**What:** Generate SVG markup for 49-node graphs directly in Python, return as template variable.
**When to use:** Graph construction section showing grid vs k-NN topology.
**Example:**
```python
def generate_graph_svg(edge_index, grid_h=7, grid_w=7, width=280, height=280):
    """Generate SVG string for a 7x7 node graph with edges."""
    padding = 20
    step_x = (width - 2 * padding) / (grid_w - 1)
    step_y = (height - 2 * padding) / (grid_h - 1)

    lines = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']
    # Draw edges
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        sx, sy = padding + (src % grid_w) * step_x, padding + (src // grid_w) * step_y
        tx, ty = padding + (tgt % grid_w) * step_x, padding + (tgt // grid_w) * step_y
        lines.append(f'<line x1="{sx}" y1="{sy}" x2="{tx}" y2="{ty}" stroke="#c5c5d4" stroke-width="0.5"/>')
    # Draw nodes
    for node in range(grid_h * grid_w):
        cx = padding + (node % grid_w) * step_x
        cy = padding + (node // grid_w) * step_y
        lines.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#24389c"/>')
    lines.append('</svg>')
    return '\n'.join(lines)
```

### Anti-Patterns to Avoid
- **Loading models per-request:** Models are 90-260MB each. Loading on every request would make response time 30+ seconds. Load once at startup.
- **Using Streamlit:** The requirements specify FastAPI. Streamlit would be simpler but doesn't meet the spec.
- **Client-side JS frameworks (React/Vue):** HTMX + Alpine.js is the specified stack. Adding a JS framework defeats the server-rendered architecture.
- **Saving temp files to disk:** Generate all visualizations in-memory (BytesIO for matplotlib, string for SVG). No temp file cleanup needed.
- **Running doctr text detection at demo time:** doctr DBNet is slow (~2-5s per image on CPU) and requires MPS->CPU fallback. For the demo, compute text density live but warn about latency, or pre-compute for sample images.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Template fragment rendering | Custom header-checking middleware | jinja2-fragments `Jinja2Blocks` | Handles HX-Request detection, block rendering, multi-block concat |
| File upload handling | Manual multipart parsing | FastAPI `UploadFile` + `python-multipart` | Handles streaming, temp files, content-type validation |
| Image preprocessing | Manual resize/normalize pipeline | `torchvision.transforms.Compose` with `ResNet50_Weights.IMAGENET1K_V2.transforms()` | Matches exact normalization used during training |
| CORS/security | Custom middleware | FastAPI built-in `CORSMiddleware` | Not needed for localhost demo, but available if needed |
| Accordion animation | Custom CSS transitions | Alpine.js `x-collapse` plugin | Smooth height animation with zero custom code |

## Common Pitfalls

### Pitfall 1: Model Classes Not in src/model.py
**What goes wrong:** Attempting to load checkpoints for exp16, exp23, exp25, exp26, exp27 fails because the model architecture classes only exist in notebook cells, not in importable Python modules.
**Why it happens:** Experiments iterated in notebooks; model classes were never promoted to `src/`.
**How to avoid:** Before building the app, promote ALL model classes to `src/model.py`:
- `FusionGraphSAGE` (exp16) -- uses SAGEConv, feature-kNN, in_channels=2048
- `FusionGAT` (exp23) -- uses GATConv with 4 heads, feature-kNN, in_channels=2048
- `HybridGraphSAGE` (already in src/model.py) -- node_dim=2050 (CNN+PE)
- `TextAwareGraphSAGE` (already in src/model.py) -- node_dim=2051 (CNN+PE+text_density)
- `BocFusionGraphSAGE` (exp25) -- node_dim=2120 (CNN+PE+BoC70)
- `GatedBoCGraphSAGE` (exp26) -- cnn_dim=2050, boc_dim=70, proj_dim=16
- `AttnPoolGraphSAGE` (exp27) -- node_dim=2050, attention pooling replaces global_mean_pool
**Warning signs:** ImportError or KeyError when loading state_dict.

### Pitfall 2: Graph Construction Functions Missing for Some Models
**What goes wrong:** `feature_map_to_graph_gated_boc` is referenced in exp26 notebook but not in `src/graph.py`.
**How to avoid:** Promote graph construction variants to `src/graph.py` alongside model classes.

### Pitfall 3: Checkpoint Format Inconsistency
**What goes wrong:** Different checkpoints use different save formats:
- `best_model_resnet50_03.pt`: dict with `model_state_dict`, `optimizer_state_dict`, `epoch`, etc.
- `exp14b_finetuned_resnet50_cnn.pt`: raw state_dict (keys start with `conv1.weight`)
- `exp16-27 GNN models`: raw state_dict (keys start with `conv1.lin_l.weight`)
**How to avoid:** Write a unified `load_checkpoint` function that detects format and extracts state_dict consistently. Use `_unwrap_resnet_checkpoint` pattern from `src/features.py` as reference.

### Pitfall 4: Model Input Dimension Mismatch
**What goes wrong:** Each model expects different node feature dimensions:
- exp16 FusionGraphSAGE: in_channels=2048 (raw CNN features, feature-kNN)
- exp23 FusionGAT: in_channels=2048 (raw CNN features, feature-kNN)
- HybridGraphSAGE: node_dim=2050 (CNN 2048 + PE 2)
- TextAwareGraphSAGE: node_dim=2051 (CNN 2048 + PE 2 + text_density 1)
- exp25 BocFusion: node_dim=2120 (CNN 2048 + PE 2 + BoC 70)
- exp26 GatedBoCGraphSAGE: cnn_dim=2050 + boc_dim=70 (separate inputs)
- exp27 AttnPoolGraphSAGE: node_dim=2050 (CNN 2048 + PE 2)
**How to avoid:** Each model entry in the registry must declare its required feature preparation pipeline. Build a `ModelSpec` dataclass that encodes: model class, constructor args, checkpoint path, required graph construction function, feature dimensions.

### Pitfall 5: Matplotlib Memory Leak in Server
**What goes wrong:** Not calling `plt.close(fig)` after generating heatmaps causes matplotlib to accumulate figures in memory, eventually OOMing the server.
**How to avoid:** Always use `plt.close(fig)` after converting to bytes. Better: use `matplotlib.use("Agg")` at module import and always create figures with explicit `fig, ax = plt.subplots()` pattern, never `plt.figure()`.

### Pitfall 6: MPS Device for doctr
**What goes wrong:** doctr DBNet does not support MPS. If device is MPS, text density extraction will crash.
**How to avoid:** Already handled in `src/text_features.py` via `_get_detection_device()` which falls back to CPU. Reuse this function.

### Pitfall 7: Total Model Memory at Startup
**What goes wrong:** Loading all 7 models into memory: ~370MB on disk, ~500-700MB in RAM (model parameters + buffers).
**Why it happens:** ResNet-50 backbone (90-259MB) is the largest. GNN models are small (4.5-8.5MB each).
**How to avoid:** Only load one ResNet-50 feature extractor (from exp14b fine-tuned checkpoint, 90MB). The CNN baseline prediction can share this backbone. GNN models are lightweight. Total RAM for all models: ~200MB (one ResNet-50 + 5 GNN heads). This is fine for a laptop demo.

### Pitfall 8: Inference Timing Budget
**What goes wrong:** Pipeline exceeds the 10-second target.
**Expected timing breakdown (laptop, MPS/CPU):**
- Image preprocessing: ~10ms
- ResNet-50 feature extraction: ~50-100ms (MPS)
- Text density (doctr, CPU): ~1-3s (this is the bottleneck)
- OCR/Tesseract for BoC: ~1-2s (if needed for exp25/26)
- Graph construction: ~5ms per model
- GNN inference (5 models): ~50ms total
- Visualization generation: ~200-500ms
- Total estimate: 2-6 seconds (within budget)
**How to avoid:** Run feature extraction once, share across all models. Pre-compute Tesseract OCR once. For sample images, consider caching the OCR results.

## Code Examples

### Unified Model Registry Pattern
```python
from dataclasses import dataclass
from typing import Callable
from pathlib import Path

@dataclass
class ModelSpec:
    name: str
    display_name: str
    model_type: str  # "CNN" | "GNN" | "GNN+OCR"
    model_class: type
    constructor_kwargs: dict
    checkpoint_path: Path
    graph_builder: str  # function name in graph.py
    needs_global_feat: bool
    needs_boc: bool
    needs_text_density: bool

MODEL_SPECS = [
    ModelSpec(
        name="cnn_baseline",
        display_name="CNN Baseline (ResNet-50)",
        model_type="CNN",
        # ... CNN uses ResNet-50 fc head directly
    ),
    ModelSpec(
        name="hybrid_graphsage",
        display_name="Hybrid GraphSAGE",
        model_type="GNN",
        model_class=HybridGraphSAGE,
        constructor_kwargs={"node_dim": 2050, "hidden_channels": 256, "embed_channels": 128},
        checkpoint_path=Path("models/exp16_fusion_featknn_graphsage.pt"),
        graph_builder="feature_knn",
        needs_global_feat=True,
        needs_boc=False,
        needs_text_density=False,
    ),
    # ... etc for all 7 models
]
```

### FastAPI UploadFile to PIL Image
```python
# Source: FastAPI docs (Request Files)
from fastapi import UploadFile
from PIL import Image
import io

async def upload_to_pil(file: UploadFile) -> Image.Image:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
```

### Activation Heatmap from Layer4 Features
```python
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

def generate_activation_heatmap(layer4_features: torch.Tensor, original_image: Image.Image) -> str:
    """Generate base64-encoded activation heatmap overlay.

    Args:
        layer4_features: [2048, 7, 7] from ResNet-50 layer4
        original_image: Original document image for overlay

    Returns:
        Base64-encoded PNG string
    """
    # Mean across channel dimension -> [7, 7]
    activation = layer4_features.mean(dim=0).numpy()

    # Normalize to [0, 1]
    activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)

    # Resize to match image dimensions
    img_array = np.array(original_image.resize((224, 224)))

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=100)
    ax.imshow(img_array, cmap="gray" if img_array.ndim == 2 else None)
    ax.imshow(activation, cmap="jet", alpha=0.4,
              extent=[0, 224, 224, 0], interpolation="bilinear")
    ax.axis("off")

    b64 = fig_to_base64(fig)
    return b64
```

### Text Density Grid Visualization
```python
def generate_text_density_html(text_density: torch.Tensor, primary_color: str = "#24389c") -> str:
    """Generate inline HTML for a 7x7 text density grid.

    Args:
        text_density: [7, 7] tensor with values in [0, 1]

    Returns:
        HTML string with 49 colored divs in a CSS grid
    """
    cells = []
    for i in range(49):
        row, col = divmod(i, 7)
        opacity = text_density[row, col].item()
        cells.append(
            f'<div class="aspect-square rounded-sm" '
            f'style="background: {primary_color}; opacity: {opacity:.2f}"></div>'
        )
    return '\n'.join(cells)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| FastAPI `on_event("startup")` | `lifespan` async context manager | FastAPI 0.93+ | Startup/shutdown events are deprecated; lifespan is the current pattern |
| Separate partial template files | jinja2-fragments block rendering | 2023+ | One file serves full page + HTMX fragments |
| `plt.figure()` global state | `fig, ax = plt.subplots()` explicit | Best practice | Prevents cross-request state pollution in server context |
| Tailwind CSS build step | Tailwind CDN play mode | Tailwind 3.x+ | Fine for demos; production needs build |

## Open Questions

1. **Which models to include in the demo?**
   - What we know: 7 checkpoints exist. The CNN baseline (exp14b) + 5 GNN variants + 1 older CNN (exp03).
   - What's unclear: Does the presenter want all 7, or a curated subset? The exp03 checkpoint (259MB) uses a different save format and may be redundant with exp14b.
   - Recommendation: Include exp14b (CNN baseline) + exp16 (HybridGraphSAGE) + exp23 (GAT) + exp25 (BoC SAGE) + exp26 (Gated BoC) + exp27 (Attn Pool). Skip exp03 to save memory and because exp14b is the better CNN baseline.

2. **Pre-computing OCR for sample images**
   - What we know: Tesseract OCR takes 1-2s per image. For preloaded samples, this is wasteful to compute each time.
   - Recommendation: Cache Tesseract BoC features alongside sample images as .pt files at app build time.

3. **Model class promotion scope**
   - What we know: exp16/exp23 FusionGraphSAGE/FusionGAT use a `nn.Sequential` classifier pattern with 2 linear layers (in=2176, hidden=512, out=16), while the existing `HybridGraphSAGE` in model.py uses a single Linear. The architectures are structurally different.
   - Recommendation: Promote each as a distinct class. Do not try to unify them under one parameterized class -- the differences (GATConv heads, gated BoC inputs, attention pooling) are too varied.

4. **Correct class label ordering**
   - What we know: Labels are loaded dynamically from HuggingFace dataset features in `src/data.py`. For the demo, we need hardcoded labels (can't require HF dataset at demo time).
   - From notebooks: `["letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication", "specification", "file folder", "news article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"]`
   - Recommendation: Hardcode these 16 labels in app config, matching the training label order.

## Sources

### Primary (HIGH confidence)
- `src/model.py` -- Existing model architectures (GraphSAGEClassifier, HybridGraphSAGE, TextAwareGraphSAGE)
- `src/features.py` -- ResNet-50 feature extraction, checkpoint loading patterns
- `src/graph.py` -- Graph construction functions (grid, feature-kNN, hybrid, text-aware)
- `src/text_features.py` -- Text density extraction with doctr, MPS fallback
- `src/ocr_features.py` -- Tesseract OCR + BoC feature computation
- `/tmp/stitch-demo.html` -- Complete UI prototype (428 lines, Tailwind + MD3 tokens)
- Model checkpoint inspection -- verified all 7 checkpoint formats and state_dict keys
- Notebook experiment code -- `.lab/workspace/exp_23_gnn_arch_comparison.ipynb`, `exp_25_text_gatv2.ipynb`, `exp_26_gated_boc.ipynb`, `exp_27_attn_pool.ipynb`

### Secondary (MEDIUM confidence)
- [FastAPI Templates docs](https://fastapi.tiangolo.com/advanced/templates/) -- Jinja2Templates setup, StaticFiles mounting
- [jinja2-fragments PyPI](https://pypi.org/project/jinja2-fragments/) -- Block rendering for HTMX partials
- [HTMX Template Fragments essay](https://htmx.org/essays/template-fragments/) -- Pattern for partial rendering
- [Alpine.js Accordion](https://alpinejs.dev/component/accordion) -- x-data/x-show/x-collapse pattern

### Tertiary (LOW confidence)
- WebSearch results on FastAPI+HTMX project structure -- patterns confirmed via official docs
- WebSearch results on matplotlib base64 encoding -- standard pattern, well-documented

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - FastAPI/Jinja2/HTMX are well-documented, versions stable
- Architecture: HIGH - Based on actual codebase inspection and verified prototype
- Model loading: HIGH - Inspected all 7 checkpoint files, verified state_dict structures
- Pitfalls: HIGH - Discovered critical blockers (missing model classes) through direct code inspection
- Visualization: MEDIUM - matplotlib base64 pattern is standard but graph SVG generation is custom

**Research date:** 2026-04-06
**Valid until:** 2026-04-17 (project deadline)
