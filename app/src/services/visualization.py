import io
import base64
from html import escape

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_activation_heatmap(
    layer4_features: torch.Tensor, original_image: Image.Image
) -> str:
    activation = layer4_features.mean(dim=0).detach().cpu().numpy()
    vmin, vmax = activation.min(), activation.max()
    if vmax - vmin > 0:
        activation = (activation - vmin) / (vmax - vmin)
    else:
        activation = activation * 0.0

    resized = original_image.resize((224, 224), Image.LANCZOS).convert("RGB")

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(resized)
    ax.imshow(activation, cmap="jet", alpha=0.4, interpolation="bilinear", extent=[0, 224, 224, 0])
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig_to_base64(fig)


def generate_text_density_html(
    text_density: torch.Tensor | None, primary_color: str = "#24389c"
) -> str:
    if text_density is None:
        return (
            '<div class="flex items-center justify-center h-full text-slate-400 font-label text-xs">'
            "Text density not available</div>"
        )

    density = text_density.detach().cpu().float()
    vmin, vmax = density.min(), density.max()
    if vmax - vmin > 0:
        density = (density - vmin) / (vmax - vmin)
    else:
        density = density * 0.0

    values = density.flatten().tolist()
    cells = []
    for v in values:
        opacity = max(0.05, min(1.0, v))
        cells.append(
            f'<div class="aspect-square rounded-sm" '
            f'style="background:{primary_color};opacity:{opacity:.2f}"></div>'
        )
    return (
        '<div class="grid grid-cols-7 grid-rows-7 gap-1 w-full h-full">'
        + "".join(cells)
        + "</div>"
    )


def generate_graph_svg(
    edge_index: torch.Tensor,
    grid_h: int = 7,
    grid_w: int = 7,
    width: int = 280,
    height: int = 280,
    edge_color: str = "#c5c5d4",
    node_color: str = "#24389c",
) -> str:
    padding = 20
    inner_w = width - 2 * padding
    inner_h = height - 2 * padding

    num_nodes = grid_h * grid_w
    positions = []
    for row in range(grid_h):
        for col in range(grid_w):
            x = padding + col * inner_w / max(grid_w - 1, 1)
            y = padding + row * inner_h / max(grid_h - 1, 1)
            positions.append((x, y))

    edges_cpu = edge_index.detach().cpu()
    src_nodes = edges_cpu[0].tolist()
    dst_nodes = edges_cpu[1].tolist()

    lines = []
    for s, d in zip(src_nodes, dst_nodes):
        if 0 <= s < num_nodes and 0 <= d < num_nodes:
            x1, y1 = positions[s]
            x2, y2 = positions[d]
            lines.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{escape(edge_color)}" stroke-width="1"/>'
            )

    circles = []
    for x, y in positions:
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{escape(node_color)}"/>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">'
        + "".join(lines)
        + "".join(circles)
        + "</svg>"
    )


def generate_graph_overlay(
    original_image: Image.Image,
    edge_index: torch.Tensor,
    grid_h: int = 7,
    grid_w: int = 7,
    edge_color: str = "#2563eb",
    node_color: str = "#24389c",
    title: str = "",
) -> str:
    """Overlay graph edges and nodes on the actual document image (GVdoc style)."""
    img = original_image.resize((224, 224), Image.LANCZOS).convert("RGB")
    img_array = np.array(img)

    num_nodes = grid_h * grid_w
    cell_h = 224 / grid_h
    cell_w = 224 / grid_w

    # Node positions at grid cell centers
    positions = []
    for row in range(grid_h):
        for col in range(grid_w):
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            positions.append((cx, cy))

    edges_cpu = edge_index.detach().cpu()
    src_nodes = edges_cpu[0].tolist()
    dst_nodes = edges_cpu[1].tolist()

    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(img_array)

    # Draw edges
    for s, d in zip(src_nodes, dst_nodes):
        if 0 <= s < num_nodes and 0 <= d < num_nodes:
            x1, y1 = positions[s]
            x2, y2 = positions[d]
            ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=0.5, alpha=0.4)

    # Draw nodes
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    ax.scatter(xs, ys, c=node_color, s=8, zorder=5, edgecolors="white", linewidths=0.3)

    if title:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=4)

    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0, right=1, top=0.93 if title else 1, bottom=0)
    return fig_to_base64(fig)


def generate_probability_bars_html(
    probabilities: list[float], labels: list[str], top_n: int = 3
) -> str:
    paired = sorted(zip(probabilities, labels), key=lambda x: x[0], reverse=True)
    top = paired[:top_n]

    rows = []
    for i, (prob, label) in enumerate(top):
        pct = prob * 100
        bar_color = "bg-primary" if i == 0 else "bg-slate-300"
        text_color = "font-bold" if i == 0 else "font-bold text-slate-400"
        rows.append(
            f'<div class="flex items-center gap-4">'
            f'<span class="w-20 font-label text-xs truncate">{escape(label)}</span>'
            f'<div class="flex-1 h-3 bg-white rounded-full overflow-hidden">'
            f'<div class="h-full {bar_color} rounded-full" style="width:{pct:.1f}%"></div>'
            f"</div>"
            f'<span class="w-12 font-label text-xs {text_color}">{pct:.1f}%</span>'
            f"</div>"
        )
    return '<div class="space-y-3">' + "".join(rows) + "</div>"


def generate_16class_bar_chart(
    probabilities: list[float], labels: list[str], predicted_index: int
) -> str:
    fig, ax = plt.subplots(figsize=(5, 2.5))

    colors = [
        "#24389c" if i == predicted_index else "#e2e8f0" for i in range(len(labels))
    ]
    ax.bar(range(len(labels)), probabilities, color=colors, width=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Probability", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig_to_base64(fig)


def generate_node_importance_html(
    node_features_post_conv: torch.Tensor | None,
    primary_color: str = "#24389c",
) -> str:
    if node_features_post_conv is None:
        return (
            '<div class="flex items-center justify-center h-full text-slate-400 font-label text-xs">'
            "Node importance not available</div>"
        )

    norms = torch.linalg.norm(node_features_post_conv.detach().cpu().float(), dim=1)
    vmin, vmax = norms.min(), norms.max()
    if vmax - vmin > 0:
        norms = (norms - vmin) / (vmax - vmin)
    else:
        norms = norms * 0.0

    values = norms.tolist()
    cells = []
    for v in values:
        opacity = max(0.05, min(1.0, v))
        cells.append(
            f'<div class="aspect-square rounded-sm" '
            f'style="background:{primary_color};opacity:{opacity:.2f}"></div>'
        )
    return (
        '<div class="grid grid-cols-7 grid-rows-7 gap-1 w-full h-full">'
        + "".join(cells)
        + "</div>"
    )


def generate_boc_density_html(
    boc_features: torch.Tensor | None,
    primary_color: str = "#795300",
) -> str:
    """Render BoC character density as a 7x7 heatmap grid.

    Args:
        boc_features: [49, 70] BoC tensor or None if OCR unavailable.
        primary_color: CSS color for the cells.
    """
    if boc_features is None:
        return (
            '<div class="flex items-center justify-center h-full text-slate-400 font-label text-xs">'
            "OCR not available — install Tesseract</div>"
        )

    density = boc_features.detach().cpu().float().sum(dim=1)  # [49]
    vmin, vmax = density.min(), density.max()
    if vmax - vmin > 0:
        density = (density - vmin) / (vmax - vmin)
    else:
        density = density * 0.0

    cells = []
    for v in density.tolist():
        opacity = max(0.05, min(1.0, v))
        cells.append(
            f'<div class="aspect-square rounded-sm" '
            f'style="background:{primary_color};opacity:{opacity:.2f}"></div>'
        )
    return (
        '<div class="grid grid-cols-7 grid-rows-7 gap-1 w-full h-full">'
        + "".join(cells)
        + "</div>"
    )


def generate_document_graph_plotly(
    original_image: Image.Image,
    boxes: list,
    edge_index: torch.Tensor,
    ocr_texts: list,
    layout_class_colors: dict,
    layout_class_names: dict,
) -> str:
    """Interactive Plotly document graph — matches the notebook Viz 3 exactly.

    Args:
        boxes: [(x, y, w, h, class_id), ...] region nodes (no global node)
        edge_index: [2, E] undirected edges (includes global node at index len(boxes))
        ocr_texts: [str, ...] OCR text per region node (global text is last)
        layout_class_colors: {class_id: hex_color}
        layout_class_names: {class_id: str}

    Returns:
        HTML string (Plotly div, CDN-linked) for embedding in a template.
    """
    import plotly.graph_objects as go

    imgRgb = original_image.convert("RGB")
    imgW, imgH = imgRgb.size

    # Encode image as base64 for Plotly background
    buf = io.BytesIO()
    imgRgb.save(buf, format="PNG")
    imgB64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    nRegions = len(boxes)
    nNodes = nRegions + 1  # +1 for global node
    globalIdx = nRegions
    boxesWithGlobal = list(boxes) + [(0, 0, imgW, imgH, -1)]

    # Node positions and areas
    cxPx, cyPx, areas = [], [], []
    for x, y, w, h, _ in boxesWithGlobal:
        cxPx.append(x + w / 2)
        cyPx.append(y + h / 2)
        areas.append((w * h) / (imgW * imgH))

    # Node sizes scaled by area
    areaArr = np.array(areas[:nRegions]) if nRegions > 0 else np.array([1.0])
    if areaArr.max() > areaArr.min():
        scaled = 10 + 25 * (areaArr - areaArr.min()) / (areaArr.max() - areaArr.min())
    else:
        scaled = np.full(len(areaArr), 16.0)
    nodeSizes = list(scaled) + [30]

    # Hover cards
    allOcrTexts = list(ocr_texts) + [ocr_texts[-1] if ocr_texts else ""]
    hoverTexts = []
    for i, (x, y, w, h, clsId) in enumerate(boxesWithGlobal):
        isGlobal = i >= nRegions
        clsName = "GLOBAL (full image)" if isGlobal else layout_class_names.get(clsId, "?")
        rawText = allOcrTexts[i] if i < len(allOcrTexts) else ""
        textSnip = rawText[:150].replace("\n", "<br>").strip() or "<i>(no text)</i>"
        hoverTexts.append(
            f"<b>Node {i}</b>"
            f"<br><b>Class:</b> {clsName}"
            f"<br><b>Position:</b> ({x:.0f}, {y:.0f}) — {w:.0f}×{h:.0f}px"
            f"<br><b>Area:</b> {areas[i]:.1%} of page"
            f"<br><br><b>OCR:</b><br>{textSnip}"
        )

    # Figure sizing — preserve document aspect ratio
    PLOT_HEIGHT = 650
    MARGIN_T, MARGIN_B, MARGIN_L, MARGIN_R = 40, 60, 20, 20
    plotAreaH = PLOT_HEIGHT - MARGIN_T - MARGIN_B
    plotAreaW = int(plotAreaH * (imgW / imgH))
    PLOT_WIDTH = plotAreaW + MARGIN_L + MARGIN_R

    fig = go.Figure()

    # 1) Document image as background
    fig.add_layout_image(
        source=f"data:image/png;base64,{imgB64}",
        xref="x", yref="y",
        x=0, y=0,
        sizex=imgW, sizey=imgH,
        sizing="stretch", layer="below", opacity=0.4,
    )

    # 2) Bounding box overlays
    for x, y, w, h, clsId in boxes:
        color = layout_class_colors.get(clsId, "#888")
        fig.add_shape(
            type="rect", x0=x, y0=y, x1=x + w, y1=y + h,
            line=dict(color=color, width=2, dash="dot"),
            fillcolor=color, opacity=0.12, layer="below",
        )

    # 3) Edges — region-region vs global-region
    edgesCpu = edge_index.detach().cpu()
    edgePairs = list(zip(edgesCpu[0].tolist(), edgesCpu[1].tolist()))
    seenEdges: set[tuple] = set()
    regionEdgeX, regionEdgeY = [], []
    globalEdgeX, globalEdgeY = [], []
    for s, d in edgePairs:
        key = (min(s, d), max(s, d))
        if key in seenEdges:
            continue
        seenEdges.add(key)
        ex = [cxPx[s], cxPx[d], None]
        ey = [cyPx[s], cyPx[d], None]
        if s >= nRegions or d >= nRegions:
            globalEdgeX += ex
            globalEdgeY += ey
        else:
            regionEdgeX += ex
            regionEdgeY += ey

    fig.add_trace(go.Scatter(
        x=regionEdgeX, y=regionEdgeY, mode="lines",
        line=dict(width=2.0, color="rgba(60,60,60,0.65)"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=globalEdgeX, y=globalEdgeY, mode="lines",
        line=dict(width=1.8, color="rgba(218,165,32,0.75)", dash="dot"),
        hoverinfo="skip", showlegend=False,
    ))

    # 4) Region nodes — one trace per layout class
    classesPresentSorted = sorted(set(b[4] for b in boxes))
    for clsId in classesPresentSorted:
        clsName = layout_class_names.get(clsId, f"cls_{clsId}")
        idxs = [i for i in range(nRegions) if boxesWithGlobal[i][4] == clsId]
        fig.add_trace(go.Scatter(
            x=[cxPx[i] for i in idxs],
            y=[cyPx[i] for i in idxs],
            mode="markers+text",
            marker=dict(
                size=[nodeSizes[i] for i in idxs],
                color=layout_class_colors.get(clsId, "#888"),
                opacity=0.9,
                line=dict(width=1.5, color="white"),
            ),
            text=[str(i) for i in idxs],
            textfont=dict(size=8, color="white", family="Arial Black"),
            textposition="middle center",
            hovertext=[hoverTexts[i] for i in idxs],
            hoverinfo="text",
            name=clsName,
        ))

    # 5) Global node
    fig.add_trace(go.Scatter(
        x=[cxPx[globalIdx]], y=[cyPx[globalIdx]],
        mode="markers+text",
        marker=dict(size=28, color="#FFD700", symbol="star",
                    opacity=0.95, line=dict(width=2, color="#333")),
        text=["G"],
        textfont=dict(size=10, color="#333", family="Arial Black"),
        textposition="top center",
        hovertext=[hoverTexts[globalIdx]],
        hoverinfo="text",
        name="Global node",
    ))

    fig.update_layout(
        xaxis=dict(
            range=[-imgW * 0.02, imgW * 1.02],
            showgrid=False, zeroline=False, showticklabels=False,
            constrain="domain",
        ),
        yaxis=dict(
            range=[imgH * 1.02, -imgH * 0.02],  # reversed: y=0 at top
            showgrid=False, zeroline=False, showticklabels=False,
            scaleanchor="x", scaleratio=1,
            constrain="domain",
        ),
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        legend=dict(
            title=dict(text="Layout Classes", font=dict(size=10), side="left"),
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ddd", borderwidth=1,
            orientation="h",
            x=0, y=-0.06, xanchor="left", yanchor="top",
        ),
        hoverlabel=dict(
            bgcolor="white", bordercolor="#ccc",
            font=dict(size=12, family="Arial"),
        ),
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=MARGIN_T, b=MARGIN_B),
    )

    return fig.to_html(include_plotlyjs=False, full_html=False, config={"responsive": True})


def generate_original_image_base64(
    image: Image.Image, max_size: int = 400
) -> str:
    img = image.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
