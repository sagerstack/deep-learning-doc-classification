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

    values = text_density.detach().cpu().flatten().tolist()
    cells = []
    for v in values:
        opacity = max(0.0, min(1.0, v))
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
