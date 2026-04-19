"""Graph helpers for the web app inference path."""

import torch


def build_grid_edge_index(grid_h: int, grid_w: int, k: int = 8) -> torch.Tensor:
    """Build grid edges for a fixed HxW feature map."""
    if k == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif k == 8:
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        raise ValueError(f"k must be 4 or 8, got {k}")

    sources: list[int] = []
    targets: list[int] = []

    for row in range(grid_h):
        for col in range(grid_w):
            node_idx = row * grid_w + col
            for dr, dc in offsets:
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    sources.append(node_idx)
                    targets.append(nr * grid_w + nc)

    return torch.tensor([sources, targets], dtype=torch.long)


def add_positional_encoding_2d(
    x: torch.Tensor,
    grid_h: int = 7,
    grid_w: int = 7,
) -> torch.Tensor:
    """Append normalized (row, col) coordinates to node features."""
    rows = torch.arange(grid_h, dtype=torch.float32) / (grid_h - 1)
    cols = torch.arange(grid_w, dtype=torch.float32) / (grid_w - 1)

    row_coords = rows.repeat_interleave(grid_w)
    col_coords = cols.repeat(grid_h)
    pos_enc = torch.stack([row_coords, col_coords], dim=1).to(x.device)

    return torch.cat([x, pos_enc], dim=1)


def build_doc_knn_edges(
    features: torch.Tensor,
    k: int = 5,
    batch_size: int = 2000,
) -> torch.Tensor:
    """Build directed kNN edges over cosine similarity between document features.

    Used by the Inductive GCN: one document = one node, edges connect each node
    to its k most cosine-similar neighbors (self excluded). Matches the training
    setup in final-notebooks/400K_GCN.ipynb (build_knn_graph_safe, k=5).

    Args:
        features: [N, D] L2-normalized document features.
        k: number of neighbors per node.
        batch_size: source-batch size for memory-safe computation.

    Returns:
        edge_index [2, N*k] long tensor on the same device as ``features``.
    """
    num_nodes = features.shape[0]
    device = features.device
    src_chunks: list[torch.Tensor] = []
    dst_chunks: list[torch.Tensor] = []

    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        sims = features[start:end] @ features.T  # [B, N]
        # Mask self-similarity so each node picks its nearest OTHER nodes.
        diag_rows = torch.arange(end - start, device=device)
        diag_cols = torch.arange(start, end, device=device)
        sims[diag_rows, diag_cols] = float("-inf")

        _, topk = sims.topk(k, dim=1)  # [B, k]
        src = torch.arange(start, end, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = topk.reshape(-1)
        src_chunks.append(src)
        dst_chunks.append(dst)

    return torch.stack([torch.cat(src_chunks), torch.cat(dst_chunks)], dim=0).long()
