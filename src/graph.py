"""Graph construction from spatial feature maps for GNN classification."""

from typing import Dict, List

import torch
from torch_geometric.data import Data


def build_grid_edge_index(grid_h: int, grid_w: int, k: int = 8) -> torch.Tensor:
    """Build k-NN edge index for a fixed spatial grid.

    Connects each node to its k nearest neighbors on the grid.
    k=4: cardinal neighbors (up, down, left, right)
    k=8: Moore neighborhood (cardinal + diagonal)

    Args:
        grid_h: Grid height (e.g., 7 for ResNet-50 layer4)
        grid_w: Grid width (e.g., 7 for ResNet-50 layer4)
        k: Number of neighbors (4 or 8)

    Returns:
        edge_index: Tensor [2, num_edges] — bidirectional edges
    """
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

    sources = []
    targets = []

    for row in range(grid_h):
        for col in range(grid_w):
            node_idx = row * grid_w + col
            for dr, dc in offsets:
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    neighbor_idx = nr * grid_w + nc
                    sources.append(node_idx)
                    targets.append(neighbor_idx)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return edge_index


def feature_map_to_graph(
    features: torch.Tensor,
    label: int,
    edge_index: torch.Tensor,
) -> Data:
    """Convert a spatial feature map to a PyG Data object.

    Reshapes [C, H, W] features into [H*W, C] node features,
    then attaches the shared edge_index and class label.

    Args:
        features: Tensor [C, H, W] — e.g., [2048, 7, 7] from ResNet-50 layer4
        label: Integer class label
        edge_index: Tensor [2, num_edges] from build_grid_edge_index

    Returns:
        PyG Data(x=[num_nodes, C], edge_index=[2, num_edges], y=[1])
    """
    c, h, w = features.shape
    x = features.reshape(c, h * w).T  # [H*W, C]
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def build_graph_dataset(
    cached_features: List[Dict],
    edge_index: torch.Tensor,
) -> List[Data]:
    """Convert all cached features into a list of PyG Data objects.

    Args:
        cached_features: List of dicts with "features" [C, H, W] and "label" int
        edge_index: Shared edge_index from build_grid_edge_index

    Returns:
        List of PyG Data objects
    """
    graphs = []
    for item in cached_features:
        graph = feature_map_to_graph(item["features"], item["label"], edge_index)
        graphs.append(graph)
    return graphs


def add_positional_encoding_2d(x: torch.Tensor, grid_h: int = 7, grid_w: int = 7) -> torch.Tensor:
    """Append normalized 2D positional encoding to node features.

    Adds (row, col) coordinates normalized to [0, 1] range as additional features.

    Args:
        x: Node features [num_nodes, feature_dim]
        grid_h: Grid height (default 7 for ResNet-50 layer4)
        grid_w: Grid width (default 7 for ResNet-50 layer4)

    Returns:
        Node features with PE appended [num_nodes, feature_dim + 2]
    """
    # Generate normalized coordinates in [0, 1]
    rows = torch.arange(grid_h, dtype=torch.float32) / (grid_h - 1)
    cols = torch.arange(grid_w, dtype=torch.float32) / (grid_w - 1)

    # Create grid coordinates
    row_coords = rows.repeat_interleave(grid_w)  # [0, 0, ..., 1, 1]
    col_coords = cols.repeat(grid_h)             # [0, 0.167, ..., 1, 0, 0.167, ..., 1]

    # Stack into [num_nodes, 2]
    pos_enc = torch.stack([row_coords, col_coords], dim=1)

    # Concatenate to node features
    return torch.cat([x, pos_enc.to(x.device)], dim=1)


def feature_map_to_graph_hybrid(
    features: torch.Tensor,
    global_feat: torch.Tensor,
    label: int,
    edge_index: torch.Tensor,
) -> Data:
    """Convert spatial feature map to PyG Data with positional encoding and global features.

    Args:
        features: Spatial features [C, H, W] — e.g., [2048, 7, 7] from ResNet-50 layer4
        global_feat: Global features [C] — e.g., [2048] from ResNet-50 avgpool
        label: Integer class label
        edge_index: Tensor [2, num_edges] from build_grid_edge_index

    Returns:
        PyG Data(x=[num_nodes, C+2], edge_index=[2, num_edges], y=[1], global_feat=[1, C])
    """
    c, h, w = features.shape

    # Reshape to node features [H*W, C]
    x = features.reshape(c, h * w).T

    # Add positional encoding -> [H*W, C+2]
    x = add_positional_encoding_2d(x, grid_h=h, grid_w=w)

    # Create PyG Data object
    y = torch.tensor([label], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

    # Store global_feat as 2D [1, C] for correct PyG batching
    data.global_feat = global_feat.unsqueeze(0)

    return data


def build_graph_dataset_hybrid(
    cached_features: List[Dict],
    edge_index: torch.Tensor,
) -> List[Data]:
    """Convert cached features with global_feat into hybrid PyG Data objects.

    Args:
        cached_features: List of dicts with "features" [C, H, W], "global_feat" [C], and "label" int
        edge_index: Shared edge_index from build_grid_edge_index

    Returns:
        List of PyG Data objects with positional encoding and global features
    """
    graphs = []
    for item in cached_features:
        graph = feature_map_to_graph_hybrid(
            item["features"],
            item["global_feat"],
            item["label"],
            edge_index
        )
        graphs.append(graph)
    return graphs


def feature_map_to_graph_text_aware(
    features: torch.Tensor,
    global_feat: torch.Tensor,
    text_density: torch.Tensor,
    label: int,
    edge_index: torch.Tensor,
) -> Data:
    """Convert spatial feature map to PyG Data with positional encoding, global features, and text density.

    Extends the hybrid graph construction by appending a per-node text density
    scalar derived from doctr DBNet probability maps. Each node gains one extra
    feature channel representing how much text covers that spatial grid cell.

    Args:
        features: Spatial features [C, H, W] — e.g., [2048, 7, 7] from ResNet-50 layer4
        global_feat: Global features [C] — e.g., [2048] from ResNet-50 avgpool
        text_density: Text density heatmap [H, W] — e.g., [7, 7] from doctr
        label: Integer class label
        edge_index: Tensor [2, num_edges] from build_grid_edge_index

    Returns:
        PyG Data(x=[num_nodes, C+3], edge_index=[2, num_edges], y=[1], global_feat=[1, C])
        where node features = CNN features [C] + PE [2] + text density [1] = C+3
    """
    c, h, w = features.shape

    # Reshape CNN features to node features [H*W, C]
    x = features.reshape(c, h * w).T

    # Add 2D positional encoding -> [H*W, C+2]
    x = add_positional_encoding_2d(x, grid_h=h, grid_w=w)

    # Append text density as per-node feature -> [H*W, C+3]
    td_flat = text_density.reshape(h * w, 1).to(x.device)  # [H*W, 1]
    x = torch.cat([x, td_flat], dim=1)

    # Create PyG Data object
    y = torch.tensor([label], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

    # Store global_feat as 2D [1, C] for correct PyG batching
    data.global_feat = global_feat.unsqueeze(0)

    return data


def build_graph_dataset_text_aware(
    cached_features: List[Dict],
    edge_index: torch.Tensor,
) -> List[Data]:
    """Convert cached features with text_density into text-aware PyG Data objects.

    Args:
        cached_features: List of dicts with "features" [C, H, W], "global_feat" [C],
                         "text_density" [H, W], and "label" int.
                         Raises ValueError if any item is missing "text_density".
        edge_index: Shared edge_index from build_grid_edge_index

    Returns:
        List of PyG Data objects with positional encoding, global features, and text density
    """
    graphs = []
    for i, item in enumerate(cached_features):
        if "text_density" not in item:
            raise ValueError(
                f"Item at index {i} is missing required 'text_density' key. "
                f"Available keys: {list(item.keys())}. "
                "Run augment_cache_with_text_density first to add text density to cached features."
            )
        graph = feature_map_to_graph_text_aware(
            item["features"],
            item["global_feat"],
            item["text_density"],
            item["label"],
            edge_index,
        )
        graphs.append(graph)
    return graphs
