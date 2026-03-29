"""GNN model architectures for document classification."""

import os

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "16"))


class GraphSAGEClassifier(nn.Module):
    """Two-layer GraphSAGE classifier with global mean pooling.

    Architecture:
        SAGEConv(in, hidden) → ReLU → Dropout
        → SAGEConv(hidden, embed) → ReLU
        → global_mean_pool
        → Linear(embed, num_classes)

    Args:
        in_channels: Input feature dimension per node (e.g., 2048)
        hidden_channels: First SAGEConv output dimension
        embed_channels: Second SAGEConv output dimension
        num_classes: Number of classification categories
        dropout: Dropout probability after first SAGEConv
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        embed_channels: int = 128,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embed_channels)
        self.classifier = nn.Linear(embed_channels, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        """Forward pass.

        Args:
            x: Node features [num_nodes_total, in_channels]
            edge_index: Edge indices [2, num_edges_total]
            batch: Batch assignment vector [num_nodes_total]

        Returns:
            Logits [batch_size, num_classes]
        """
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


class HybridGraphSAGE(nn.Module):
    """Two-layer GraphSAGE with CNN+GNN hybrid fusion classifier.

    Combines local spatial structure via GNN on 7x7 feature map graph
    with global context from ResNet-50 avgpool via concatenation fusion.

    Architecture:
        GNN path: SAGEConv(node_dim, hidden) → ReLU → Dropout
                  → SAGEConv(hidden, embed) → ReLU
                  → global_mean_pool → gnn_embed [batch_size, embed_channels]
        CNN path: global_feat [batch_size, global_channels]
        Fusion: concat [gnn_embed, global_feat] → Linear(fusion_dim, num_classes)

    Args:
        node_dim: Input node feature dimension (2048 features + 2 PE coords = 2050)
        hidden_channels: First SAGEConv output dimension
        embed_channels: Second SAGEConv output dimension (GNN embedding size)
        global_channels: CNN global feature dimension (ResNet-50 avgpool = 2048)
        num_classes: Number of classification categories
        dropout: Dropout probability after first SAGEConv
    """

    def __init__(
        self,
        node_dim: int = 2050,
        hidden_channels: int = 256,
        embed_channels: int = 128,
        global_channels: int = 2048,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
    ):
        super().__init__()
        # GNN layers
        self.conv1 = SAGEConv(node_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embed_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Fusion classifier
        fusion_dim = embed_channels + global_channels  # 128 + 2048 = 2176
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, x, edge_index, batch, global_feat):
        """Forward pass with hybrid CNN+GNN fusion.

        Args:
            x: Node features [num_nodes_total, node_dim]
            edge_index: Edge indices [2, num_edges_total]
            batch: Batch assignment vector [num_nodes_total]
            global_feat: Global CNN features [batch_size, global_channels]

        Returns:
            Logits [batch_size, num_classes]
        """
        # GNN path
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        # Graph-level pooling
        gnn_embed = global_mean_pool(x, batch)  # [batch_size, embed_channels]

        # Handle potential extra dimension from PyG batching
        if global_feat.dim() == 3:
            global_feat = global_feat.squeeze(1)

        # Fusion: concatenate GNN embedding with CNN global features
        fused = torch.cat([gnn_embed, global_feat], dim=1)  # [batch_size, fusion_dim]

        # Classification
        logits = self.classifier(fused)
        return logits
