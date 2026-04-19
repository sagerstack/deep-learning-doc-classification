"""Model definitions used by the FastAPI demo app."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool, global_add_pool
from torch_geometric.utils import scatter, softmax as pyg_softmax

NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "16"))


class DocumentGAT(nn.Module):
    """Multimodal GAT trained on YOLO regions + ResNet + OCR + MiniLM features.

    Node feature dim: 2448 = 2048 (ResNet) + 384 (MiniLM) + 4 (bbox) + 11 (layout class onehot) + 1 (global flag)
    """

    def __init__(
        self,
        in_dim: int = 2447,
        hidden_dim: int = 256,
        out_dim: int = NUM_CLASSES,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        pooling: str = "mean",
    ):
        super().__init__()
        self.dropout = dropout
        self.pooling_type = pooling
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                if num_layers == 1:
                    self.convs.append(GATConv(in_dim, hidden_dim, heads=1, concat=False, dropout=dropout))
                    self.norms.append(nn.LayerNorm(hidden_dim))
                else:
                    self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
                    self.norms.append(nn.LayerNorm(hidden_dim * heads))
            elif i < num_layers - 1:
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
                self.norms.append(nn.LayerNorm(hidden_dim * heads))
            else:
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout))
                self.norms.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)

        if self.pooling_type == "mean":
            x = global_mean_pool(x, batch)
        else:
            x = global_add_pool(x, batch)

        return self.classifier(x)


class HybridGraphSAGE(nn.Module):
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
        self.conv1 = SAGEConv(node_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embed_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.gnn_bn = nn.BatchNorm1d(embed_channels)
        self.cnn_bn = nn.BatchNorm1d(global_channels)
        self.classifier = nn.Linear(embed_channels + global_channels, num_classes)

    def forward(self, x, edge_index, batch, global_feat):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)

        gnn_embed = global_mean_pool(x, batch)
        if global_feat.dim() == 3:
            global_feat = global_feat.squeeze(1)

        gnn_embed = self.gnn_bn(gnn_embed)
        global_feat = self.cnn_bn(global_feat)
        fused = torch.cat([gnn_embed, global_feat], dim=1)
        return self.classifier(fused)


class FusionGraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int = 2048,
        hidden: int = 256,
        gnn_embed: int = 128,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
        global_dim: int = 2048,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, gnn_embed)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        fusion_dim = global_dim + gnn_embed
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, edge_index, batch, global_feat):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        gnn_out = global_mean_pool(x, batch)

        if global_feat.dim() == 3:
            global_feat = global_feat.squeeze(1)

        fused = torch.cat([global_feat, gnn_out], dim=1)
        return self.classifier(fused)


class FusionGAT(nn.Module):
    def __init__(
        self,
        in_channels: int = 2048,
        hidden: int = 256,
        embed: int = 128,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
        heads: int = 4,
    ):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden, embed, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(2048 + embed, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, edge_index, batch, global_feat):
        h = self.conv1(x, edge_index)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.relu(h)
        gnn_out = global_mean_pool(h, batch)

        if global_feat.dim() == 3:
            global_feat = global_feat.squeeze(1)

        fused = torch.cat([global_feat, gnn_out], dim=1)
        return self.classifier(fused)


class GatedBoCGraphSAGE(nn.Module):
    def __init__(
        self,
        cnn_dim: int = 2050,
        boc_dim: int = 70,
        proj_dim: int = 16,
        hidden_channels: int = 256,
        embed_channels: int = 128,
        global_channels: int = 2048,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.text_proj = nn.Linear(boc_dim, proj_dim)
        self.text_gate = nn.Linear(boc_dim, proj_dim)
        self.conv1 = SAGEConv(cnn_dim + proj_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embed_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.gnn_bn = nn.BatchNorm1d(embed_channels)
        self.cnn_bn = nn.BatchNorm1d(global_channels)
        self.classifier = nn.Linear(embed_channels + global_channels, num_classes)

    def forward(self, x, edge_index, batch, global_feat, x_boc):
        gate = torch.sigmoid(self.text_gate(x_boc))
        projected = self.text_proj(x_boc)
        gated_text = gate * projected
        x = torch.cat([x, gated_text], dim=1)

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)

        gnn_embed = global_mean_pool(x, batch)
        if global_feat.dim() == 3:
            global_feat = global_feat.squeeze(1)

        gnn_embed = self.gnn_bn(gnn_embed)
        global_feat = self.cnn_bn(global_feat)
        fused = torch.cat([gnn_embed, global_feat], dim=1)
        return self.classifier(fused)


class AttentionPoolFusionSAGE(nn.Module):
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
        self.conv1 = SAGEConv(node_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embed_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.attn_query = nn.Parameter(torch.randn(1, embed_channels))
        self.attn_key = nn.Linear(embed_channels, embed_channels, bias=False)
        self.gnn_bn = nn.BatchNorm1d(embed_channels)
        self.cnn_bn = nn.BatchNorm1d(global_channels)
        self.classifier = nn.Linear(embed_channels + global_channels, num_classes)

    def forward(self, x, edge_index, batch, global_feat):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)

        keys = self.attn_key(x)
        attn_scores = (keys * self.attn_query).sum(dim=-1)
        attn_weights = pyg_softmax(attn_scores, batch)
        gnn_embed = scatter(x * attn_weights.unsqueeze(-1), batch, dim=0, reduce="sum")

        if global_feat.dim() == 3:
            global_feat = global_feat.squeeze(1)

        gnn_embed = self.gnn_bn(gnn_embed)
        global_feat = self.cnn_bn(global_feat)
        fused = torch.cat([gnn_embed, global_feat], dim=1)
        return self.classifier(fused)


class InductiveGCN(nn.Module):
    """3-layer GCN trained inductively on a 320k-document cosine-kNN graph.

    One document = one node (L2-normalized 2048-d ResNet-50 avgpool feature).
    Edges are built by k-NN (k=5) over cosine similarity. Per-node classification
    — no pooling. Single-doc inference requires connecting the query into a
    reference feature bank and reading the query node's logits.

    Matches training: GCNClassifier(2048, [512, 128], 16, dropout=0.3).
    """

    def __init__(
        self,
        in_dim: int = 2048,
        hidden: int = 512,
        embed: int = 128,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(in_dim, hidden),
            GCNConv(hidden, embed),
        ])
        self.out = GCNConv(embed, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, edge_index)), p=self.dropout, training=self.training)
        return self.out(x, edge_index)
