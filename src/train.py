"""Training utilities for Fusion GraphSAGE (Exp16).

Provides:
  - FeatureKNNDataset: lazy-loading dataset from cached .pt files
  - train_fusion_sage: full training loop with cosine LR, DropEdge, early stopping
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import dropout_edge
from sklearn.metrics import accuracy_score, f1_score


class FeatureKNNDataset:
    """Lazy-loading dataset that builds feature-space k-NN graphs from cached .pt files.

    Each .pt file contains:
        features:    [2048, 7, 7]  spatial CNN features
        global_feat: [2048]        avgpool CNN features
        label:       int

    Graphs are constructed on-the-fly: 49 nodes with cosine-similarity k-NN edges.
    """

    def __init__(self, split_dir, k=8):
        self.files = sorted(Path(split_dir).glob("*.pt"))
        self.k = k
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        features = data["features"]      # [2048, 7, 7]
        global_feat = data["global_feat"]  # [2048]
        label = data["label"]

        c, h, w = features.shape
        x = features.reshape(c, h * w).T  # [49, 2048]
        y = torch.tensor([label], dtype=torch.long)

        # Feature-space k-NN edges (cosine similarity)
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        sim = x_norm @ x_norm.T
        sim.fill_diagonal_(-float('inf'))
        _, topk_indices = sim.topk(self.k, dim=1)
        sources = torch.arange(x.shape[0]).unsqueeze(1).expand_as(topk_indices).flatten()
        targets = topk_indices.flatten()
        edge_index = torch.stack([sources, targets], dim=0)

        return Data(x=x, edge_index=edge_index, y=y,
                    global_feat=global_feat.unsqueeze(0))


def train_fusion_sage(
    model,
    train_dir,
    val_dir,
    checkpoint_path,
    device,
    max_epochs=40,
    patience=10,
    lr=1e-3,
    weight_decay=5e-4,
    batch_size=64,
    label_smoothing=0.1,
    dropedge_p=0.2,
    k=8,
    num_workers=8,
):
    """Train Fusion GraphSAGE with cosine LR, DropEdge, and early stopping.

    Args:
        model: FusionGraphSAGE model instance
        train_dir: Path to training cached features directory
        val_dir: Path to validation cached features directory
        checkpoint_path: Where to save best model weights
        device: torch.device
        max_epochs: Maximum training epochs
        patience: Early stopping patience (on val F1)
        lr: Initial learning rate
        weight_decay: Adam weight decay
        batch_size: Training batch size
        label_smoothing: CrossEntropyLoss label smoothing
        dropedge_p: DropEdge probability during training
        k: k-NN neighbors for graph construction
        num_workers: DataLoader workers

    Returns:
        dict with best_acc, best_f1, best_epoch, training_time
    """
    model = model.to(device)
    train_ds = FeatureKNNDataset(train_dir, k=k)
    val_ds = FeatureKNNDataset(val_dir, k=k)
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_loader = PyGDataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = PyGDataLoader(val_ds, shuffle=False, **loader_kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_f1 = 0.0
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
    }

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training Fusion GraphSAGE: {max_epochs} max epochs, patience={patience}")
    print(f"Parameters: {params:,}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Batch: {batch_size}")

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            global_feat = batch.global_feat.to(device)
            optimizer.zero_grad()

            edge_index, _ = dropout_edge(batch.edge_index, p=dropedge_p, training=True)
            out = model(batch.x, edge_index, batch.batch, global_feat)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            correct += (out.argmax(1) == batch.y).sum().item()
            total += batch.num_graphs

        scheduler.step()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss_sum = 0.0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                global_feat = batch.global_feat.to(device)
                out = model(batch.x, batch.edge_index, batch.batch, global_feat)
                val_loss_sum += criterion(out, batch.y).item() * batch.num_graphs
                val_total += batch.num_graphs
                all_preds.extend(out.argmax(1).cpu().tolist())
                all_labels.extend(batch.y.cpu().tolist())

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        train_acc = correct / total
        train_loss = total_loss / total
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        if (epoch + 1) % 2 == 0 or improved:
            print(f"  Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Train: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.1e} | {epoch_time:.0f}s {'*' if improved else ''}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time
    print(f"\nBest: epoch {best_epoch}, val_acc={best_acc:.4f}, val_f1={best_f1:.4f}")
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Training time: {elapsed/60:.1f} min")

    return {
        "best_acc": best_acc,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "training_time_sec": elapsed,
        "history": history,
    }
