"""Graph neural network model definitions for drug embeddings."""

from __future__ import annotations


def build_gnn_model(input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
    try:
        import torch
        from torch import nn
        from torch_geometric.nn import GCNConv
    except Exception:  # pragma: no cover
        return None

    class DrugGNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
            self.activation = nn.ReLU()

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = self.activation(x)
            x = self.conv2(x, edge_index)
            return x

    return DrugGNN()


def has_gnn_support() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        return True
    except Exception:
        return False
