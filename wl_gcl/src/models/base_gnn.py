# src/models/base_gnn.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, GINConv


class GCNEncoder(nn.Module):
    """
    Simple 2-layer GCN encoder for node embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Compute node embeddings.

        Args:
            x: (N, in_dim)
            edge_index: (2, E)

        Returns:
            z: (N, out_dim)
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv2(x, edge_index)

        # Normalize embeddings so they lie on the unit hypersphere.
        # This is required for Cosine Similarity stability in the Loss function.
        z = F.normalize(x, p=2, dim=-1)

        return z
    

class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.6, **kwargs):
        super(GATEncoder, self).__init__()
        self.dropout = dropout
        
        # Layer 1
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        
        # Layer 2
        # Input to layer 2 is hidden_dim * heads
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Normalize for Contrastive Loss
        return F.normalize(x, p=2, dim=-1)


class GINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, **kwargs):
        super(GINEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            dim_in = input_dim if i == 0 else hidden_dim
            dim_out = output_dim if i == num_layers - 1 else hidden_dim
            
            # GIN requires an MLP to be passed to it
            mlp = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.BatchNorm1d(dim_out),
                nn.ReLU(),
                nn.Linear(dim_out, dim_out)
            )
            
            self.layers.append(GINConv(mlp))
            
            # Batch Norm between GIN layers (optional but good for deep GINs)
            if i < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(dim_out))
    
    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            
            if i < len(self.layers) - 1:
                x = self.batch_norms[i](x)
                x = F.relu(x)
        
        return F.normalize(x, p=2, dim=-1)