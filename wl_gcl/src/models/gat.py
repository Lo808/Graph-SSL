import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.6):
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