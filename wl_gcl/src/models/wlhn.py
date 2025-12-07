import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from math import sqrt

from .hypernn import MobiusLinear 

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

class WLHNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, tau=1.0, dropout=0.5):
        """
        Adapted WLHN Model for Node-Level Contrastive Learning.
        """
        super(WLHNEncoder, self).__init__()
        self.n_layers = n_layers
        self.scaling = torch.tanh(torch.tensor(tau / 2))

        # 1. Initial Projection to Hidden Dim
        self.fc0 = nn.Linear(input_dim, hidden_dim)

        # 2. Hyperbolic bias vector
        self.p = (-1./sqrt(hidden_dim)) * torch.ones(hidden_dim, requires_grad=False)

        # 3. GIN Layers (Base MPNN)
        lst = list()
        # First Layer
        lst.append(GINConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU())))

        # Subsequent Layers
        for i in range(n_layers-1):
            lst.append(GINConv(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())))

        self.conv = nn.ModuleList(lst)

        # 4. Final Projection Head (Optional, mapping hidden to output_dim)
        # We use a simple linear layer here to match the output dimension requested
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    # --- Hyperbolic Geometric Utilities (From original paper) ---
    def isometric_transform(self, x, a):
        r2 = torch.sum(a ** 2, dim=-1, keepdim=True) - 1.
        u = x - a
        return r2 / torch.sum(u ** 2, dim=-1, keepdim=True).clamp_min(MIN_NORM) * u + a

    def reflection_center(self, mu):
        return mu / torch.sum(mu ** 2, dim=-1, keepdim=True).clamp_min(MIN_NORM)

    def reflect_at_zero(self, x, mu):
        a = self.reflection_center(mu)
        return self.isometric_transform(x, a)

    def reflect_through_zero(self, p, q, x):
        p_ = p / torch.norm(p, dim=-1, keepdim=True).clamp_min(MIN_NORM)
        q_ = q / torch.norm(q, dim=-1, keepdim=True).clamp_min(MIN_NORM)
        r = q_ - p_
        m = torch.sum(r * x, dim=-1, keepdim=True) / torch.sum(r * r, dim=-1, keepdim=True)
        return x - 2 * r * m

    def project(self, x):
        norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        eps = BALL_EPS[x.dtype]
        maxnorm = (1 - eps)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def logmap0(self, y):
        """Map from Poincaré Ball -> Euclidean Tangent Space"""
        y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        return y / y_norm / 1. * torch.atanh(y_norm.clamp(-1 + 1e-15, 1 - 1e-15))

    # --- Forward Pass ---
    def forward(self, x, edge_index):
        # Initial Embedding
        x = self.relu(self.fc0(x))
        
        # Lists to store states for WL-Hyperbolic aggregation
        xs = [x]
        z = [torch.zeros(1, x.size(1), device=x.device, requires_grad=False)]
        inv = [torch.zeros(x.size(0), dtype=torch.long, device=x.device, requires_grad=False)]
        
        # Initial Unique Hashing
        with torch.no_grad():
            unique_all, inv_all = torch.unique(x, sorted=False, return_inverse=True, dim=0)
        
        unique_all_norm = unique_all/torch.norm(unique_all, dim=1).unsqueeze(1)
        z.append(self.scaling * unique_all_norm)
        inv.append(inv_all)

        # Iterative Updates (The WLHN Logic)
        for i in range(self.n_layers):
            # 1. MPNN Step (GIN)
            x = self.conv[i](x, edge_index)
            xs.append(x)
            
            # 2. Hyperbolic Reflection Step (DiffHypCon)
            with torch.no_grad():
                # Hash nodes based on history (Concatenation of all previous states)
                unique_all, inv_all, count_all = torch.unique(
                    torch.cat(xs, dim=1), sorted=False, return_inverse=True, return_counts=True, dim=0
                )
            
            # Retrieve representations
            unique_all = unique_all[:, -x.size(1):] # Get current layer features
            unique_all_norm = unique_all / torch.norm(unique_all, dim=1).unsqueeze(1)
            z_children = self.scaling * unique_all_norm
            
            # Calculate parent/child relationships in the WL tree
            t = torch.zeros(unique_all.size(0), dtype=torch.long, device=x.device)
            t.scatter_add_(0, inv_all, inv[i+1])
            t = torch.div(t, count_all).long()
            z_current = torch.gather(z[i+1], 0, t.unsqueeze(1).repeat(1, z[i+1].size(1)))
            
            t = torch.zeros(unique_all.size(0), dtype=torch.long, device=x.device)
            t.scatter_add_(0, inv_all, inv[i])
            t = torch.div(t, count_all).long()
            z_parent = torch.gather(z[i], 0, t.unsqueeze(1).repeat(1, z[i].size(1)))
            
            # Apply Reflections
            z_parent = self.reflect_at_zero(z_parent, z_current)
            z_children = self.reflect_through_zero(z_parent, self.p.to(x.device), z_children)
            z_all = self.reflect_at_zero(z_children, z_current)
            
            inv.append(inv_all)
            z.append(z_all)
            
        # 3. Final Mapping
        # Logmap maps Hyperbolic -> Euclidean (Tangent space)
        x_hyper = self.logmap0(z[-1])
        
        # Reconstruct node ordering
        x_out = torch.index_select(x_hyper, 0, inv[-1])
        
        # Final Linear Projection to requested output dimension
        out = self.fc_out(x_out)
        
        # Normalize for Contrastive Loss
        return F.normalize(out, p=2, dim=-1)