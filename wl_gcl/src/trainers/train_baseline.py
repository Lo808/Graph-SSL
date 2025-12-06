# wl_gcl/src/trainers/train_baseline.py
import argparse
import torch
from torch.optim import Adam

from wl_gcl.src.data_loader.dataset import load_dataset
from wl_gcl.src.models.base_gnn import BaseGCN
from wl_gcl.src.contrastive.losses import nt_xent_loss

def train_baseline(args):
    """
    Hala's original simple baseline logic (Simple GCN + SimCLR).
    """
    print(f"\n=== Running Baseline (Simple GCN) on {args.dataset.upper()} ===")
    
    # 1. Load Data
    node_dataset = load_dataset(args.dataset)
    data = node_dataset.data
    
    device = torch.device(args.device)
    data = data.to(device)

    # 2. Init Model
    model = BaseGCN(
        in_dim=node_dataset.num_features,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # 3. Training Loop
    print(f"[Baseline] Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Simple augmentation: just passing same graph twice (Identity)
        z1 = model(data.x, data.edge_index)
        z2 = model(data.x, data.edge_index)

        loss = nt_xent_loss(z1, z2, temperature=args.temperature)

        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            print(f"[Epoch {epoch:03d}] Loss = {loss.item():.4f}")

    print("Baseline Training complete.")