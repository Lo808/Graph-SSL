# wl_gcl/src/trainers/train_baseline.py
from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Dict

import torch
from torch.optim import Adam

from wl_gcl.src.data_loader.dataset import load_dataset
from wl_gcl.src.models.base_gnn import BaseGCN
from wl_gcl.src.contrastive.losses import nt_xent_loss
from wl_gcl.src.augmentations.graph_augmentor import GraphAugmentor
from wl_gcl.configs.baseline import cfg as default_cfg
from wl_gcl.configs.baseline import BaselineConfig
from wl_gcl.src.trainers.eval import evaluate_linear_probe


# Trainer
def train_baseline(cfg: BaselineConfig) -> Dict[str, float]:
    """
    Baseline SimCLR training with GCN encoder and graph augmentations.
    """
    device = torch.device(cfg.device)

    # Data
    dataset = load_dataset(cfg.dataset)
    data = dataset.data.to(device)

    # Model
    model = BaseGCN(
        in_dim=dataset.num_features,
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.out_dim,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr)

    # Augmentor
    augmentor = GraphAugmentor(
        edge_drop_prob=cfg.edge_drop_prob,
        feature_mask_prob=cfg.feature_mask_prob,
    )

    # Training
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Two stochastic views
        x1, edge_index1 = augmentor.augment(data.x, data.edge_index)
        x2, edge_index2 = augmentor.augment(data.x, data.edge_index)

        z1 = model(x1, edge_index1)
        z2 = model(x2, edge_index2)

        loss = nt_xent_loss(z1, z2, temperature=cfg.temperature)

        loss.backward()
        optimizer.step()

        if epoch % cfg.log_interval == 0 or epoch == cfg.epochs:
            print(
                f"[Baseline | {cfg.dataset:<12}] "
                f"Epoch {epoch:03d}/{cfg.epochs}  "
                f"Loss: {loss.item():.4f}"
            )

    test_acc = evaluate_linear_probe(
        model=model,
        data=data,
        num_classes=dataset.num_classes,
        device=device,
    )

    print(
        f"[Evaluation | {cfg.dataset:<12}] "
        f"Linear Probe Accuracy: {test_acc:.4f}"
    )

    return {
        "dataset": cfg.dataset,
        "final_loss": float(loss.item()),
        "test_accuracy": test_acc,
        "epochs": cfg.epochs,
    }


# CLI Runner
def main() -> None:
    parser = argparse.ArgumentParser(description="Run GCN SimCLR baseline.")

    parser.add_argument("--dataset")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--out_dim", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--edge_drop_prob", type=float)
    parser.add_argument("--feature_mask_prob", type=float)
    parser.add_argument("--device")
    parser.add_argument("--log_interval", type=int)

    args = parser.parse_args()

    run_cfg = default_cfg
    for k, v in vars(args).items():
        if v is not None:
            run_cfg = replace(run_cfg, **{k: v})

    train_baseline(run_cfg)


if __name__ == "__main__":
    main()
