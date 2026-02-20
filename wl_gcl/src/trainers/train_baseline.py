# wl_gcl/src/trainers/train_baseline.py
from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Dict
import copy
from pathlib import Path
import json

import torch
from torch.optim import Adam

from wl_gcl.src.data_loader.dataset import load_dataset
from wl_gcl.src.models import get_model
from wl_gcl.src.contrastive.losses import nt_xent_loss
from wl_gcl.src.augmentations.graph_augmentor import GraphAugmentor
from wl_gcl.configs.baseline import make_baseline_cfg
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
    model =  get_model(
        name=cfg.model,
        input_dim=dataset.num_features,
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.out_dim,
        dropout=cfg.dropout,
        tau=cfg.tau,  #gin
        num_layers=cfg.num_layers,  #gin
        heads=cfg.heads  #gat 
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr)

    # Augmentor
    augmentor = GraphAugmentor(
        edge_drop_prob=cfg.drop_edge_prob,
        feature_mask_prob=cfg.feature_mask_prob,
    )

    best_acc = 0.0
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
            acc = evaluate_linear_probe(
                model,
                data,
                dataset.num_classes,
                device,
            )

            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())

            print(
                f"[Baseline | {cfg.dataset:<12}] "
                f"Epoch {epoch:03d}/{cfg.epochs}  "
                f"Loss: {loss.item():.4f}"
            )

    print(f"[WL-BASELINE | {cfg.dataset.upper():<12}] Best Acc: {best_acc:.4f}")

    best_ckpt_path = None

    if getattr(cfg, "save_best", True) and best_state is not None:
        out_dir = Path(getattr(cfg, "output_dir", "runs/wl_hierarchy")) / cfg.dataset / cfg.model
        out_dir.mkdir(parents=True, exist_ok=True)

        best_ckpt_path = out_dir / "best_encoder.pt"
        torch.save(
            {
                "encoder_state_dict": best_state,
                "best_accuracy": best_acc,
                "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else None,
            },
            best_ckpt_path,
        )

    return {
        "dataset": str(cfg.dataset),
        "best_accuracy": float(best_acc),
        "epochs": int(cfg.epochs),
        "best_ckpt_path": str(best_ckpt_path) if best_ckpt_path is not None else None,
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

    run_cfg = make_baseline_cfg(args.dataset or "cora")
    for k, v in vars(args).items():
        if v is not None:
            run_cfg = replace(run_cfg, **{k: v})
    print(run_cfg)

    train_baseline(run_cfg)


if __name__ == "__main__":
    main()
