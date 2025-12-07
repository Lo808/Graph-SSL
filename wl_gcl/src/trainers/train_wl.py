from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from typing import Dict
import random

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

from wl_gcl.src.data_loader.dataset import load_dataset
from wl_gcl.src.models import get_model
from wl_gcl.src.utils.wl_core import WLHierarchyEngine
from wl_gcl.src.augmentations.graph_augmentor import GraphAugmentor
from wl_gcl.src.contrastive.dual_view_miner import DualViewMiner
from wl_gcl.src.contrastive.losses import ExtendedMoCHILoss
from wl_gcl.configs.wl import WLConfig, make_wl_cfg
from wl_gcl.src.trainers.eval import evaluate_linear_probe

# Trainer
def train_wl(cfg: WLConfig) -> Dict[str, float]:
    device = torch.device(cfg.device)

    # Data
    dataset = load_dataset(cfg.dataset)
    data = dataset.data.to(device)

    nodes = list(range(data.num_nodes))
    edges = data.edge_index.t().tolist()

    # WL hierarchy
    wl_engine = WLHierarchyEngine(nodes, edges)
    wl_engine.build_wl_tree(
        max_iterations=10,
        force_convergence=(data.num_nodes < 1000),
    )

    # Encoder
    encoder = get_model(
        name=cfg.model,
        input_dim=dataset.num_features,
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.out_dim,
        dropout=cfg.dropout,
        tau=cfg.tau,
    ).to(device)

    print(
        f"\n[WL-GCL] Dataset={cfg.dataset.upper()} | "
        f"Model={cfg.model.upper()} | "
        f"Number of nodes={data.num_nodes} | "
        f"Number of Edges={data.edge_index.size(1)} | "
        f"Representation dimensionality={cfg.out_dim} | "
        f"epochs={cfg.epochs} | "
        f"device={cfg.device}\n"
    )


    optimizer = Adam(
        encoder.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = (
        CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        if cfg.scheduler
        else None
    )

    # GCL components
    augmentor = GraphAugmentor(
        edge_drop_prob=cfg.drop_edge_prob,
        feature_mask_prob=cfg.feature_mask_prob,
    )

    miner = DualViewMiner(
        wl_engine,
        nodes,
        theta=cfg.theta,
        delta=cfg.delta,
    )

    criterion = ExtendedMoCHILoss(
        temperature=cfg.temperature,
        num_negatives=cfg.num_negatives,
    )

    # Training
    best_acc = 0.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad()

        # Mining warm-up cycle
        if epoch % 10 == 1:
            with torch.no_grad():
                encoder.eval()
                h = encoder(data.x, data.edge_index)
                ext_pos, hard_neg = miner.mine_candidates(h)
                encoder.train()

        # Views
        x1, ei1 = augmentor.augment(data.x, data.edge_index)
        x2, ei2 = augmentor.augment(data.x, data.edge_index)

        z1 = encoder(x1, ei1)
        z2 = encoder(x2, ei2)

        # Contrastive loss (batch subset)
        batch_size = min(cfg.batch_size, data.num_nodes)
        batch_idx = torch.randperm(data.num_nodes, device=device)[:batch_size]

        total_loss = 0.0
        for idx in batch_idx:
            i = idx.item()
            anchor = z1[i].unsqueeze(0)
            pos_v2 = z2[i].unsqueeze(0)
            
            # Positive Sampling 
            if len(ext_pos) > 0 and len(ext_pos[i]) > 0:
                indices = ext_pos[i]
                # Limit extended positives to avoid noise
                if len(indices) > 5: indices = random.sample(indices, 5)
                pos_ext = z2[indices]
                all_pos = torch.cat([pos_v2, pos_ext], dim=0)
            else:
                all_pos = pos_v2
                
            # Negative Sampling 
            if len(hard_neg) > 0 and len(hard_neg[i]) > 0:
                indices = hard_neg[i]
                # Limit hard negatives
                if len(indices) > 20: indices = random.sample(indices, 20)
                h_neg = z2[indices]
            else:
                h_neg = torch.tensor([]).to(cfg.device)
                
            total_loss += criterion(anchor, all_pos, h_neg)

        loss = total_loss / len(batch_idx)
        loss.backward()

        optimizer.step()
        if scheduler:
            scheduler.step()

        # Logging & evaluation
        if epoch % cfg.log_interval == 0 or epoch == cfg.epochs:
            acc = evaluate_linear_probe(
                encoder,
                data,
                dataset.num_classes,
                device,
            )

            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(encoder.state_dict())

            print(
                f"[WL-GCL | {cfg.dataset:<12}] "
                f"Epoch {epoch:03d}/{cfg.epochs}  "
                f"Loss: {loss.item():.4f}  "
                f"Acc: {acc:.4f}"
            )

    print(f"[WL-GCL | {cfg.dataset.upper():<12}] Best Acc: {best_acc:.4f}")

    return {
        "dataset": cfg.dataset,
        "best_accuracy": best_acc,
        "epochs": cfg.epochs,
    }


# CLI

def main() -> None:

    parser = argparse.ArgumentParser(description="Run WL-GCL training")

    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--out_dim", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--theta", type=float)
    parser.add_argument("--delta", type=int)
    parser.add_argument("--drop_edge_prob", type=float)
    parser.add_argument("--feature_mask_prob", type=float)
    parser.add_argument("--num_negatives", type=int)
    parser.add_argument("--device")

    args = parser.parse_args()

    run_cfg = make_wl_cfg(args.dataset or "cora")
    for k, v in vars(args).items():
        if v is not None:
            run_cfg = replace(run_cfg, **{k: v})
    print(run_cfg)


    train_wl(run_cfg)


if __name__ == "__main__":
    main()
