from __future__ import annotations

import copy
from typing import Dict
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from wl_gcl.src.data_loader.dataset import load_dataset
from wl_gcl.src.models import get_model
from wl_gcl.src.models.wl_multilevel import WLMultilevelModel
from wl_gcl.src.utils.wl_core import WLHierarchyEngine
from wl_gcl.src.losses.wl_losses import (
    hierarchy_regularization,
    wl_contrastive_loss,
)
from wl_gcl.src.trainers.eval import evaluate_linear_probe


def _levels_for_epoch(sorted_levels, epoch, warmup):
    k = min(len(sorted_levels), 1 + epoch // max(1, warmup // len(sorted_levels)))
    return sorted_levels[:k]


def train_wl_hierarchy(cfg) -> Dict[str, float]:

    device = torch.device(cfg.device)

    # -------- DATA --------
    dataset = load_dataset(cfg.dataset)
    data = dataset.data.to(device)

    nodes = list(range(data.num_nodes))
    edges = data.edge_index.t().tolist()

    # -------- WL ENGINE --------
    engine = WLHierarchyEngine(nodes, edges)
    engine.build_wl_tree(max_iterations=10, force_convergence=(data.num_nodes < 1000))

    # -------- LEVEL TARGETS --------
    level_targets = {}
    all_levels = sorted({t for v in nodes for t in engine.node_path[v].keys()})

    for t in all_levels:
        if t == 0:
            continue
        y, cid2idx, C = engine.get_level_targets(t)
        if y is not None:
            level_targets[t] = {
                "y": y.to(device),
                "num_classes": C,
            }

    sorted_levels = sorted(level_targets.keys())

    # -------- ENCODER --------
    encoder = get_model(
        name=cfg.model,
        input_dim=dataset.num_features,
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.out_dim,
        dropout=cfg.dropout,
        tau=cfg.tau,
        num_layers=cfg.num_layers,
        heads=cfg.heads,
    ).to(device)

    model = WLMultilevelModel(encoder, cfg.out_dim, level_targets).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs) if cfg.scheduler else None

    print(
        f"\n[WL-HIERARCHY] Dataset={cfg.dataset.upper()} | "
        f"Model={cfg.model.upper()} | "
        f"epochs={cfg.epochs} | device={cfg.device}\n"
    )

    # -------- TRAINING --------
    best_acc = 0.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):

        model.train()
        optimizer.zero_grad()

        z, logits = model(data.x, data.edge_index)

        active_levels = _levels_for_epoch(sorted_levels, epoch, cfg.warmup)

        # WL classification
        loss_wl = torch.tensor(0.0, device=device)
        for t in active_levels:
            y_t = level_targets[t]["y"]
            loss_wl += F.cross_entropy(logits[t], y_t)

        # Hierarchy regularization
        max_t = max(active_levels)
        reg_levels = [t for t in range(1, max_t + 1) if t in sorted_levels]
        loss_hier = hierarchy_regularization(engine, z, reg_levels)

        # WL contrastive
        t_contrast = max(active_levels)
        loss_nce = wl_contrastive_loss(
            engine,
            z,
            level=t_contrast,
            temperature=cfg.temperature,
        )

        loss = (
            loss_wl
            + cfg.lambda_hier * loss_hier
            + cfg.lambda_nce * loss_nce
        )

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # -------- EVAL --------
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
                f"[WL-HIERARCHY | {cfg.dataset:<12}] "
                f"Epoch {epoch:03d}/{cfg.epochs}  "
                f"Loss: {loss.item():.4f}  "
                f"Acc: {acc:.4f}"
            )

    print(f"[WL-HIERARCHY | {cfg.dataset.upper():<12}] Best Acc: {best_acc:.4f}")

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
