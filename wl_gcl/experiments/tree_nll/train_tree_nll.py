from __future__ import annotations

import copy
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Dict, Any

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from wl_gcl.src.data_loader.dataset import load_dataset
from wl_gcl.src.models import get_model
from wl_gcl.src.models.wl_multilevel import WLMultilevelModel
from wl_gcl.src.utils.wl_core import WLHierarchyEngine
from wl_gcl.src.losses.wl_losses import (
    hierarchy_regularization,
    wl_contrastive_loss,
    wl_tree_nll_loss,
)
from wl_gcl.src.trainers.eval import evaluate_linear_probe


DATASET_DEFAULTS = {
    "cora": {
        "lr": 5e-4,
        "hidden_dim": 512,
        "out_dim": 256,
        "epochs": 200,
        "temperature": 0.2,
        "num_negatives": 256,
    },
    "citeseer": {
        "lr": 1e-4,
        "hidden_dim": 512,
        "out_dim": 256,
        "epochs": 300,
        "temperature": 0.5,
        "num_negatives": 128,
    },
    "amazon-photo": {
        "lr": 5e-4,
        "hidden_dim": 512,
        "out_dim": 256,
        "epochs": 500,
        "temperature": 0.2,
        "num_negatives": 512,
    },
    "actor": {
        "lr": 1e-3,
        "hidden_dim": 256,
        "out_dim": 128,
        "epochs": 400,
        "temperature": 0.5,
        "num_negatives": 256,
    },
    "squirrel": {
        "lr": 1e-3,
        "hidden_dim": 256,
        "out_dim": 128,
        "epochs": 500,
        "temperature": 0.5,
        "num_negatives": 512,
    },
    "chameleon": {
        "lr": 1e-3,
        "hidden_dim": 256,
        "out_dim": 128,
        "epochs": 500,
        "temperature": 0.5,
        "num_negatives": 512,
    },
}


@dataclass(frozen=True)
class TreeNLLConfig:
    dataset: str = "cora"
    model: str = "gin"

    # Encoder
    hidden_dim: int = 256
    out_dim: int = 128
    dropout: float = 0.1

    # GIN / GAT
    tau: float = 2.0
    num_layers: int = 3
    heads: int = 4

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 300
    scheduler: bool = True
    log_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # WL hierarchy curriculum
    warmup: int = 40
    wl_max_iter: int = 10

    # Loss weights
    lambda_hier: float = 1e-2
    lambda_nce: float = 1.0

    # Contrastive
    temperature: float = 0.5
    num_negatives: int = 256

    save_best: bool = True
    output_dir: str = "runs/wl_hierarchy_tree_nll"


def make_tree_nll_cfg(dataset: str) -> TreeNLLConfig:
    base = TreeNLLConfig(dataset=dataset)
    overrides = DATASET_DEFAULTS.get(dataset.lower(), {})
    return replace(base, **overrides)


def _levels_for_epoch(sorted_levels: list[int], epoch: int, warmup: int) -> list[int]:
    if not sorted_levels:
        return []
    stride = max(1, warmup // max(1, len(sorted_levels)))
    k = min(len(sorted_levels), 1 + epoch // stride)
    return sorted_levels[:k]


def train_tree_nll(cfg: TreeNLLConfig) -> Dict[str, Any]:
    device = torch.device(cfg.device)

    dataset = load_dataset(cfg.dataset)
    data = dataset.data.to(device)

    nodes = list(range(data.num_nodes))
    edges = data.edge_index.t().tolist()

    engine = WLHierarchyEngine(nodes, edges)
    engine.build_wl_tree(
        max_iterations=cfg.wl_max_iter,
        force_convergence=(data.num_nodes < 1000),
    )

    level_targets: Dict[int, Dict[str, Any]] = {}
    all_levels = sorted({t for v in nodes for t in engine.node_path[v].keys()})

    for t in all_levels:
        if t == 0:
            continue
        y, _cid2idx, num_classes = engine.get_level_targets(t)
        if y is not None and num_classes is not None:
            level_targets[t] = {
                "y": y.to(device),
                "num_classes": int(num_classes),
            }

    sorted_levels = sorted(level_targets.keys())
    if not sorted_levels:
        raise RuntimeError("No WL levels available for Tree-NLL training.")

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
        f"\n[WL-HIERARCHY-TREE-NLL] Dataset={cfg.dataset.upper()} | "
        f"Model={cfg.model.upper()} | "
        f"epochs={cfg.epochs} | device={cfg.device}\n"
    )

    best_acc = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        z, logits = model(data.x, data.edge_index)

        active_levels = _levels_for_epoch(sorted_levels, epoch, cfg.warmup)

        # Replace level-wise CE by Tree Path NLL.
        loss_tree = wl_tree_nll_loss(engine, logits, active_levels)

        max_t = max(active_levels)
        reg_levels = [t for t in range(1, max_t + 1) if t in sorted_levels]
        loss_hier = hierarchy_regularization(engine, z, reg_levels)

        t_contrast = max(active_levels)
        loss_nce = wl_contrastive_loss(
            engine,
            z,
            level=t_contrast,
            temperature=cfg.temperature,
        )

        loss = loss_tree + cfg.lambda_hier * loss_hier + cfg.lambda_nce * loss_nce

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if epoch % cfg.log_interval == 0 or epoch == cfg.epochs:
            acc = evaluate_linear_probe(
                encoder,
                data,
                dataset.num_classes,
                device,
            )

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_state = copy.deepcopy(encoder.state_dict())

            print(
                f"[TREE-NLL | {cfg.dataset:<12}] "
                f"Epoch {epoch:03d}/{cfg.epochs} "
                f"Loss: {loss.item():.4f} "
                f"Tree: {loss_tree.item():.4f} "
                f"Hier: {loss_hier.item():.4f} "
                f"NCE: {loss_nce.item():.4f} "
                f"Acc: {acc:.4f}"
            )

    print(f"[TREE-NLL | {cfg.dataset.upper():<12}] Best Acc: {best_acc:.4f} @ epoch {best_epoch}")

    best_ckpt_path = None
    if cfg.save_best and best_state is not None:
        out_dir = Path(cfg.output_dir) / cfg.dataset / cfg.model
        out_dir.mkdir(parents=True, exist_ok=True)

        best_ckpt_path = out_dir / "best_encoder.pt"
        torch.save(
            {
                "encoder_state_dict": best_state,
                "best_accuracy": best_acc,
                "best_epoch": best_epoch,
                "cfg": asdict(cfg),
            },
            best_ckpt_path,
        )

    return {
        "dataset": str(cfg.dataset),
        "best_accuracy": float(best_acc),
        "best_epoch": int(best_epoch),
        "epochs": int(cfg.epochs),
        "best_ckpt_path": str(best_ckpt_path) if best_ckpt_path is not None else None,
    }
