from __future__ import annotations

from dataclasses import dataclass, replace

import torch


DATASET_DEFAULTS = {
    "cora": {
        "lr": 5e-4,
        "hidden_dim": 512,
        "out_dim": 256,
        "epochs": 200,
    },
    "citeseer": {
        "lr": 1e-4,
        "hidden_dim": 512,
        "out_dim": 256,
        "epochs": 250,
    },
    "amazon-photo": {
        "lr": 5e-4,
        "hidden_dim": 512,
        "out_dim": 256,
        "epochs": 250,
    },
    "actor": {
        "lr": 1e-3,
        "hidden_dim": 256,
        "out_dim": 128,
        "epochs": 250,
    },
    "squirrel": {
        "lr": 1e-3,
        "hidden_dim": 256,
        "out_dim": 128,
        "epochs": 250,
    },
    "chameleon": {
        "lr": 1e-3,
        "hidden_dim": 256,
        "out_dim": 128,
        "epochs": 250,
    },
}


@dataclass(frozen=True)
class WLDinoConfig:
    dataset: str = "cora"
    model: str = "gin"

    # Encoder
    hidden_dim: int = 256
    out_dim: int = 128
    dropout: float = 0.1
    tau: float = 2.0
    num_layers: int = 3
    heads: int = 4

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    scheduler: bool = True
    log_interval: int = 10
    batch_size: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Teacher / student distillation
    tau_t: float = 0.05
    tau_s: float = 0.1
    m: float = 0.99
    tau_t_warmup_start: float = 0.04
    tau_t_warmup_epochs: int = 30
    center_momentum: float = 0.9
    distill_space: str = "candidate"  # {"candidate", "prototype"}
    num_prototypes: int = 256
    use_dual_view_miner_pairs: bool = False
    miner_theta: float = 0.8
    miner_delta: int = 2
    miner_refresh_epochs: int = 10

    # WL-guided distribution
    tau_wl: float = 2.0
    lambda_wl: float = 0.5
    wl_loss_type: str = "align"  # {"align", "kl"}
    use_adaptive_wl_balance: bool = False
    wl_balance_ema_momentum: float = 0.9
    wl_balance_eps: float = 1e-8
    wl_balance_min_scale: float = 1.0
    wl_balance_max_scale: float = 1e4
    use_wl_repulsion: bool = False
    wl_repulsion_beta: float = 0.1
    wl_repulsion_threshold: float = 2.0
    wl_depth: int = 4
    k_wl: int = 32
    k_feat: int = 32
    num_random_neg: int = 0
    max_candidates: int = 96
    feat_knn_chunk_size: int = 1024

    # Objective ablation mode
    # - "dino": distill only
    # - "byol": BYOL regression only
    # - "bgrl": BGRL regression only
    # - "wl": WL loss only (selected by wl_loss_type)
    # - "full": distill + lambda_wl * WL loss
    objective: str = "full"
    lambda_sup: float = 0.0

    # Optional augmentations (disabled by default)
    use_augmentations: bool = False
    drop_edge_prob: float = 0.2
    feature_mask_prob: float = 0.2

    # I/O
    save_best: bool = False
    output_dir: str = "runs/wl_dino"


def make_wl_dino_cfg(dataset: str) -> WLDinoConfig:
    base = WLDinoConfig(dataset=dataset)
    overrides = DATASET_DEFAULTS.get(dataset.lower(), {})
    return replace(base, **overrides)
