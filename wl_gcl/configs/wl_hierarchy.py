# wl_gcl/configs/wl_hierarchy.py
from dataclasses import dataclass, replace
import torch


# ------------------------------------------------------------
# Dataset–Specific Overrides (very light at the beginning)
# You can tune later after experiments
# ------------------------------------------------------------

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
        "epochs": 200,
        "temperature": 0.2,
        "num_negatives": 512,
    },
}


# ------------------------------------------------------------
# WL-Hierarchy Config
# ------------------------------------------------------------

@dataclass(frozen=True)
class WLHierarchyConfig:
    dataset: str = "cora"
    model: str = "gin"

    # Encoder
    hidden_dim: int = 256
    out_dim: int = 128
    dropout: float = 0.1

    # GIN / GAT specific
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

    # WL Hierarchy curriculum
    warmup: int = 40           # how fast to add WL levels
    wl_max_iter: int = 10

    # Loss weights (VERY important)
    lambda_hier: float = 1e-2
    lambda_nce: float = 1.0

    # Contrastive
    temperature: float = 0.5
    num_negatives: int = 256


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------

def make_wl_hierarchy_cfg(dataset: str) -> WLHierarchyConfig:
    base = WLHierarchyConfig(dataset=dataset)
    overrides = DATASET_DEFAULTS.get(dataset.lower(), {})
    return replace(base, **overrides)


# Default instance
cfg = make_wl_hierarchy_cfg("cora")
