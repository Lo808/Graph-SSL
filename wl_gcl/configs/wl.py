# wl_gcl/configs/wl.py
from dataclasses import dataclass, replace


# Base WL Config (typed defaults)

@dataclass(frozen=True)
class WLConfig:
    dataset: str = "cora"
    model: str = "gin"

    hidden_dim: int = 256
    out_dim: int = 128
    dropout: float = 0.1

    #GIN specific params
    tau: float = 2.0
    num_layers: int = 3

    #GAT specific params
    heads: int = 4

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 400
    scheduler: bool = True
    log_interval: int = 10
    device: str = "cpu"

    # WL Mining
    theta: float = 0.5
    delta: int = 2

    # Loss
    temperature: float = 0.5
    num_negatives: int = 256
    batch_size: int = 512

    # Augmentation
    drop_edge_prob: float = 0.4
    feature_mask_prob: float = 0.1


# Dataset–Specific Overrides 

DATASET_DEFAULTS = {
    "cora": {
        "lr": 0.0005,
        "hidden_dim": 512,
        "out_dim": 256,
        "epochs": 200,
        "theta": 0.6,
        "temperature": 0.2,
        "num_negatives": 256,
        "drop_edge_prob": 0.25,
        "feature_mask_prob": 0.25,
    },
    "citeseer": {
        "lr": 0.0001,
        "hidden_dim": 512,
        "out_dim": 256,
        "epochs": 300,
        "theta": 0.75,
        "temperature": 0.5,
        "num_negatives": 128,
        "drop_edge_prob": 0.10,
        "feature_mask_prob": 0.10,
    },
    "amazon-photo": {
        "lr": 0.0005,
        "hidden_dim": 512,
        "out_dim": 256,
        "epochs": 200,
        "theta": 0.7,
        "temperature": 0.2,
        "num_negatives": 512,
        "drop_edge_prob": 0.30,
        "feature_mask_prob": 0.30,
    },
    "actor": {
        "lr": 0.001,
        "hidden_dim": 256,
        "out_dim": 128,
        "epochs": 400,
        "theta": 0.5,
        "temperature": 0.5,
        "num_negatives": 256,
        "drop_edge_prob": 0.40,
        "feature_mask_prob": 0.10,
    },
    "squirrel": {
        "lr": 0.001,
        "hidden_dim": 256,
        "out_dim": 128,
        "epochs": 500,
        "theta": 0.4,
        "temperature": 0.5,
        "num_negatives": 512,
        "drop_edge_prob": 0.50,
        "feature_mask_prob": 0.20,
    },
    "chameleon": {
        "lr": 0.001,
        "hidden_dim": 256,
        "out_dim": 128,
        "epochs": 500,
        "theta": 0.4,
        "temperature": 0.5,
        "num_negatives": 512,
        "drop_edge_prob": 0.50,
        "feature_mask_prob": 0.20,
    },
    "texas": {
        "lr": 0.01,
        "hidden_dim": 64,
        "out_dim": 32,
        "epochs": 400,
        "theta": 0.5,
        "temperature": 0.5,
        "num_negatives": 32,
        "drop_edge_prob": 0.40,
        "feature_mask_prob": 0.10,
    },
    "wisconsin": {
        "lr": 0.01,
        "hidden_dim": 64,
        "out_dim": 32,
        "epochs": 400,
        "theta": 0.5,
        "temperature": 0.5,
        "num_negatives": 32,
        "drop_edge_prob": 0.40,
        "feature_mask_prob": 0.10,
    },
}


# Factory — merges dataset defaults with base config

def make_wl_cfg(dataset: str) -> WLConfig:
    base = WLConfig(dataset=dataset)
    overrides = DATASET_DEFAULTS.get(dataset.lower(), {})

    return replace(base, **overrides)


# Default config instance
cfg = make_wl_cfg("cora")
