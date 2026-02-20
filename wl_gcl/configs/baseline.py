from dataclasses import dataclass, replace
import torch


DATASET_DEFAULTS = {
    "cora": {
        "lr": 0.0005,
        "hidden_dim": 512,
        "out_dim": 256,
        "epochs": 200,
        "temperature": 0.2,
        "drop_edge_prob": 0.25,
        "feature_mask_prob": 0.25,
    },
    "citeseer": {
        "lr": 0.0001,
        "hidden_dim": 512,
        "out_dim": 256,
        "epochs": 300,
        "temperature": 0.5,
        "drop_edge_prob": 0.10,
        "feature_mask_prob": 0.10,
    },
    "amazon-photo": {
        "lr": 0.0005,
        "hidden_dim": 512,
        "out_dim": 256,
        "epochs": 500,
        "temperature": 0.2,
        "drop_edge_prob": 0.30,
        "feature_mask_prob": 0.30,
    },
    "actor": {
        "lr": 0.001,
        "hidden_dim": 256,
        "out_dim": 128,
        "epochs": 400,
        "temperature": 0.5,
        "drop_edge_prob": 0.40,
        "feature_mask_prob": 0.10,
    },
    "squirrel": {
        "lr": 0.001,
        "hidden_dim": 256,
        "out_dim": 128,
        "epochs": 500,
        "temperature": 0.5,
        "drop_edge_prob": 0.50,
        "feature_mask_prob": 0.20,
    },
    "chameleon": {
        "lr": 0.001,
        "hidden_dim": 256,
        "out_dim": 128,
        "epochs": 500,
        "temperature": 0.5,
        "drop_edge_prob": 0.50,
        "feature_mask_prob": 0.20,
    },
    "texas": {
        "lr": 0.01,
        "hidden_dim": 64,
        "out_dim": 32,
        "epochs": 400,
        "temperature": 0.5,
        "drop_edge_prob": 0.40,
        "feature_mask_prob": 0.10,
    },
    "wisconsin": {
        "lr": 0.01,
        "hidden_dim": 64,
        "out_dim": 32,
        "epochs": 400,
        "temperature": 0.5,
        "drop_edge_prob": 0.40,
        "feature_mask_prob": 0.10,
    },
}

@dataclass(frozen=True)
class BaselineConfig:
    dataset: str = "Cora"
    model: str = "gin"

    hidden_dim: int = 128
    out_dim: int = 64
    dropout: float = 0.5

    #GIN specific params
    tau: float = 2.0
    num_layers: int = 3

    #GAT specific params
    heads: int = 4

    lr: float = 1e-3
    temperature: float = 0.2
    epochs: int = 200
    log_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler: bool = True
    batch_size: int = 512

    # Graph augmentation
    drop_edge_prob: float = 0.4
    feature_mask_prob: float = 0.1

    save_best: bool = False
    output_dir: str = "runs/baseline"
    



def make_baseline_cfg(dataset: str) -> BaselineConfig:
    base = BaselineConfig(dataset=dataset)
    overrides = DATASET_DEFAULTS.get(dataset.lower(), {})

    return replace(base, **overrides)


cfg = make_baseline_cfg("cora")