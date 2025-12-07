from dataclasses import dataclass

@dataclass(frozen=True)
class BaselineConfig:
    dataset: str = "Cora"
    hidden_dim: int = 128
    out_dim: int = 64
    dropout: float = 0.5
    lr: float = 1e-3
    temperature: float = 0.2
    epochs: int = 200
    log_interval: int = 10
    device: str = "cpu"

    # Graph augmentation
    edge_drop_prob: float = 0.2
    feature_mask_prob: float = 0.2

cfg = BaselineConfig()