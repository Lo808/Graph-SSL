import torch
import torch.nn as nn


class WLMultilevelModel(nn.Module):
    """
    Encoder + one linear classification head per WL level.
    """

    def __init__(self, encoder: nn.Module, out_dim: int, level_targets: dict):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict()

        for t, info in level_targets.items():
            self.heads[str(t)] = nn.Linear(out_dim, info["num_classes"])

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)  # [N, d]
        logits = {int(t): head(z) for t, head in self.heads.items()}
        return z, logits
