from __future__ import annotations

from typing import Any

from wl_gcl.configs.wl_hierarchy import make_wl_hierarchy_cfg
from wl_gcl.configs.baseline import make_baseline_cfg
from wl_gcl.configs.wl import make_wl_cfg
from wl_gcl.configs.wl_dino import make_wl_dino_cfg


def make_cfg(trainer: str, dataset: str) -> Any:
    t = trainer.strip().lower()
    dataset = dataset.strip().lower()
    if t == "wl_hierarchy":
        return make_wl_hierarchy_cfg(dataset)
    if t == "wl":
        return make_wl_cfg(dataset)
    if t == "baseline":
        return make_baseline_cfg(dataset)
    if t == "wl_dino":
        return make_wl_dino_cfg(dataset)
    raise KeyError(f"Unknown trainer '{trainer}'")
