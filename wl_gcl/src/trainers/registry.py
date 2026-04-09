# wl_gcl/src/trainers/registry.py

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Mapping, Any

from wl_gcl.src.trainers import train_wl
from wl_gcl.src.trainers import train_baseline
from wl_gcl.src.trainers import train_wl_hierarchy
from wl_gcl.src.trainers import train_wl_dino

TrainerFn = Callable[[Any], Dict[str, float]]

_TRAINERS: Mapping[str, TrainerFn] = {
    "wl": train_wl,
    "baseline": train_baseline,
    "wl_hierarchy": train_wl_hierarchy,
    "wl_dino": train_wl_dino,
}

def get_trainer(name: str) -> TrainerFn:
    key = name.strip().lower()
    if key not in _TRAINERS:
        raise KeyError(
            f"Unknown trainer '{name}'. Available: {sorted(_TRAINERS.keys())}"
        )
    return _TRAINERS[key]

def available_trainers() -> list[str]:
    return sorted(_TRAINERS.keys())
