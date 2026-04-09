# wl_gcl/main.py
from __future__ import annotations

import argparse
from dataclasses import replace

from wl_gcl.src.trainers.train_wl import train_wl
from wl_gcl.src.trainers.train_baseline import train_baseline
from wl_gcl.src.trainers.train_wl_hierarchy import train_wl_hierarchy
from wl_gcl.src.trainers.train_wl_dino import train_wl_dino

from wl_gcl.configs.wl import make_wl_cfg
from wl_gcl.configs.wl_hierarchy import make_wl_hierarchy_cfg
from wl_gcl.configs.baseline import make_baseline_cfg
from wl_gcl.configs.wl_dino import make_wl_dino_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph SSL Training Hub")

    parser.add_argument(
        "--method",
        choices=["wl", "baseline", "wl_hierarchy", "wl_hierarchy_bis", "wl_dino"],
        default="wl",
        help="Training method.",
    )
    parser.add_argument(
        "--dataset",
        default="cora",
        help="Dataset name.",
    )
    parser.add_argument(
        "--model",
        choices=["gcn", "gin", "gat", "wlhn"],
        default="gin",
        help="GNN backbone.",
    )
    parser.add_argument(
        "--loss",
        choices=["ce", "triplet"],
        default="triplet",
        help="Alignment Loss",
    )

    args = parser.parse_args()

    # WL-GCL
    if args.method == "wl":
        cfg = make_wl_cfg(args.dataset)
        cfg = replace(cfg, model=args.model)

        print(
            f"[RUN] Method=WL-GCL | "
            f"Dataset={cfg.dataset.upper()} | "
            f"Model={cfg.model.upper()}"
        )

        train_wl(cfg)

    # WL-Hierarchy 
    elif args.method == "wl_hierarchy":
        cfg = make_wl_hierarchy_cfg(args.dataset)
        cfg = replace(cfg, model=args.model)

        print(
            f"[RUN] Method=WL-HIERARCHY | "
            f"Dataset={cfg.dataset.upper()} | "
            f"Model={cfg.model.upper()}"
        )

        train_wl_hierarchy(cfg)
    elif args.method == "wl_dino":
        cfg = make_wl_dino_cfg(args.dataset)
        cfg = replace(cfg, model=args.model)

        print(
            f"[RUN] Method=WL-DINO | "
            f"Dataset={cfg.dataset.upper()} | "
            f"Model={cfg.model.upper()}"
        )

        train_wl_dino(cfg)

    # Baseline
    else:
        cfg = make_baseline_cfg(args.dataset)
        cfg = replace(cfg, model=args.model)

        print(
            f"[RUN] Method=BASELINE | "
            f"Dataset={cfg.dataset.upper()}| "
            f"Model={cfg.model.upper()}"
        )

        train_baseline(cfg)


if __name__ == "__main__":
    main()
