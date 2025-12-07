# wl_gcl/main.py
from __future__ import annotations

import argparse
from dataclasses import replace

from wl_gcl.src.trainers.train_wl import train_wl
from wl_gcl.src.trainers.train_baseline import train_baseline

from wl_gcl.configs.wl import make_wl_cfg
from wl_gcl.configs.baseline import cfg as baseline_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph SSL Training Hub")

    parser.add_argument(
        "--method",
        choices=["wl", "baseline"],
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
        help="GNN backbone (WL-GCL only).",
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

    # Baseline
    else:
        cfg = replace(baseline_cfg, dataset=args.dataset)

        print(
            f"[RUN] Method=BASELINE | "
            f"Dataset={cfg.dataset.upper()}"
        )

        train_baseline(cfg)


if __name__ == "__main__":
    main()
