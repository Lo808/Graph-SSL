from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from wl_gcl.configs.wl import make_wl_cfg
from wl_gcl.configs.wl_dino import make_wl_dino_cfg
from wl_gcl.src.trainers.train_wl import train_wl
from wl_gcl.src.trainers.train_wl_dino import train_wl_dino


DATASET_ORDER = [
    "cora",
    "citeseer",
    "amazon-photo",
    "actor",
    "squirrel",
    "chameleon",
]

DATASET_DISPLAY = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "amazon-photo": "Amazon-Photo",
    "actor": "Actor",
    "squirrel": "Squirrel",
    "chameleon": "Chameleon",
}

METHODS = {
    "wl_baseline": "WL-GIN",
    "dino_only": "DINO-only",
    "wl_only": "WL-only",
    "dino_wl": "DINO+WL",
    "byol": "BYOL",
    "bgrl": "BGRL",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_datasets(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(DATASET_ORDER)
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": float(statistics.mean(values)), "std": float(statistics.stdev(values))}


def fmt_pct(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "N/A"
    return f"{100.0 * v:.2f}%"


def fmt_pct_pm(mean: float | None, std: float | None) -> str:
    if mean is None or std is None or not np.isfinite(mean) or not np.isfinite(std):
        return "N/A"
    return f"{100.0 * mean:.2f}% ± {100.0 * std:.2f}%"


def run_single(
    dataset: str,
    model: str,
    method: str,
    device: str,
    epochs: int | None,
    log_interval: int,
    use_augmentations: bool,
) -> float:
    if method == "wl_baseline":
        cfg = make_wl_cfg(dataset)
        cfg = replace(
            cfg,
            model=model,
            device=device,
            log_interval=log_interval,
            save_best=False,
        )
        if epochs is not None:
            cfg = replace(cfg, epochs=epochs)
        metrics = train_wl(cfg)
        return float(metrics["best_accuracy"])

    cfg = make_wl_dino_cfg(dataset)
    objective = "full"
    if method == "dino_only":
        objective = "dino"
    elif method == "wl_only":
        objective = "wl"
    elif method == "dino_wl":
        objective = "full"
    elif method == "byol":
        objective = "byol"
    elif method == "bgrl":
        objective = "bgrl"
    else:
        raise ValueError(f"Unknown method: {method}")

    cfg = replace(
        cfg,
        model=model,
        objective=objective,
        device=device,
        log_interval=log_interval,
        use_augmentations=use_augmentations,
        save_best=False,
    )
    if epochs is not None:
        cfg = replace(cfg, epochs=epochs)

    metrics = train_wl_dino(cfg)
    return float(metrics["best_accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WL-guided DINO experiments (5-seed summary).")
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--models", type=str, default="gin,wlhn")
    parser.add_argument("--methods", type=str, default="wl_baseline,dino_only,wl_only,dino_wl,byol,bgrl")
    parser.add_argument("--seeds", type=str, default="7,11,19,23,31")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--use_augmentations", action="store_true")
    parser.add_argument("--out_dir", type=str, default="runs/wl_dino_experiments")
    args = parser.parse_args()

    datasets = parse_datasets(args.datasets)
    models = [m.lower() for m in parse_csv(args.models)]
    methods = [m.lower() for m in parse_csv(args.methods)]
    seeds = [int(x) for x in parse_csv(args.seeds)]

    if not datasets:
        raise ValueError("No datasets requested.")
    if not models:
        raise ValueError("No models requested.")
    if not methods:
        raise ValueError("No methods requested.")
    if not seeds:
        raise ValueError("No seeds requested.")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        device = "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_records: List[Dict[str, object]] = []
    aggregated_rows: List[Dict[str, object]] = []

    total_runs = len(datasets) * len(models) * len(methods) * len(seeds)
    run_idx = 0

    for model in models:
        for dataset in datasets:
            by_method: Dict[str, List[float]] = {m: [] for m in methods}

            for method in methods:
                for seed in seeds:
                    run_idx += 1
                    print(
                        f"\n[RUN {run_idx}/{total_runs}] "
                        f"dataset={dataset} model={model} method={method} seed={seed}"
                    )

                    set_seed(seed)
                    acc = run_single(
                        dataset=dataset,
                        model=model,
                        method=method,
                        device=device,
                        epochs=args.epochs,
                        log_interval=args.log_interval,
                        use_augmentations=args.use_augmentations,
                    )
                    by_method[method].append(acc)

                    raw_records.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "method": method,
                            "seed": seed,
                            "best_accuracy": acc,
                        }
                    )

            row: Dict[str, object] = {
                "dataset": dataset,
                "display": DATASET_DISPLAY.get(dataset, dataset),
                "model": model,
            }
            for method in methods:
                stats = mean_std(by_method[method])
                row[f"{method}_mean"] = stats["mean"]
                row[f"{method}_std"] = stats["std"]
            aggregated_rows.append(row)

    raw_fp = out_dir / "raw_results.jsonl"
    with raw_fp.open("w", encoding="utf-8") as f:
        for rec in raw_records:
            f.write(json.dumps(rec) + "\n")

    summary_fp = out_dir / "summary.json"
    summary_fp.write_text(json.dumps(aggregated_rows, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("Summary Tables")
    print("=" * 90)
    for model in models:
        print(f"\nBackbone: {model.upper()}")
        method_headers = [METHODS.get(m, m) for m in methods]
        print("Dataset | " + " | ".join(method_headers))
        print("---|" + "|".join(["---:"] * len(method_headers)))

        rows_for_model = [r for r in aggregated_rows if r["model"] == model]
        rows_for_model.sort(key=lambda r: DATASET_ORDER.index(str(r["dataset"])))

        for row in rows_for_model:
            values = [
                fmt_pct_pm(row.get(f"{m}_mean"), row.get(f"{m}_std"))
                for m in methods
            ]
            print(f"{row['display']} | " + " | ".join(values))

    print(f"\nSaved raw run records: {raw_fp}")
    print(f"Saved summary table data: {summary_fp}")


if __name__ == "__main__":
    main()
