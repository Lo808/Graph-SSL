from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
import torch

from wl_gcl.configs.wl_dino import WLDinoConfig, make_wl_dino_cfg
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

METHOD_TO_OBJECTIVE = {
    "dino_wl": "full",
    "dino_only": "dino",
    "byol": "byol",
    "bgrl": "bgrl",
    "wl_only": "wl",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    raise TypeError(f"Expected dataclass config, got: {type(cfg)}")


def stable_trial_id(cfg: Any) -> str:
    payload = json.dumps(cfg_to_dict(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def apply_params(cfg: Any, params: Dict[str, Any]) -> Any:
    fields = {f.name for f in dataclasses.fields(cfg)}
    filtered = {k: v for k, v in params.items() if k in fields}
    return replace(cfg, **filtered) if filtered else cfg


def grid_product(space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(space.keys())
    for values in itertools.product(*[space[k] for k in keys]):
        yield dict(zip(keys, values))


def sample_random(space: Dict[str, List[Any]], rng: random.Random) -> Dict[str, Any]:
    return {k: rng.choice(v) for k, v in space.items()}


def search_space(method: str, model: str, include_miner: bool) -> Dict[str, List[Any]]:
    method = method.lower()
    model = model.lower()

    space: Dict[str, List[Any]] = {
        "lr": [1e-4, 3e-4, 5e-4, 1e-3],
        "weight_decay": [1e-6, 1e-5, 1e-4],
        "hidden_dim": [256, 512],
        "out_dim": [128, 256],
        "dropout": [0.0, 0.1, 0.2],
        "num_layers": [2, 3, 4],
        "m": [0.99, 0.995, 0.998],
        "batch_size": [256, 512],
        "drop_edge_prob": [0.1, 0.2, 0.3],
        "feature_mask_prob": [0.1, 0.2, 0.3],
        "use_augmentations": [True],
        "scheduler": [True],
        "wl_depth": [4],
        "k_wl": [32],
        "k_feat": [32],
        "max_candidates": [96],
    }

    if model == "gat":
        space["heads"] = [2, 4, 8]

    if method in {"dino_wl", "dino_only", "wl_only"}:
        space.update(
            {
                "distill_space": ["prototype"],
                "num_prototypes": [128, 256, 512],
                "tau_t": [0.04, 0.05, 0.07],
                "tau_s": [0.07, 0.1, 0.2],
                "tau_t_warmup_start": [0.02, 0.04],
                "tau_t_warmup_epochs": [10, 30, 50],
                "center_momentum": [0.9, 0.95],
            }
        )

    if method == "dino_wl":
        space.update(
            {
                "wl_loss_type": ["kl"],
                "lambda_wl": [25.0, 50.0, 100.0, 200.0],
                "tau_wl": [1.0, 2.0, 4.0],
            }
        )
    elif method == "wl_only":
        space.update(
            {
                "wl_loss_type": ["kl", "align"],
                "lambda_wl": [0.5, 1.0, 2.0, 10.0, 50.0, 100.0],
                "tau_wl": [1.0, 2.0, 4.0],
            }
        )
    else:
        # BYOL/BGRL/DINO-only do not use WL loss in objective.
        space["lambda_wl"] = [0.0]

    if include_miner:
        space.update(
            {
                "use_dual_view_miner_pairs": [False, True],
                "miner_theta": [0.6, 0.8],
                "miner_delta": [2, 3],
                "miner_refresh_epochs": [5, 10],
            }
        )

    return space


def results_path(out_dir: Path) -> Path:
    return out_dir / "results.jsonl"


def iter_results_jsonl(fp: Path) -> Iterable[Dict[str, Any]]:
    if not fp.exists():
        return
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_completed_and_records(out_dir: Path) -> Tuple[Set[str], List[Dict[str, Any]]]:
    fp = results_path(out_dir)
    records = list(iter_results_jsonl(fp))
    completed = {r.get("trial_id") for r in records if isinstance(r, dict)}
    completed.discard(None)
    return completed, records


def objective(metrics: Dict[str, Any]) -> float:
    return float(metrics.get("best_accuracy", -1.0))


def select_best(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise RuntimeError("No records available to select best from.")
    return max(records, key=lambda r: objective(r.get("metrics", {})))


def append_result(out_dir: Path, record: Dict[str, Any]) -> None:
    fp = results_path(out_dir)
    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def write_best(out_dir: Path, best: Dict[str, Any]) -> None:
    (out_dir / "best.json").write_text(
        json.dumps(best, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def sample_unique_random_candidates(
    space: Dict[str, List[Any]],
    rng: random.Random,
    make_cfg_fn,
    completed_ids: set[str],
    n_trials: int,
    max_attempts: int = 20000,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    attempts = 0
    reserved = set(completed_ids)

    while len(out) < n_trials and attempts < max_attempts:
        attempts += 1
        params = sample_random(space, rng)
        cfg = make_cfg_fn(params)
        tid = stable_trial_id(cfg)
        if tid in reserved:
            continue
        out.append(params)
        reserved.add(tid)

    if len(out) < n_trials:
        print(
            f"[WARN] Sampled only {len(out)}/{n_trials} new trials "
            f"after {attempts} attempts."
        )

    return out


def run_trial(cfg: WLDinoConfig, trial_seed: int) -> Dict[str, Any]:
    set_seed(trial_seed)
    t0 = time.time()
    metrics = train_wl_dino(cfg)
    dt = time.time() - t0
    return {
        "trial_id": stable_trial_id(cfg),
        "seed": trial_seed,
        "wall_time_sec": round(dt, 3),
        "cfg": cfg_to_dict(cfg),
        "metrics": metrics,
    }


def parse_datasets(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(DATASET_ORDER)
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "N/A"
    return f"{100.0 * x:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune WL-DINO family methods (DINO+WL, DINO, BYOL, BGRL)."
    )
    parser.add_argument("--datasets", type=str, default="cora")
    parser.add_argument("--model", type=str, default="gin", choices=["gin", "gcn", "gat", "wlhn"])
    parser.add_argument(
        "--method",
        type=str,
        default="dino_wl",
        choices=["dino_wl", "dino_only", "byol", "bgrl", "wl_only"],
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--search", type=str, default="random", choices=["random", "grid"])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--max_trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="runs/tune_wl_dino")
    parser.add_argument("--include_miner", action="store_true")
    parser.add_argument("--save_best_encoder", action="store_true")
    args = parser.parse_args()

    datasets = parse_datasets(args.datasets)
    if not datasets:
        raise ValueError("No datasets provided.")

    print(f"Device requested: {args.device}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    summary_rows: List[Dict[str, Any]] = []

    for dsi, dataset in enumerate(datasets):
        print("\n" + "=" * 80)
        print(
            f"[DATASET {dsi + 1}/{len(datasets)}] {dataset} | "
            f"method={args.method} | model={args.model}"
        )
        print("=" * 80)

        out_dir = Path(args.out_dir) / args.method / args.model / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        completed_ids, records = load_completed_and_records(out_dir)
        space = search_space(args.method, args.model, include_miner=args.include_miner)
        rng = random.Random(args.seed + dsi)

        def make_trial_cfg(params: Dict[str, Any]) -> WLDinoConfig:
            cfg = make_wl_dino_cfg(dataset)
            cfg = apply_params(cfg, params)
            cfg = apply_params(
                cfg,
                {
                    "model": args.model,
                    "objective": METHOD_TO_OBJECTIVE[args.method],
                    "device": args.device,
                    "log_interval": args.log_interval,
                    "save_best": args.save_best_encoder,
                    "output_dir": str(Path(args.out_dir) / "checkpoints" / args.method / args.model),
                },
            )
            if args.epochs is not None:
                cfg = apply_params(cfg, {"epochs": args.epochs})

            # Safety: BYOL/BGRL and DINO-only do not use WL objective.
            if args.method in {"dino_only", "byol", "bgrl"}:
                cfg = apply_params(cfg, {"lambda_wl": 0.0})

            # Strong default for DINO+WL tuning.
            if args.method == "dino_wl":
                cfg = apply_params(cfg, {"distill_space": "prototype", "wl_loss_type": "kl"})

            return cfg

        if args.search == "grid":
            candidates = list(grid_product(space))
        else:
            candidates = sample_unique_random_candidates(
                space=space,
                rng=rng,
                make_cfg_fn=make_trial_cfg,
                completed_ids=completed_ids,
                n_trials=args.n_trials,
            )

        if args.max_trials is not None:
            candidates = candidates[: args.max_trials]

        print(f"Existing trials: {len(completed_ids)}")
        print(f"Planned candidates: {len(candidates)}")

        ran_any = False
        for i, params in enumerate(candidates, start=1):
            cfg = make_trial_cfg(params)
            tid = stable_trial_id(cfg)

            if tid in completed_ids:
                continue

            ran_any = True
            print(f"\n--- [{i}/{len(candidates)}] Trial {tid} ({dataset}) ---")
            print(json.dumps(params, indent=2, sort_keys=True))

            rec = run_trial(cfg, trial_seed=args.seed + i)
            append_result(out_dir, rec)
            records.append(rec)
            completed_ids.add(tid)

            best = select_best(records)
            write_best(out_dir, best)
            print(
                f"[BEST {dataset}] acc={best['metrics'].get('best_accuracy', -1.0):.4f} "
                f"trial={best['trial_id']}"
            )

        if not records:
            print(f"[WARN] No records for dataset={dataset}")
            summary_rows.append(
                {
                    "dataset": dataset,
                    "display": DATASET_DISPLAY.get(dataset, dataset),
                    "model": args.model,
                    "method": args.method,
                    "best_accuracy": None,
                }
            )
            continue

        best = select_best(records)
        write_best(out_dir, best)

        if not ran_any:
            print("No new trials were run (all candidates already completed).")

        best_acc = float(best["metrics"].get("best_accuracy", -1.0))
        summary_rows.append(
            {
                "dataset": dataset,
                "display": DATASET_DISPLAY.get(dataset, dataset),
                "model": args.model,
                "method": args.method,
                "best_accuracy": best_acc,
                "best_trial_id": best["trial_id"],
            }
        )
        print(f"[DONE {dataset}] best_accuracy={best_acc:.4f} trial={best['trial_id']}")

    summary_fp = Path(args.out_dir) / args.method / args.model / "summary.json"
    summary_fp.parent.mkdir(parents=True, exist_ok=True)
    summary_fp.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("Dataset | Method | Model | Best Accuracy")
    print("---|---|---|---:")
    for row in summary_rows:
        print(
            f"{row['display']} | {row['method']} | {row['model'].upper()} | "
            f"{fmt_pct(row.get('best_accuracy'))}"
        )
    print(f"\nSaved summary: {summary_fp}")


if __name__ == "__main__":
    main()

