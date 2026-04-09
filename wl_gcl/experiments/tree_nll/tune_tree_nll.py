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

from wl_gcl.experiments.tree_nll.train_tree_nll import (
    TreeNLLConfig,
    make_tree_nll_cfg,
    train_tree_nll,
)


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


def search_space(model: str) -> Dict[str, List[Any]]:
    common = {
        "lr": [1e-4, 3e-4, 1e-3],
        "hidden_dim": [128, 256, 512],
        "out_dim": [64, 128, 256],
        "dropout": [0.0, 0.1, 0.2],
        "num_layers": [2, 3, 4],
        "tau": [1.0, 2.0],
        "weight_decay": [1e-6, 1e-5, 1e-4],
        "temperature": [0.1, 0.2, 0.5],
        "scheduler": [True],
        "warmup": [20, 40, 80],
        "wl_max_iter": [5, 10],
        "lambda_hier": [1e-3, 1e-2, 1e-1],
        "lambda_nce": [0.1, 0.5, 1.0],
    }

    if model.lower() == "gat":
        return {
            **common,
            "heads": [2, 4, 8],
        }

    return common


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


def load_existing_wl_gin_best(dataset: str) -> float | None:
    fp = Path("runs") / "tune" / "wl" / dataset / "best.json"
    if not fp.exists():
        return None

    try:
        rec = json.loads(fp.read_text(encoding="utf-8"))
        if "metrics" in rec and "best_accuracy" in rec["metrics"]:
            return float(rec["metrics"]["best_accuracy"])
    except Exception:
        return None

    return None


def load_prior_wl_hierarchy_cfg(dataset: str) -> Dict[str, Any] | None:
    fp = Path("runs") / "tune" / "wl_hierarchy" / dataset / "best.json"
    if not fp.exists():
        return None

    try:
        rec = json.loads(fp.read_text(encoding="utf-8"))
        prior_cfg = rec.get("cfg", {})
        if not isinstance(prior_cfg, dict):
            return None
        valid_fields = {f.name for f in dataclasses.fields(TreeNLLConfig)}
        return {k: v for k, v in prior_cfg.items() if k in valid_fields}
    except Exception:
        return None


def run_trial(
    cfg: TreeNLLConfig,
    trial_seed: int,
) -> Dict[str, Any]:
    set_seed(trial_seed)
    t0 = time.time()
    metrics = train_tree_nll(cfg)
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


def dataset_report_row(dataset: str, tree_best_acc: float | None) -> Dict[str, Any]:
    wl_gin_acc = load_existing_wl_gin_best(dataset)
    return {
        "dataset": dataset,
        "display": DATASET_DISPLAY.get(dataset, dataset),
        "wl_gin_best_accuracy": wl_gin_acc,
        "tree_nll_best_accuracy": tree_best_acc,
    }


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "N/A"
    return f"{100.0 * x:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune WL-Hierarchy+Contrastive with Tree-NLL supervision (standalone)."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated datasets or 'all'.",
    )
    parser.add_argument("--model", type=str, default="gin", choices=["gin", "gcn", "gat", "wlhn"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--search", type=str, default="random", choices=["random", "grid"])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--max_trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for all trials.")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="runs/tune_tree_nll")
    parser.add_argument("--save_best_encoder", action="store_true")
    parser.add_argument(
        "--skip_prior_best",
        action="store_true",
        help="Do not include previous wl_hierarchy best config as a warm-start candidate.",
    )
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
        print(f"[DATASET {dsi + 1}/{len(datasets)}] {dataset}")
        print("=" * 80)

        out_dir = Path(args.out_dir) / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        completed_ids, records = load_completed_and_records(out_dir)

        space = search_space(args.model)
        rng = random.Random(args.seed + dsi)

        def make_trial_cfg(params: Dict[str, Any]) -> TreeNLLConfig:
            cfg = make_tree_nll_cfg(dataset)
            cfg = apply_params(cfg, params)
            cfg = apply_params(
                cfg,
                {
                    "model": args.model,
                    "device": args.device,
                    "log_interval": args.log_interval,
                    "save_best": args.save_best_encoder,
                    "output_dir": str(Path(args.out_dir) / "checkpoints"),
                },
            )
            if args.epochs is not None:
                cfg = apply_params(cfg, {"epochs": args.epochs})
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

        if not args.skip_prior_best:
            prior = load_prior_wl_hierarchy_cfg(dataset)
            if prior:
                prior_cfg = make_trial_cfg(prior)
                prior_tid = stable_trial_id(prior_cfg)
                candidate_ids = {stable_trial_id(make_trial_cfg(c)) for c in candidates}
                if prior_tid not in completed_ids and prior_tid not in candidate_ids:
                    candidates.insert(0, prior)

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
            summary_rows.append(dataset_report_row(dataset, None))
            continue

        best = select_best(records)
        write_best(out_dir, best)

        if not ran_any:
            print("No new trials were run (all candidates already completed).")

        best_acc = float(best["metrics"].get("best_accuracy", -1.0))
        summary_rows.append(dataset_report_row(dataset, best_acc))

        print(f"[DONE {dataset}] best_accuracy={best_acc:.4f} trial={best['trial_id']}")

    summary_fp = Path(args.out_dir) / "summary_tree_nll.json"
    summary_fp.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Comparison Table")
    print("=" * 80)
    print("Dataset | WL-GIN | WL-Hierarchical + Contrastive (GIN, Tree-NLL)")
    print("---|---:|---:")
    for row in summary_rows:
        print(
            f"{row['display']} | {fmt_pct(row['wl_gin_best_accuracy'])} | "
            f"{fmt_pct(row['tree_nll_best_accuracy'])}"
        )

    print(f"\nSaved summary: {summary_fp}")


if __name__ == "__main__":
    main()
