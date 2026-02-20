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

from wl_gcl.configs.factory import make_cfg
from wl_gcl.src.trainers.registry import get_trainer, available_trainers


# Repro
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Config utils
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


# Search space
def grid_product(space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(space.keys())
    for values in itertools.product(*[space[k] for k in keys]):
        yield dict(zip(keys, values))


def sample_random(space: Dict[str, List[Any]], rng: random.Random) -> Dict[str, Any]:
    return {k: rng.choice(v) for k, v in space.items()}


def search_space(trainer: str, dataset: str) -> Dict[str, List[Any]]:
    t = trainer.lower()

    common = {
        "lr": [1e-4, 3e-4, 1e-3],
        "hidden_dim": [128, 256, 512],
        "out_dim": [64, 128, 256],
        "dropout": [0.0, 0.1, 0.2, 0.5],
        "tau": [1.0, 2.0],
        "num_layers": [2, 3, 4],
        "heads": [2, 4, 8],
        "temperature": [0.1, 0.2, 0.5],
        "epochs": [200, 300, 400],
        "scheduler": [True],
        "weight_decay": [1e-6, 1e-5, 1e-4],
    }

    if t == "baseline":
        return {
            **common,
            "batch_size": [256, 512, 1024],
            "drop_edge_prob": [0.2, 0.4, 0.6],
            "feature_mask_prob": [0.0, 0.1, 0.2],
        }

    if t == "wl":
        return {
            **common,
            "theta": [0.3, 0.5, 0.7],
            "delta": [1, 2, 3],
            "batch_size": [256, 512, 1024],
            "num_negatives": [64, 128, 256, 512],
            "drop_edge_prob": [0.2, 0.4, 0.6],
            "feature_mask_prob": [0.0, 0.1, 0.2],
        }

    if t == "wl_hierarchy":
        return {
            **common,
            "num_negatives": [64, 128, 256, 512],
            "warmup": [20, 40, 80],
            "wl_max_iter": [5, 10],
            "lambda_hier": [1e-3, 1e-2, 1e-1],
            "lambda_nce": [0.5, 1.0, 2.0],
        }

    raise KeyError(f"Unknown trainer '{trainer}'")



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
                # ignore malformed/partial lines
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
        json.dumps(best, indent=2, sort_keys=True), encoding="utf-8"
    )

def sample_unique_random_candidates(
    space: Dict[str, List[Any]],
    rng: random.Random,
    make_cfg_fn,
    completed_ids: set[str],
    n_trials: int,
    max_attempts: int = 20_000,
) -> List[Dict[str, Any]]:
    """
    Randomly sample params until we get n_trials *new* trial_ids (not in completed_ids).
    This makes 'random search' actually progress across repeated runs with resume enabled.
    """
    out: List[Dict[str, Any]] = []
    attempts = 0

    # Reserve to avoid duplicates within the same run
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
            f"[WARN] Only sampled {len(out)}/{n_trials} new trials "
            f"after {attempts} attempts (space may be exhausted)."
        )

    return out



# Trial execution
def run_trial(trainer_name: str, cfg: Any, seed: int) -> Dict[str, Any]:
    set_seed(seed)

    trainer_fn = get_trainer(trainer_name)

    t0 = time.time()
    metrics = trainer_fn(cfg)
    dt = time.time() - t0

    return {
        "trial_id": stable_trial_id(cfg),
        "trainer": trainer_name,
        "seed": seed,
        "wall_time_sec": round(dt, 3),
        "cfg": cfg_to_dict(cfg),
        "metrics": metrics,
    }



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trainer", type=str, required=True, choices=available_trainers())
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model", type=str, default="gin")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--search", type=str, default="random", choices=["grid", "random"])
    p.add_argument("--n_trials", type=int, default=30, help="Used for random search.")
    p.add_argument("--max_trials", type=int, default=None, help="Hard cap for either mode.")
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--out_dir", type=str, default="runs/tune")
    p.add_argument("--epochs", type=int, default=None, help="Optional override for all trials.")
    args = p.parse_args()

    out_dir = Path(args.out_dir) / args.trainer / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    completed_ids, records = load_completed_and_records(out_dir)

    space = search_space(args.trainer, args.dataset)
    rng = random.Random(args.seed)

    def make_trial_cfg(params: Dict[str, Any]) -> Any:
        cfg = make_cfg(args.trainer, args.dataset)
        cfg = apply_params(cfg, {"model": args.model, "device": args.device})
        cfg = apply_params(cfg, params)
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

    if args.max_trials is not None:
        candidates = candidates[: args.max_trials]

    print(f"Existing trials in results.jsonl: {len(completed_ids)}")
    print(f"Planned candidates: {len(candidates)}")

    ran_any = False

    for i, params in enumerate(candidates, start=1):
        cfg = make_trial_cfg(params)
        tid = stable_trial_id(cfg)

        if tid in completed_ids:
            continue

        ran_any = True
        print(f"\n=== [{i}/{len(candidates)}] Trial {tid} ({args.trainer}/{args.dataset}) ===")
        print(json.dumps(params, indent=2, sort_keys=True))

        rec = run_trial(args.trainer, cfg, args.seed)

        append_result(out_dir, rec)

        # Update in-memory state from disk-compatible record
        records.append(rec)
        completed_ids.add(tid)

        best = select_best(records)
        write_best(out_dir, best)

        print(
            f"[BEST GLOBAL] best_accuracy={best['metrics'].get('best_accuracy', -1.0):.4f} "
            f"trial={best['trial_id']}"
        )

    if not records:
        raise RuntimeError("No results.jsonl found and no new trials executed.")

    best = select_best(records)
    write_best(out_dir, best)

    if not ran_any:
        print("\nNo new trials were run (all candidates already completed).")

    print("\nTrials Done!")
    print(f"Best best_accuracy: {best['metrics'].get('best_accuracy', -1.0):.4f}")
    print(f"Best trial_id: {best['trial_id']}")
    print(f"Saved: {out_dir / 'best.json'}")
    print(f"Results: {results_path(out_dir)}")


if __name__ == "__main__":
    main()
