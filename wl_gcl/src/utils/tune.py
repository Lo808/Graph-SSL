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
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from wl_gcl.configs.factory import make_cfg
from wl_gcl.src.trainers.registry import get_trainer, available_trainers


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
    """
    Safe patch: only apply keys that are actual dataclass fields.
    This keeps the tuner agnostic across different config classes.
    """
    fields = {f.name for f in dataclasses.fields(cfg)}
    filtered = {k: v for k, v in params.items() if k in fields}
    return replace(cfg, **filtered) if filtered else cfg


def grid_product(space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(space.keys())
    for values in itertools.product(*[space[k] for k in keys]):
        yield dict(zip(keys, values))


def sample_random(space: Dict[str, List[Any]], rng: random.Random) -> Dict[str, Any]:
    return {k: rng.choice(v) for k, v in space.items()}



def load_completed(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    trials_dir = out_dir / "trials"
    if not trials_dir.exists():
        return done

    for d in trials_dir.iterdir():
        if not d.is_dir():
            continue
        fp = d / "result.json"
        if fp.exists():
            try:
                done[d.name] = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                # ignore partial/corrupt writes
                pass
    return done



def search_space(trainer: str, dataset: str) -> Dict[str, List[Any]]:
    """
    Trainer-aware search space. The tuner remains trainer-agnostic.
    """
    t = trainer.lower()
    ds = dataset.lower()

    # Common knobs that exist in all configs
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
    }

    common["weight_decay"] = [1e-6, 1e-5, 1e-4]

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


def objective(m: Dict[str, Any]) -> float:
    return float(m.get("best_accuracy", -1.0))


def select_best(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(records, key=lambda r: objective(r["metrics"]))



def run_trial(
    trainer_name: str,
    cfg: Any,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    set_seed(seed)

    trainer_fn = get_trainer(trainer_name)
    trial_id = stable_trial_id(cfg)
    trial_dir = out_dir / "trials" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    metrics = trainer_fn(cfg)
    dt = time.time() - t0

    record = {
        "trial_id": trial_id,
        "trainer": trainer_name,
        "seed": seed,
        "wall_time_sec": round(dt, 3),
        "cfg": cfg_to_dict(cfg),
        "metrics": metrics,
    }

    (trial_dir / "result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record




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

    completed = load_completed(out_dir)
    records: List[Dict[str, Any]] = list(completed.values())

    space = search_space(args.trainer, args.dataset)

    rng = random.Random(args.seed)

    def make_trial_cfg(params: Dict[str, Any]) -> Any:
        cfg = make_cfg(args.trainer, args.dataset)
        cfg = apply_params(cfg, {"model": args.model, "device": args.device})
        cfg = apply_params(cfg, params)
        if args.epochs is not None:
            cfg = apply_params(cfg, {"epochs": args.epochs})
        return cfg

    # Build parameter candidates
    candidates: List[Dict[str, Any]] = []
    if args.search == "grid":
        candidates = list(grid_product(space))
    else:
        candidates = [sample_random(space, rng) for _ in range(args.n_trials)]
    
    print(f"Number of Candidates : {len(candidates)}")

    if args.max_trials is not None:
        candidates = candidates[: args.max_trials]


    for i, params in enumerate(candidates, start=1):
        cfg = make_trial_cfg(params)
        tid = stable_trial_id(cfg)

        if tid in completed:
            continue

        print(f"\n=== [{i}/{len(candidates)}] Trial {tid} ({args.trainer}/{args.dataset}) ===")
        print(json.dumps(params, indent=2, sort_keys=True))

        rec = run_trial(
            trainer_name=args.trainer,
            cfg=cfg,
            seed=args.seed,
            out_dir=out_dir,
        )
        records.append(rec)

        best = select_best(records)
        print(
            f"[BEST SO FAR] best_accuracy={best['metrics'].get('best_accuracy', -1.0):.4f} "
            f"trial={best['trial_id']}"
        )

        # write checkpoint of best after every trial
        (out_dir / "best.json").write_text(
            json.dumps(best, indent=2, sort_keys=True), encoding="utf-8"
        )

    if not records:
        raise RuntimeError("No completed trials found and no new trials executed.")

    best = select_best(records)
    (out_dir / "best.json").write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")

    print("\nTrials Done!")
    print(f"Best best_accuracy: {best['metrics'].get('best_accuracy', -1.0):.4f}")
    print(f"Best trial_id: {best['trial_id']}")
    print(f"Saved: {out_dir / 'best.json'}")


# if __name__ == "__main__":
main()

