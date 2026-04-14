# Graph Contrastive Learning with Weisfeiler-Lehman (WL-GCL)

This repository implements a Graph Self-Supervised Learning (SSL) framework that leverages the **Weisfeiler–Lehman (WL) Kernel** for structural mining and **Hyperbolic Embeddings** (WLHN) to capture hierarchical patterns.

The project reproduces and extends the WL-GCL pipeline, and compares it against standard baselines (GCN + SimCLR).

---

## Recommended Method (Current Main Objective)

The current flagship method is:

- `objective=bgrl_wl_naive_cls`

It is a fully WL-based method that combines:

1. BGRL bootstrap loss with WL-based positive-pair sampling (`bgrl_wl_naive` branch).
2. Multi-level WL pseudo-label classification loss (`bgrl_wl_cls` branch).

Formally:

\[
\mathcal{L}
=
\mathcal{L}_{\text{BGRL-WL-pairs}}
 \lambda_{wl}\sum_{t \in \mathcal{T}} \alpha_t \,\mathcal{L}^{cls}_{wl,t}
\]

where \(\mathcal{L}^{cls}_{wl,t}\) is cross-entropy on WL colors at level \(t\).

Important defaults:

- Augmentations are **off by default** in `train_wl_dino.py`.
- WL pair sampling mode is controlled by `--wl_naive_pair_sampling`.
- WL classification weighting is controlled by `--wl_cls_alpha_scheme`.

---

## Project Structure

The codebase is organized as a modular Python package:

```text
├── main.py                     # Entry point for training and evaluation
├── wl_gcl/  
|   ├── configs/                # Config files
|   |   ├── baseline.py/    
|   |   ├── wl.py/           
│   ├── src/
│   │   ├── augmentations/      # Graph augmentations (Edge Drop, Feature Masking)
│   │   ├── contrastive/        # Contrastive losses (MoCHi, InfoNCE) + DualViewMiner
│   │   ├── data_loader/        # Dataset loaders (Cora, Citeseer, heterophilic datasets, ...)
│   │   ├── models/             # Model architectures (GIN, GAT, WLHN, GCN)
│   │   ├── trainers/           # Training loops (WL-Advanced vs Baseline)
│   │   └── utils/              # WL Hierarchy Engine and core utilities
├── baselines/                  # External baseline implementations
├── tests/                      # Unit tests
└── requirements.txt            # Dependencies
```

---

## Usage

Experiments are launched with the root-level `main.py`.

### 1. Running the WL-GCL Framework

This executes the full pipeline:  
**WL Structural Mining → Dual Augmentations → MoCHi Loss**.

```bash
# Default GIN encoder on Cora
python main.py --method wl --dataset cora --model gin

# Hyperbolic WLHN encoder
python main.py --method wl --dataset cora --model wlhn

# GAT encoder
python main.py --method wl --dataset cora --model gat
```

### 2. Running the Baseline

This uses **GCN + SimCLR** for contrastive pretraining.

```bash
python main.py --method baseline --dataset cora
```

---

## WL-DINO/BGRL Trainer Entry Point

For the latest SSL objectives (DINO/BYOL/BGRL/WL variants), use:

```bash
python3 -u -m wl_gcl.src.trainers.train_wl_dino ...
```

### Run the Main Method (Augmentation-Free)

```bash
python3 -u -m wl_gcl.src.trainers.train_wl_dino \
  --dataset cora \
  --model gcn \
  --objective bgrl_wl_naive_cls \
  --epochs 500 \
  --device cuda \
  --use_max_wl_depth \
  --wl_naive_pair_sampling wl_distance \
  --wl_naive_distance_beta 1.0 \
  --lambda_wl 0.5 \
  --wl_cls_levels all \
  --wl_cls_alpha_scheme deeper_more
```

Notes:

- Do **not** pass `--use_augmentations` if you want augmentation-free training.
- For uniform WL pair sampling, set `--wl_naive_pair_sampling uniform`.

---

## Hyperparameter Tuning (Ready for Coworker)

Yes, hyperparameter tuning for `bgrl_wl_naive_cls` is already available in:

```text
wl_gcl/experiments/wl_dino/tune_wl_dino.py
```

Method key:

- `--method bgrl_wl_naive_cls`

You can force all results to be written in a dedicated folder via `--out_dir`.

### Single model example

```bash
python3 -u -m wl_gcl.experiments.wl_dino.tune_wl_dino \
  --datasets all \
  --model gcn \
  --method bgrl_wl_naive_cls \
  --device cuda \
  --use_max_wl_depth \
  --search random \
  --n_trials 60 \
  --epochs 500 \
  --out_dir runs/tune_bgrl_wl_naive_cls_2026_04_14
```

### All encoders in one root output folder

```bash
for m in gin gcn gat wlhn; do
  python3 -u -m wl_gcl.experiments.wl_dino.tune_wl_dino \
    --datasets all \
    --model "$m" \
    --method bgrl_wl_naive_cls \
    --device cuda \
    --use_max_wl_depth \
    --search random \
    --n_trials 60 \
    --epochs 500 \
    --out_dir runs/tune_bgrl_wl_naive_cls_2026_04_14
done
```

### One-command script for coworker

```bash
bash scripts/tune_bgrl_wl_naive_cls_all.sh runs/tune_bgrl_wl_naive_cls_2026_04_14
```

Optional env overrides:

```bash
DATASETS=all MODELS="gin gcn gat wlhn" N_TRIALS=80 EPOCHS=500 DEVICE=cuda \
bash scripts/tune_bgrl_wl_naive_cls_all.sh runs/tune_bgrl_wl_naive_cls_2026_04_14
```

Per-model and per-dataset results are stored under:

```text
<out_dir>/bgrl_wl_naive_cls/<model>/<dataset>/
```

with `results.jsonl`, `best.json`, and a per-model `summary.json`.

---

## Arguments Reference

| Argument | Default | Description |
|---------|---------|-------------|
| `--method` | `wl` | Choose framework: `wl` (ours) or `baseline` (standard SimCLR framework). |
| `--dataset` | `cora` | Supported datasets: `cora`, `citeseer`, `pubmed`, `amazon-photo`, `actor`, `squirrel`, `texas`, ... |
| `--model` | `gin` | Encoder choices: `gin`, `gcn`, `gat`, `wlhn`. |
| `--epochs` | `200` | Number of training epochs. |
| `--lr` | `0.001` | Learning rate (WL mode overrides this with dataset-specific configs). |
| `--device` | `cuda` | Computation device (`cuda` or `cpu`). |

---

## Models Implemented

1. **GIN (Graph Isomorphism Network):** Strong structural baseline.
2. **GAT (Graph Attention Network):** Attention-weighted neighbor aggregation.
3. **WLHN (Hyperbolic Weisfeiler–Lehman Network):** Embeds representations in the Poincaré ball to reflect the hierarchical structure extracted by the WL kernel.
4. **GCN (Baseline):** Graph Convolutional Network

---

## Configuration

For the **WL-GCL framework**, hyperparameters such as:

- temperature `tau`
- number of negative samples
- learning rate
- augmentations strengths  
are tuned separately **for each dataset**.

You can modify these settings in:

```
wl_gcl/configs/wl.py
```
.
