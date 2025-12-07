# Graph Contrastive Learning with Weisfeiler-Lehman (WL-GCL)

This repository implements a Graph Self-Supervised Learning (SSL) framework that leverages the **Weisfeiler–Lehman (WL) Kernel** for structural mining and **Hyperbolic Embeddings** (WLHN) to capture hierarchical patterns.

The project reproduces and extends the WL-GCL pipeline, and compares it against standard baselines (GCN + SimCLR).

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



