# WL-DINO/BYOL/BGRL Daily Report (2026-04-09)

## 1) Context and Goal

This report summarizes all graph SSL experiments run today on **Cora** with the objective of improving unsupervised node representation quality.

Main goals explored:

1. Strengthen DINO-style student/teacher training on graphs.
2. Use WL structure effectively (without over-constraining learning).
3. Test alternatives (WL-align, dual-view mined pairing, BYOL, BGRL).
4. Track whether any method can beat the current best Cora result.

---

## 2) Environment

- Repo: `Graph-SSL`
- Date: `2026-04-09`
- Hardware: CUDA GPU (base environment)
- Dataset focus: `Cora` only
- Backbone in runs below: `GIN` (unless noted)
- Evaluation: linear probing / node classification accuracy (same pipeline used in trainer)

---

## 3) What Was Implemented Today

## 3.1 DINO Stabilization Features

Added to WL-DINO trainer/config:

- Teacher centering
- Teacher temperature warmup schedule
- Prototype distillation option
- Optional supervised anchor term (`lambda_sup`)

Primary files:

- `wl_gcl/configs/wl_dino.py`
- `wl_gcl/src/trainers/train_wl_dino.py`

## 3.2 WL Loss Variants

Implemented both:

- `wl_loss_type=kl` (existing baseline style)
- `wl_loss_type=align` (WL-weighted geometric alignment)
- Optional weak repulsion for align mode

## 3.3 Adaptive WL Balancing

Implemented EMA-based adaptive WL scaling:

- scale = EMA(distill) / EMA(wl)
- with clamp min/max and logging (`WL_scale`, `WL_term`)

Result: this specific strategy failed badly (details in results section).

## 3.4 Dual-View Mined Pairing for DINO

Added mode to replace same-node teacher target with a mined partner:

- partner chosen from dual-view intersection (WL + feature space)
- logs coverage per epoch (`PairCov`)
- used with `objective=dino` and no WL loss term

CLI flags added:

- `--use_dual_view_miner_pairs`
- `--miner_theta`
- `--miner_delta`
- `--miner_refresh_epochs`

## 3.5 BYOL Objective

Added `objective=byol`:

- online encoder + projection + predictor
- EMA teacher encoder + projection
- symmetric regression loss over two augmented views

## 3.6 BGRL Objective

Added `objective=bgrl`:

- online encoder + predictor
- EMA teacher encoder
- symmetric regression on encoder embeddings

---

## 4) Experiment Results

All results are from terminal runs today on Cora + CUDA.

### 4.1 Best Scores by Method Family

| Method Family | Key Setup | Best Accuracy |
| --- | --- | --- |
| DINO + WL(KL) | augmentations on, prototype distill | **0.5730** |
| DINO + WL-align | augmentations on, lambda sweep | 0.4350 |
| DINO + dual-view mined pairs (no WL loss) | objective=dino, mined partner teacher | 0.5010 (100 ep) / 0.4830 (200 ep) |
| BYOL | plain | 0.4620 |
| BYOL + dual-view mined pairs | theta=0.6, delta=2 | 0.4940 |
| BGRL | plain | 0.4610 |
| BGRL + dual-view mined pairs | theta=0.6, delta=2 | 0.4580 |
| Hybrid supervised anchor (for reference) | DINO+WL(KL) + lambda_sup=1.0 | 0.6240 |

Current unsupervised winner on Cora remains:

- **DINO + WL(KL)** (~57.3%)

---

### 4.2 DINO + WL(KL) Baselines (Cora)

Observed across runs today:

- No augmentation, 200 epochs, lambda_wl=100: best around `0.500-0.503`
- With augmentation, 200 epochs, lambda_wl=100: best around `0.549`
- Matched rerun, 100 epochs, lambda_wl=100: `0.546`
- Later rerun, same family: **`0.573` best**

Takeaway:

- Augmentations materially help compared to no-augmentation setup.
- Run-to-run variance is visible (single-seed runs).

---

### 4.3 WL-align Ablation (replace KL with alignment)

From `runs/wl_dino_align_ablation/cora_results.json`:

| Variant | Best Accuracy |
| --- | --- |
| baseline_kl (lambda=100) | 0.5660 |
| align_l0.5 | 0.4310 |
| align_l1.0 | 0.4340 |
| align_l2.0 | 0.4350 |

Additional direct epoch-wise run:

- align + lambda=1.0 (100 epochs): best `0.4200`

Optional repulsion test:

- align + repulsion(beta=0.1): ~`0.4200` (not helpful)

Takeaway:

- WL-align as implemented did not beat KL baseline.

---

### 4.4 Adaptive WL Balance Test

Tested adaptive scale on align:

- `use_adaptive_wl_balance=True`
- scale quickly saturated at max clamp (`10000`)
- accuracy collapsed to `0.2940`

Observed behavior:

- `WL_align` rapidly shrank toward ~0
- ratio-based scale exploded
- optimization degraded

Takeaway:

- This adaptive recipe is not stable in current form and should stay off.

---

### 4.5 DINO with Dual-View Mined Teacher Pairs (No WL Loss)

Settings:

- `objective=dino`
- dual-view mining on teacher target partner
- no WL-KL term

Runs:

| Miner Setup | Epochs | PairCov | Best Accuracy |
| --- | --- | --- | --- |
| theta=0.8, delta=2 | 100 | ~0.513 | 0.4910 |
| theta=0.6, delta=2 | 100 | ~0.513 | **0.5010** |
| theta=0.6, delta=3 | 100 | ~0.995 | 0.4620 |
| theta=0.6, delta=2 | 200 | ~0.513 | 0.4830 |

Takeaway:

- Mining alone did not beat DINO+WL(KL).
- Higher pair coverage did not imply better performance.

---

### 4.6 BYOL and BGRL

BYOL:

- plain (200): `0.4620`
- + miner pairs (200): `0.4940`

BGRL:

- plain (200): `0.4610`
- + miner pairs (200): `0.4580`

Takeaway:

- In today’s settings, neither BYOL nor BGRL beat DINO+WL(KL).

---

## 5) Core Diagnostic Findings

1. Loss-scale mismatch is real.
2. In WL-align runs, distill dominated while WL term became numerically tiny.
3. Naive ratio-based adaptive balancing over-corrected and destabilized training.
4. DINO+WL(KL) remains strongest among tested unsupervised methods.
5. Current experiments are mostly single-run/single-seed; variance exists.

---

## 6) Important Notes About Comparability

- Results above are mostly single-run measurements, not 5-seed averages.
- Some values differ between repeated runs with nominally same config.
- For strict comparison against external methods (~79%), protocol alignment is critical:
  - exact split
  - identical probe settings
  - same augmentations
  - same seeds and averaging

---

## 7) Recommended Next Phase (after today)

1. Keep DINO+WL(KL) as current unsupervised baseline.
2. Run controlled hyperparameter search around that baseline:
   - `m`, `tau_t`, `tau_s`, `num_prototypes`, augmentation strengths, `lambda_wl`.
3. Run multi-seed (`5`) for robust ranking.
4. Only keep methods that consistently beat the KL baseline under matched protocol.

---

## 8) Key Artifacts

- WL-align ablation summary:
  - `runs/wl_dino_align_ablation/cora_results.json`

---

## 9) One-Line Summary

Today’s strongest unsupervised result on Cora remained **DINO + WL(KL)** (~**57.3%**); WL-align, dual-view-only DINO, BYOL, and BGRL did not surpass it in the tested settings.
