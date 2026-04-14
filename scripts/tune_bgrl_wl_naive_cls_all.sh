#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/tune_bgrl_wl_naive_cls_all.sh [OUT_DIR]
#
# Example:
#   bash scripts/tune_bgrl_wl_naive_cls_all.sh runs/tune_bgrl_wl_naive_cls_2026_04_14
#
# Environment overrides:
#   DATASETS=all
#   MODELS="gin gcn gat wlhn"
#   N_TRIALS=60
#   EPOCHS=500
#   DEVICE=cuda
#   SEED=7

OUT_DIR="${1:-runs/tune_bgrl_wl_naive_cls_$(date +%Y%m%d_%H%M%S)}"
DATASETS="${DATASETS:-all}"
MODELS="${MODELS:-gin gcn gat wlhn}"
N_TRIALS="${N_TRIALS:-60}"
EPOCHS="${EPOCHS:-500}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-7}"

echo "[BGRL-WL-NAIVE-CLS] Output folder: ${OUT_DIR}"
mkdir -p "${OUT_DIR}"

for MODEL in ${MODELS}; do
  echo "============================================================"
  echo "[BGRL-WL-NAIVE-CLS] model=${MODEL} datasets=${DATASETS}"
  echo "============================================================"

  python3 -u -m wl_gcl.experiments.wl_dino.tune_wl_dino \
    --datasets "${DATASETS}" \
    --model "${MODEL}" \
    --method bgrl_wl_naive_cls \
    --device "${DEVICE}" \
    --use_max_wl_depth \
    --search random \
    --n_trials "${N_TRIALS}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --out_dir "${OUT_DIR}"
done

echo "[BGRL-WL-NAIVE-CLS] Done. Results stored in: ${OUT_DIR}"
