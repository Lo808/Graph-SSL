#!/bin/bash
#SBATCH --job-name=bgrl_full_tuning
#SBATCH --output=logs/full_tune_%j.out
#SBATCH --error=logs/full_tune_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00        # On demande le maximum (1 jour)

# 1. Chargement de l'environnement avec chemins absolus
module load anaconda3/2023.09-0/none-none
source /gpfs/softs/spack_1.0.2/opt/spack/linux-cascadelake/anaconda3-2023.09-0-v2nbar7o4nyuwoknqbnybvxufqw3rrnk/etc/profile.d/conda.sh
conda activate /gpfs/workdir/yartaouifa/.conda/envs/env_gssl

# 2. Configuration
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=${PYTHONPATH}:${SLURM_SUBMIT_DIR}
export MPLCONFIGDIR=/tmp/mplconfig

echo "--- DÉBUT DU TUNING GLOBAL ---"
date

# 3. Lancement du script wrapper
# On utilise les variables d'environnement demandées
DATASETS=all \
MODELS="gin gcn gat wlhn" \
N_TRIALS=60 \
EPOCHS=500 \
DEVICE=cuda \
bash scripts/tune_bgrl_wl_naive_cls_all.sh runs/tune_bgrl_wl_naive_cls_full_2026_04_15

echo "--- RECHERCHE TERMINÉE ---"
date