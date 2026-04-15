#!/bin/bash
#SBATCH --job-name=wl_hier_photo
#SBATCH --output=logs/wl_tune_%j.out
#SBATCH --error=logs/wl_tune_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00

# 1. Environnement
module load anaconda3/2023.09-0/none-none
source /gpfs/softs/spack_1.0.2/opt/spack/linux-cascadelake/anaconda3-2023.09-0-v2nbar7o4nyuwoknqbnybvxufqw3rrnk/etc/profile.d/conda.sh
conda activate /gpfs/workdir/yartaouifa/.conda/envs/env_gssl

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=${PYTHONPATH}:${SLURM_SUBMIT_DIR}

# 2. Liste des datasets à processer
DATASETS=("amazon-photo"  )

echo "Démarrage du tuning WL-Hierarchy à $(date)"

# 3. Boucle sur chaque dataset
for DS in "${DATASETS[@]}"
do
    echo "------------------------------------------------"
    echo "Début du tuning pour : $DS"
    echo "------------------------------------------------"
    
    python -u wl_gcl/src/utils/tune.py \
        --trainer wl_hierarchy \
        --dataset $DS \
        --search random \
        --n_trials 40 \
        --device cuda \
        --epochs 400
        
    echo "Fin du tuning pour $DS à $(date)"
done

echo "Tuning global terminé !"