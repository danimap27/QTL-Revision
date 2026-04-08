#!/bin/bash
# ==============================================================================
# Barren plateau analysis (CPU-only, no GPU needed)
# 5 qubits x 5 depths = 25 configs
# ==============================================================================
#SBATCH --job-name=qtl_bp
#SBATCH --partition=batch
#SBATCH --array=0-24
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/bp_%A_%a.out
#SBATCH --error=slurm_logs/bp_%A_%a.err

set -euo pipefail

module load Miniconda3
conda activate qtl_revision

cd $SLURM_SUBMIT_DIR

echo "BP Task ID: ${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | $(date)"

python barren_plateaus.py \
    --config config.yaml \
    --task-id $SLURM_ARRAY_TASK_ID

echo "Done: $(date)"
