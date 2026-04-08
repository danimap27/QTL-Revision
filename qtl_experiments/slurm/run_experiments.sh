#!/bin/bash
# ==============================================================================
# SLURM array job for QTL experiments
# Usage: sbatch slurm/run_experiments.sh
# ==============================================================================
#SBATCH --job-name=qtl_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err

set -euo pipefail

module load Miniconda3
conda activate qtl_revision

cd $SLURM_SUBMIT_DIR

echo "Task ID: ${SLURM_ARRAY_TASK_ID} | Node: $(hostname) | $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python runner.py \
    --config config.yaml \
    --task-id $SLURM_ARRAY_TASK_ID

echo "Done: $(date)"
