#!/bin/bash
# ==============================================================================
# Gradient analysis - 20 tasks (CPU only)
# Barren plateau and gradient variance analysis
# ==============================================================================
#SBATCH --job-name=qtl_gradient
#SBATCH --partition=batch
#SBATCH --array=0-19
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/gradient_%A_%a.out
#SBATCH --error=slurm_logs/gradient_%A_%a.err

set -euo pipefail

# Load modules (CPU only - no CUDA needed)
module load Miniconda3
conda activate qtl_revision

cd $SLURM_SUBMIT_DIR

echo "Job array ID: ${SLURM_ARRAY_JOB_ID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

python run_gradient_analysis.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --output-dir results/gradient

echo "End time: $(date)"
