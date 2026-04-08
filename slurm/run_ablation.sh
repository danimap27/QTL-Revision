#!/bin/bash
# ==============================================================================
# Ablation study - 400 tasks
# Systematic removal of quantum circuit components
# ==============================================================================
#SBATCH --job-name=qtl_ablation
#SBATCH --partition=gpu
#SBATCH --array=0-399%30
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/ablation_%A_%a.out
#SBATCH --error=slurm_logs/ablation_%A_%a.err

set -euo pipefail

# Load modules
module load Miniconda3

conda activate qtl_revision

cd $SLURM_SUBMIT_DIR

echo "Job array ID: ${SLURM_ARRAY_JOB_ID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time: $(date)"

python run_ablation_study.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --output-dir results/ablation

echo "End time: $(date)"
