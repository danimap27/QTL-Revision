#!/bin/bash
# ==============================================================================
# Main revision experiments - 3120 tasks
# 3 classical (4 backbones x 4 datasets x 5 seeds = 240)
# + 4 quantum (4 backbones x 4 datasets x 3 qubits x 3 depths x 5 seeds = 2880)
# Total: 3120 unique runs
# ==============================================================================
#SBATCH --job-name=qtl_revision
#SBATCH --partition=gpu
#SBATCH --array=0-3119%80
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/seeds_%A_%a.out
#SBATCH --error=slurm_logs/seeds_%A_%a.err

set -euo pipefail

# Load modules
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
conda activate qtl_revision

cd $SLURM_SUBMIT_DIR

echo "Job array ID: ${SLURM_ARRAY_JOB_ID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time: $(date)"

python run_revision_experiments.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --output-dir results/seeds \
    --epochs 10 \
    --batch-size 16

echo "End time: $(date)"
