#!/bin/bash
# ==============================================================================
# Noise decomposition - 25 tasks
# Shot noise, gate noise, and decoherence analysis
# ==============================================================================
#SBATCH --job-name=qtl_noise
#SBATCH --partition=gpu
#SBATCH --array=0-24%10
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/noise_%A_%a.out
#SBATCH --error=slurm_logs/noise_%A_%a.err

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

python run_noise_decomposition.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --output-dir results/noise

echo "End time: $(date)"
