#!/bin/bash
# ==============================================================================
# SPSA optimizer control experiment - 40 tasks
# 2 datasets x 4 backbones x 5 seeds = 40
# ==============================================================================
#SBATCH --job-name=qtl_spsa
#SBATCH --partition=gpu
#SBATCH --array=0-39%10
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/spsa_%A_%a.out
#SBATCH --error=slurm_logs/spsa_%A_%a.err

set -euo pipefail

# Load modules
module load Miniconda3

conda activate qtl_revision

cd $SLURM_SUBMIT_DIR

# Decode task ID into experiment parameters
# Layout: 2 datasets x 4 backbones x 5 seeds = 40 tasks
TASK_ID=$SLURM_ARRAY_TASK_ID

DATASETS=("CIFAR-10" "EuroSAT")
BACKBONES=("resnet18" "resnet34" "resnet50" "vgg16")
NUM_SEEDS=5

SEED_IDX=$((TASK_ID % NUM_SEEDS))
REMAINING=$((TASK_ID / NUM_SEEDS))
BACKBONE_IDX=$((REMAINING % 4))
DATASET_IDX=$((REMAINING / 4))

DATASET=${DATASETS[$DATASET_IDX]}
BACKBONE=${BACKBONES[$BACKBONE_IDX]}
SEED=$((SEED_IDX + 1))

echo "Job array ID: ${SLURM_ARRAY_JOB_ID}, Task ID: ${TASK_ID}"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Configuration: dataset=${DATASET}, backbone=${BACKBONE}, seed=${SEED}"
echo "Start time: $(date)"

python train_cq_pennylane_spsa.py \
    --dataset "$DATASET" \
    --backbone "$BACKBONE" \
    --seed $SEED \
    --output-dir results/optimizer_control

echo "End time: $(date)"
