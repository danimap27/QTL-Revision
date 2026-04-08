#!/bin/bash
# ==============================================================================
# Aggregate results from all experiments
# Submit with dependency on all other jobs:
#   sbatch --dependency=afterok:$JOB1:$JOB2:$JOB3:$JOB4:$JOB5 slurm/aggregate.sh
# ==============================================================================
#SBATCH --job-name=qtl_aggregate
#SBATCH --partition=batch
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/aggregate_%j.out
#SBATCH --error=slurm_logs/aggregate_%j.err

set -euo pipefail

# Load modules (CPU only)
module load Python/3.10.8-GCCcore-12.2.0
conda activate qtl_revision

cd $SLURM_SUBMIT_DIR

echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Aggregating results from all experiments..."

python analyze_results.py \
    --results-dir results \
    --output-dir results/aggregated \
    --figures-dir results/figures

echo "Aggregation complete!"
echo "End time: $(date)"
