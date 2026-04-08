#!/bin/bash
# ==============================================================================
# Master submission script for QTL revision experiments on Hercules
# Usage: bash slurm/submit_all.sh
# ==============================================================================

set -euo pipefail

echo "============================================"
echo "  QTL Revision - Submitting All Jobs"
echo "  Hercules Supercomputer"
echo "============================================"
echo ""

# Create log directory
mkdir -p slurm_logs

echo "Submitting main experiments (3120 jobs: 240 classical + 2880 quantum)..."
JOB1=$(sbatch --parsable slurm/run_main_seeds.sh)
echo "  Job ID: $JOB1"

echo "Submitting ablation study (400 jobs)..."
JOB2=$(sbatch --parsable slurm/run_ablation.sh)
echo "  Job ID: $JOB2"

echo "Submitting gradient analysis (20 jobs)..."
JOB3=$(sbatch --parsable slurm/run_gradient.sh)
echo "  Job ID: $JOB3"

echo "Submitting noise decomposition (25 jobs)..."
JOB4=$(sbatch --parsable slurm/run_noise.sh)
echo "  Job ID: $JOB4"

echo "Submitting optimizer control (40 jobs)..."
JOB5=$(sbatch --parsable slurm/run_optimizer_control.sh)
echo "  Job ID: $JOB5"

echo "Submitting aggregation (depends on all above)..."
JOB6=$(sbatch --parsable --dependency=afterany:${JOB1}:${JOB2}:${JOB3}:${JOB4}:${JOB5} slurm/aggregate.sh)
echo "  Job ID: $JOB6"

echo ""
echo "============================================"
echo "  All jobs submitted!"
echo "============================================"
echo ""
echo "Total tasks: 3120 + 400 + 20 + 25 + 40 + 1 = 3606"
echo ""
echo "Monitor with:    squeue -u $USER"
echo "Job details:     sacct -j $JOB1 --format=JobID,State,Elapsed,MaxRSS"
echo "Cancel all with: scancel $JOB1 $JOB2 $JOB3 $JOB4 $JOB5 $JOB6"
