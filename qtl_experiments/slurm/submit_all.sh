#!/bin/bash
# ==============================================================================
# Master submission script for Hercules
# Submits main experiments + all extensions
# Usage: bash slurm/submit_all.sh
# ==============================================================================

set -euo pipefail

echo "============================================"
echo "  QTL Revision - Submitting All Jobs"
echo "============================================"

mkdir -p slurm_logs

# Count total runs
TOTAL=$(python runner.py --config config.yaml --count 2>&1 | tail -1 | awk '{print $NF}')
echo "Total GPU runs: $TOTAL"
LAST_IDX=$((TOTAL - 1))

echo ""
echo "Submitting main + extension experiments ($TOTAL tasks)..."
JOB1=$(sbatch --parsable --array=0-${LAST_IDX}%80 slurm/run_experiments.sh)
echo "  Job ID: $JOB1 (array 0-$LAST_IDX)"

echo "Submitting barren plateau analysis (25 configs, CPU-only)..."
JOB2=$(sbatch --parsable slurm/run_barren_plateaus.sh)
echo "  Job ID: $JOB2"

echo ""
echo "============================================"
echo "  All jobs submitted!"
echo "============================================"
echo ""
echo "Monitor:   squeue -u $USER"
echo "Details:   sacct -j $JOB1 --format=JobID,State,Elapsed,MaxRSS"
echo "Cancel:    scancel $JOB1 $JOB2"
