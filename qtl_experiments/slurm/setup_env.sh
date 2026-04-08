#!/bin/bash
# ==============================================================================
# Setup environment on Hercules (CICA)
# Run ONCE from an interactive node:
#   salloc --mem=16G -c 4 -t 06:00:00
#   srun --pty /bin/bash -i
#   cd ~/QTL-Revision/qtl_experiments
#   bash slurm/setup_env.sh
# ==============================================================================

set -euo pipefail

echo "============================================"
echo "  QTL Experiments - Environment Setup"
echo "  Hercules Supercomputer (CICA)"
echo "============================================"

module load Miniconda3

echo "[1/4] Creating conda environment..."
conda create --name qtl_revision --clone ibmq-dev -y || {
    echo "  ibmq-dev clone failed, creating from scratch..."
    conda create -n qtl_revision python=3.10 -y
}

conda activate qtl_revision

echo "[2/4] Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

echo "[3/4] Installing quantum libraries..."
pip install pennylane pennylane-lightning --quiet
pip install qiskit-machine-learning --quiet

echo "[4/4] Installing analysis dependencies..."
pip install scikit-learn matplotlib seaborn pandas scipy codecarbon tqdm pyyaml pillow --quiet

echo ""
echo "============================================"
echo "  Done! Activate with: conda activate qtl_revision"
echo "  Verify: python -c 'import torch, pennylane, qiskit; print(torch.cuda.is_available())'"
echo "============================================"
