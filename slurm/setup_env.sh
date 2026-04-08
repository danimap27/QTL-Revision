#!/bin/bash
# ==============================================================================
# Setup environment on Hercules (CICA - Junta de Andalucía)
# Run ONCE from an interactive node (NOT login node):
#
#   salloc --mem=16G -c 4 -t 06:00:00 srun --pty /bin/bash -i
#   cd ~/QTL-Revision
#   bash slurm/setup_env.sh
# ==============================================================================

set -euo pipefail

echo "============================================"
echo "  QTL Revision - Environment Setup"
echo "  Hercules Supercomputer (CICA)"
echo "============================================"

# Load Miniconda3 module (Hercules-specific)
module load Miniconda3

# Option A: Clone from the existing IBM Quantum environment (faster)
# The ibmq-dev env already has Qiskit optimized by IBM
echo "[1/4] Cloning ibmq-dev conda environment as base..."
conda create --name qtl_revision --clone ibmq-dev -y || {
    echo "  ibmq-dev clone failed, creating from scratch..."
    conda create -n qtl_revision python=3.10 -y
}

conda activate qtl_revision

# Install PyTorch with CUDA support
echo "[2/4] Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

# Install PennyLane (Qiskit should already be in ibmq-dev)
echo "[3/4] Installing PennyLane and missing packages..."
pip install pennylane pennylane-lightning --quiet
pip install qiskit-machine-learning --quiet  # may already exist in ibmq-dev

# Install analysis and energy tracking dependencies
echo "[4/4] Installing analysis dependencies..."
pip install scikit-learn matplotlib seaborn pandas scipy codecarbon tqdm pillow --quiet

echo ""
echo "============================================"
echo "  Done! To activate: conda activate qtl_revision"
echo "  To verify: python -c 'import torch, pennylane, qiskit; print(torch.cuda.is_available())'"
echo "============================================"
