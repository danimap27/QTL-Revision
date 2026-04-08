#!/bin/bash
# ==============================================================================
# Setup environment on Hercules supercomputer (Junta de Andalucia)
# Run this once before submitting jobs:
#   bash slurm/setup_env.sh
# ==============================================================================

set -euo pipefail

echo "============================================"
echo "  QTL Revision - Environment Setup"
echo "  Hercules Supercomputer"
echo "============================================"

# Load required modules
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1

# Create conda environment
echo "[1/4] Creating conda environment..."
conda create -n qtl_revision python=3.10 -y
conda activate qtl_revision

# Install PyTorch with CUDA 12.1 support
echo "[2/4] Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install quantum computing frameworks
echo "[3/4] Installing quantum frameworks..."
pip install pennylane pennylane-lightning
pip install qiskit qiskit-aer qiskit-machine-learning

# Install analysis and utility dependencies
echo "[4/4] Installing analysis dependencies..."
pip install scikit-learn matplotlib seaborn pandas scipy
pip install codecarbon tqdm pillow numpy

echo ""
echo "============================================"
echo "  Environment setup complete!"
echo "  Activate with: conda activate qtl_revision"
echo "============================================"
