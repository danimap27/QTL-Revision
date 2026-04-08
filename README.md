# Hybrid Classical-Quantum Transfer Learning

A systematic benchmark of quantum-enhanced transfer learning for image classification, comparing classical CNN backbones with variational quantum circuit (VQC) heads across multiple frameworks, noise models, and datasets.

## What This Project Does

This repository contains the code and experimental pipeline for a study that replaces the final fully connected classification head of pretrained CNNs with parameterized quantum circuits, then measures whether the quantum head helps, hurts, or makes no practical difference on real image classification tasks. We freeze the convolutional backbone (ResNet-18, MobileNetV2, EfficientNet-B0, RegNet-X-400MF) and attach either a standard linear layer, a size-matched MLP, or a variational quantum circuit built in PennyLane or Qiskit. Every configuration is tested on four datasets of varying difficulty.

A key concern when claiming "quantum advantage" in hybrid models is whether the results hold up under realistic hardware conditions. To address this, we include experiments with calibrated IBM-Brisbane noise models that introduce amplitude damping, phase damping, and depolarizing errors into the quantum circuits. We also run a structured noise decomposition to isolate the contribution of each noise channel individually, rather than just reporting aggregate noisy-vs-ideal comparisons.

All experiments are repeated across five random seeds and analyzed with proper statistical tests (Welch's t-test or Mann-Whitney U with Bonferroni correction), because single-run accuracy numbers on small datasets can be misleading. The pipeline also tracks energy consumption via CodeCarbon, includes a barren plateau gradient analysis, and provides an ablation study over qubit count and circuit depth.

## Key Results

<!-- TODO: Fill in after running experiments on Hercules -->

Results will be reported as mean +/- standard deviation across 5 seeds. Statistical significance is assessed with Welch's t-test (when normality holds) or Mann-Whitney U (otherwise), with Bonferroni correction for multiple comparisons.

Preliminary findings suggest that quantum classification heads do not consistently outperform classical baselines of matched capacity, and that realistic noise further erodes any marginal gains. Full tables and figures will appear in the published paper.

## Project Structure

```
QTL_Revision/
|
|-- train_cc.py                    # Classical baseline (frozen CNN + linear head)
|-- train_cc_mlp.py                # Classical MLP head (parameter-matched control)
|-- train_cq_pennylane.py          # PennyLane ideal quantum circuit head
|-- train_cq_pennylane_noisy.py    # PennyLane with IBM-Brisbane noise model
|-- train_cq_pennylane_spsa.py     # PennyLane with SPSA optimizer (gradient-free)
|-- train_cq_qiskit.py             # Qiskit ideal quantum circuit head
|-- train_cq_qiskit_noisy.py       # Qiskit with IBM-Brisbane noise model
|
|-- run_revision_experiments.py    # Master orchestrator (560 experiments, SLURM-ready)
|-- run_ablation_study.py          # Ablation over qubit count and circuit depth
|-- run_gradient_analysis.py       # Barren plateau gradient variance analysis
|-- run_noise_decomposition.py     # Isolate individual noise channel contributions
|-- run_complete_benchmark.py      # Legacy full benchmark script
|-- run_paper_experiments.py       # Legacy single-run experiment script
|
|-- analyze_results.py             # Statistical analysis, LaTeX tables, figures
|-- verify_models.py               # Sanity checks for model architecture
|-- test_qiskit_real_hardware.py   # Test connection to IBM Quantum hardware
|-- experiment_configs.json        # Centralized experiment configuration
|
|-- slurm/
|   |-- setup_env.sh               # Environment setup on HPC cluster
|   |-- submit_all.sh              # Submit all SLURM job arrays
|   |-- run_main_seeds.sh          # SLURM script for main 560 experiments
|   |-- run_ablation.sh            # SLURM script for ablation study
|   |-- run_gradient.sh            # SLURM script for gradient analysis
|   |-- run_noise.sh               # SLURM script for noise decomposition
|   |-- run_optimizer_control.sh   # SLURM script for optimizer comparison
|   |-- aggregate.sh               # Aggregate results after all jobs finish
|
|-- datasets/                      # Image datasets (not tracked in git)
|   |-- hymenoptera/               # Ants vs. bees (ImageNet subset)
|   |-- brain_tumor/               # MRI brain tumor classification
|   |-- cats_dogs/                 # Cats vs. dogs
|   |-- solar_dust/                # Solar panel dust detection
|
|-- results/                       # Experiment outputs (not tracked in git)
|   |-- seeds/                     # Main repeated-seed results
|   |-- ablation/                  # Ablation study results
|   |-- gradient/                  # Barren plateau analysis
|   |-- noise/                     # Noise decomposition results
|   |-- figures/                   # Generated plots
|   |-- aggregated/                # Aggregated tables and LaTeX
|
|-- requirements.txt
|-- LICENSE                        # MIT
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- CUDA 12.1+ and a compatible GPU (for reasonable training times)
- Approximately 10 GB disk space for datasets
- An IBM Quantum account (optional, only needed for real hardware tests)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/QTL_Revision.git
cd QTL_Revision

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: install CodeCarbon for energy tracking
pip install codecarbon
```

### Quick Test

Run a single classical baseline experiment on the smallest dataset to verify everything works:

```bash
python train_cc.py --dataset hymenoptera --backbone resnet18 --epochs 5 --seed 42
```

This should finish in under two minutes on a GPU and print test accuracy at the end. To test the quantum pipeline:

```bash
python train_cq_pennylane.py --dataset hymenoptera --backbone resnet18 --epochs 5 --seed 42
```

## Datasets

| Dataset | Classes | Approx. Size | Domain | Source |
|---------|---------|-------------|--------|--------|
| Hymenoptera | 2 (ants/bees) | ~400 images | Natural images | PyTorch tutorials / ImageNet subset |
| Brain Tumor | 2+ | ~3,000 images | Medical MRI | Kaggle |
| Cats & Dogs | 2 | ~25,000 images | Natural images | Kaggle / Microsoft |
| Solar Dust | 2 | ~1,500 images | Industrial inspection | Custom / research dataset |

Download and place each dataset in the `datasets/` directory with `train/` and `test/` (or `val/`) subdirectories. Each class should be in its own subfolder (standard ImageFolder layout):

```
datasets/hymenoptera/
    train/
        ants/
        bees/
    val/
        ants/
        bees/
```

## Running Experiments

### Single Experiment

Each training script accepts command-line arguments for dataset, backbone, number of epochs, and random seed.

**Classical baseline:**
```bash
python train_cc.py --dataset brain_tumor --backbone efficientnet_b0 --epochs 30 --seed 42
```

**PennyLane quantum head (ideal):**
```bash
python train_cq_pennylane.py --dataset brain_tumor --backbone efficientnet_b0 --epochs 30 --seed 42
```

**PennyLane quantum head (noisy, IBM-Brisbane calibration):**
```bash
python train_cq_pennylane_noisy.py --dataset brain_tumor --backbone efficientnet_b0 --epochs 30 --seed 42
```

**Qiskit quantum head (ideal / noisy):**
```bash
python train_cq_qiskit.py --dataset brain_tumor --backbone resnet18 --epochs 30 --seed 42
python train_cq_qiskit_noisy.py --dataset brain_tumor --backbone resnet18 --epochs 30 --seed 42
```

### Full Benchmark (with Statistical Validation)

The master orchestrator runs all 560 experiments (7 approaches x 4 backbones x 4 datasets x 5 seeds):

```bash
# See what would run (dry run)
python run_revision_experiments.py --run-all --dry-run

# Run everything sequentially (expect several days on a single GPU)
python run_revision_experiments.py --run-all

# Resume after interruption (skips completed experiments)
python run_revision_experiments.py --run-all --resume

# Run a single experiment by task ID (useful for debugging)
python run_revision_experiments.py --task-id 0
```

### Additional Analyses

```bash
# Ablation study: qubit count x circuit depth
python run_ablation_study.py --run-all

# Barren plateau gradient analysis
python run_gradient_analysis.py --run-all

# Noise decomposition (isolate individual noise channels)
python run_noise_decomposition.py --run-all
```

### On HPC / SLURM

The `slurm/` directory contains ready-to-use job scripts for cluster execution:

```bash
# Set up the environment on the cluster (run once)
bash slurm/setup_env.sh

# Submit all experiment arrays
bash slurm/submit_all.sh

# After all jobs finish, aggregate results
bash slurm/aggregate.sh
```

Each SLURM script uses array jobs so that individual experiments map to `--task-id $SLURM_ARRAY_TASK_ID`. This parallelizes the 560 main experiments across available GPU nodes.

## Reproducibility

- All experiments use fixed random seeds: **42, 123, 456, 789, 1024**
- Seeds are set for Python, NumPy, PyTorch (CPU and CUDA), and quantum framework RNGs
- Results are reported as **mean +/- standard deviation** across 5 seeds
- Statistical significance is tested with **Welch's t-test** (when Shapiro-Wilk confirms normality) or **Mann-Whitney U** (otherwise), with **Bonferroni correction** for multiple comparisons
- Energy consumption is tracked with **CodeCarbon** (when installed)
- PyTorch deterministic mode is enabled where possible (`torch.use_deterministic_algorithms`)

## Analysis

After experiments are complete, run the analysis pipeline:

```bash
# Generate all tables and figures
python analyze_results.py

# Specify a custom results directory
python analyze_results.py --results-dir results

# Output LaTeX tables only
python analyze_results.py --format latex
```

This produces:

- Main results table (mean +/- std for each approach/backbone/dataset)
- Pairwise statistical significance tests
- Energy consumption comparison table
- Ablation study table and heatmaps
- Barren plateau gradient variance plots
- Noise decomposition breakdown table
- Publication-quality figures at 300 DPI
- LaTeX-ready tables for direct inclusion in the manuscript

Output goes to `results/aggregated/` and `results/figures/`.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{authors2026hybrid,
  title     = {Hybrid Classical-Quantum Transfer Learning with Noisy Quantum Circuits},
  author    = {TODO},
  journal   = {Computer Modeling in Engineering \& Sciences (CMES)},
  year      = {2026},
  volume    = {TODO},
  pages     = {TODO},
  doi       = {TODO},
  note      = {Manuscript CMES-82712}
}
```

## License

This project is released under the [MIT License](LICENSE).
