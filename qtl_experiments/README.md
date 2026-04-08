# QTL Experiments Framework

Unified experiment framework for **Hybrid Classical-Quantum Transfer Learning with Noisy Quantum Circuits**.

## Architecture

```
qtl_experiments/
├── config.yaml          # All experiment configurations
├── runner.py            # Orchestrator: generates runs, filters, resumes
├── trainer.py           # Generic training loop with checkpoints
├── heads/               # Modular classification heads
│   ├── linear_head.py   # Baseline linear layer
│   ├── mlp_a_head.py    # Parameter-matched MLP (~12 params)
│   ├── mlp_b_head.py    # Standard MLP (128→64→C)
│   ├── pennylane_head.py # PennyLane VQC (ideal + noisy)
│   └── qiskit_head.py   # Qiskit VQC (ideal + noisy)
├── data/loader.py       # Unified dataset loading
├── slurm/               # HPC scripts
│   ├── setup_env.sh     # Hercules environment setup
│   ├── run_experiments.sh # SLURM array job template
│   └── distribute.py    # Multi-account job distribution
└── results/             # Output: runs.csv, predictions.csv, training_log.csv
```

## Quick Start

```bash
# Run everything
python runner.py --config config.yaml

# Filter runs
python runner.py --config config.yaml --dataset hymenoptera --head pl_ideal --seed 42

# Dry run (list all combinations)
python runner.py --config config.yaml --dry-run

# Count total runs
python runner.py --config config.yaml --count
```

## Output

Three CSV files are produced:

- **runs.csv** — One row per completed run (metrics, timing, energy)
- **predictions.csv** — One row per test sample per run
- **training_log.csv** — One row per epoch per run

## SLURM (Hercules)

```bash
# Setup (once, from interactive node)
bash slurm/setup_env.sh

# Distribute across accounts
python slurm/distribute.py --config config.yaml          # dry-run
python slurm/distribute.py --config config.yaml --submit  # submit

# Or manual single-account
sbatch --array=0-559%80 slurm/run_experiments.sh
```

## Resume

The runner automatically skips completed runs found in `runs.csv`. Just re-run the same command after an interruption.
