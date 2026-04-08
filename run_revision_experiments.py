#!/usr/bin/env python3
"""
run_revision_experiments.py -- Master orchestrator for paper revision experiments.

Full factorial grid:
  5 seeds × 4 datasets × 4 backbones × 7 approaches × 3 qubit configs × 3 depths
  = 5 × 4 × 4 × 7 × 3 × 3 = 5,040 total runs
  (classical approaches ignore qubit/depth → effectively 3,120 unique runs)

For classical approaches (classical, classical_mlp_matched, classical_mlp_standard),
qubit/depth parameters are ignored but we only run once per (backbone, dataset, seed)
rather than duplicating across qubit/depth combos.

Usage:
    # SLURM array job
    python run_revision_experiments.py --task-id $SLURM_ARRAY_TASK_ID

    # Run everything sequentially
    python run_revision_experiments.py --run-all

    # Dry run
    python run_revision_experiments.py --run-all --dry-run
"""

import argparse
import datetime
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

APPROACHES = [
    "classical",
    "classical_mlp_matched",
    "classical_mlp_standard",
    "pennylane_ideal",
    "pennylane_noisy",
    "qiskit_ideal",
    "qiskit_noisy",
]

CLASSICAL_APPROACHES = {"classical", "classical_mlp_matched", "classical_mlp_standard"}
QUANTUM_APPROACHES = {"pennylane_ideal", "pennylane_noisy", "qiskit_ideal", "qiskit_noisy"}

BACKBONES = ["resnet18", "mobilenetv2", "efficientnet_b0", "regnet_x_400mf"]
DATASETS = ["hymenoptera", "brain_tumor", "cats_dogs", "solar_dust"]
SEEDS = [42, 123, 456, 789, 1024]
QUBITS = [4, 8, 16]
DEPTHS = [1, 3, 5]

# Estimated seconds per run (for progress reporting)
_EST_SECS = {
    "classical": 120,
    "classical_mlp_matched": 120,
    "classical_mlp_standard": 120,
    "pennylane_ideal": 600,
    "pennylane_noisy": 1200,
    "qiskit_ideal": 900,
    "qiskit_noisy": 1500,
}

# ---------------------------------------------------------------------------
# Build the flat experiment list
# ---------------------------------------------------------------------------

def build_experiment_list():
    """Build the complete list of unique experiments.

    Classical approaches: 3 × 4 backbones × 4 datasets × 5 seeds = 240
    Quantum approaches:   4 × 4 backbones × 4 datasets × 5 seeds × 3 qubits × 3 depths = 2880
    Total: 3120
    """
    experiments = []

    # Classical approaches (no qubit/depth variation)
    for approach in sorted(CLASSICAL_APPROACHES):
        for backbone in BACKBONES:
            for dataset in DATASETS:
                for seed in SEEDS:
                    experiments.append({
                        'approach': approach,
                        'backbone': backbone,
                        'dataset': dataset,
                        'seed': seed,
                        'n_qubits': 0,  # Not applicable
                        'depth': 0,     # Not applicable
                    })

    # Quantum approaches (full qubit × depth grid)
    for approach in sorted(QUANTUM_APPROACHES):
        for backbone in BACKBONES:
            for dataset in DATASETS:
                for n_qubits in QUBITS:
                    for depth in DEPTHS:
                        for seed in SEEDS:
                            experiments.append({
                                'approach': approach,
                                'backbone': backbone,
                                'dataset': dataset,
                                'seed': seed,
                                'n_qubits': n_qubits,
                                'depth': depth,
                            })

    return experiments

EXPERIMENTS = build_experiment_list()
TOTAL_EXPERIMENTS = len(EXPERIMENTS)

# ---------------------------------------------------------------------------
# Run ID and resume logic
# ---------------------------------------------------------------------------

def make_run_id(exp):
    """Canonical run identifier used for filenames and logging."""
    if exp['approach'] in CLASSICAL_APPROACHES:
        return f"{exp['approach']}_{exp['backbone']}_{exp['dataset']}_seed{exp['seed']}"
    else:
        return (f"{exp['approach']}_{exp['backbone']}_{exp['dataset']}"
                f"_{exp['n_qubits']}q_d{exp['depth']}_seed{exp['seed']}")


def is_completed(output_dir, run_id):
    """Return True if a non-empty result CSV already exists."""
    path = os.path.join(output_dir, f"{run_id}.csv")
    return os.path.isfile(path) and os.path.getsize(path) > 0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_message(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    log_path = os.path.join("results", "experiment_log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)

# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

def build_command(exp, output_dir, epochs, batch_size):
    """Return the subprocess command list for the given experiment."""
    python = sys.executable
    approach = exp['approach']
    run_id = make_run_id(exp)

    common = [
        "--dataset", exp['dataset'],
        "--model", exp['backbone'],
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seed", str(exp['seed']),
        "--output-dir", output_dir,
        "--id", run_id,
    ]

    quantum = ["--n-qubits", str(exp['n_qubits']), "--depth", str(exp['depth'])]

    if approach == "classical":
        return [python, "train_cc.py"] + common
    elif approach == "classical_mlp_matched":
        return [python, "train_cc_mlp.py"] + common + ["--head-type", "matched"]
    elif approach == "classical_mlp_standard":
        return [python, "train_cc_mlp.py"] + common + ["--head-type", "standard"]
    elif approach == "pennylane_ideal":
        return [python, "train_cq_pennylane.py"] + common + quantum
    elif approach == "pennylane_noisy":
        return [python, "train_cq_pennylane_noisy.py"] + common + quantum
    elif approach == "qiskit_ideal":
        return [python, "train_cq_qiskit.py"] + common + quantum
    elif approach == "qiskit_noisy":
        return [python, "train_cq_qiskit_noisy.py"] + common + quantum
    else:
        raise ValueError(f"Unknown approach: {approach}")

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_experiment(exp, output_dir, epochs, batch_size, dry_run=False, resume=False):
    """Run a single experiment. Returns True on success."""
    run_id = make_run_id(exp)

    if resume and is_completed(output_dir, run_id):
        log_message(f"SKIP (completed) {run_id}")
        return True

    cmd = build_command(exp, output_dir, epochs, batch_size)
    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"[DRY-RUN] {cmd_str}")
        return True

    log_message(f"START {run_id} | cmd: {cmd_str}")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=14400,  # 4-hour hard timeout (16 qubits can be slow)
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            log_message(f"DONE  {run_id} | {elapsed:.1f}s")
            return True
        else:
            log_message(f"FAIL  {run_id} | {elapsed:.1f}s | rc={result.returncode}")
            err_path = os.path.join(output_dir, f"{run_id}_ERROR.log")
            os.makedirs(output_dir, exist_ok=True)
            with open(err_path, "w", encoding="utf-8") as fh:
                fh.write(result.stdout or "")
            return False

    except subprocess.TimeoutExpired:
        log_message(f"TIMEOUT {run_id} | {time.time() - t0:.1f}s (limit 14400s)")
        return False
    except Exception as exc:
        log_message(f"ERROR {run_id} | {time.time() - t0:.1f}s | {exc}")
        return False

# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def print_grid_summary():
    n_classical = len(CLASSICAL_APPROACHES) * len(BACKBONES) * len(DATASETS) * len(SEEDS)
    n_quantum = len(QUANTUM_APPROACHES) * len(BACKBONES) * len(DATASETS) * len(SEEDS) * len(QUBITS) * len(DEPTHS)

    print("=" * 70)
    print("REVISION EXPERIMENTS -- EXPANDED GRID")
    print("=" * 70)
    print(f"  Approaches:  {', '.join(APPROACHES)}")
    print(f"  Backbones:   {', '.join(BACKBONES)}")
    print(f"  Datasets:    {', '.join(DATASETS)}")
    print(f"  Seeds:       {', '.join(str(s) for s in SEEDS)}")
    print(f"  Qubits:      {', '.join(str(q) for q in QUBITS)}")
    print(f"  Depths:      {', '.join(str(d) for d in DEPTHS)}")
    print(f"  Classical runs:  {n_classical}")
    print(f"  Quantum runs:    {n_quantum}")
    print(f"  TOTAL RUNS:      {TOTAL_EXPERIMENTS}")
    print("=" * 70)


def print_progress(done, total, successes, failures, skipped):
    pct = done / total * 100 if total else 0
    bar_len = 40
    filled = int(bar_len * done / total) if total else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    print(
        f"\r  [{bar}] {done}/{total} ({pct:.1f}%) "
        f"| ok={successes} fail={failures} skip={skipped}",
        end="", flush=True,
    )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Master orchestrator: 3120 experiments across full factorial grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task-id", type=int, default=None,
                      help=f"SLURM array task index (0..{TOTAL_EXPERIMENTS - 1})")
    mode.add_argument("--run-all", action="store_true", default=False,
                      help="Run all experiments sequentially")

    p.add_argument("--resume", action="store_true", default=False,
                   help="Skip completed experiments")
    p.add_argument("--output-dir", type=str, default=os.path.join("results", "seeds"),
                   help="Output directory for CSVs (default: results/seeds)")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Print commands without executing")

    return p.parse_args(argv)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    # Single task mode (SLURM)
    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= TOTAL_EXPERIMENTS:
            print(f"ERROR: task-id {args.task_id} out of range [0, {TOTAL_EXPERIMENTS})")
            return 1

        exp = EXPERIMENTS[args.task_id]
        run_id = make_run_id(exp)
        print(f"Task ID {args.task_id} -> {run_id}")
        print(f"  {exp}")

        ok = run_experiment(exp, args.output_dir, args.epochs, args.batch_size,
                           dry_run=args.dry_run, resume=args.resume)
        return 0 if ok else 1

    # Run-all mode
    print_grid_summary()
    if args.dry_run:
        print("\n--- DRY RUN ---\n")

    successes, failures, skipped = 0, 0, 0
    t_start = time.time()

    for idx, exp in enumerate(EXPERIMENTS):
        run_id = make_run_id(exp)

        if args.resume and is_completed(args.output_dir, run_id):
            skipped += 1
            print_progress(idx + 1, TOTAL_EXPERIMENTS, successes, failures, skipped)
            continue

        ok = run_experiment(exp, args.output_dir, args.epochs, args.batch_size,
                           dry_run=args.dry_run, resume=False)
        if ok:
            successes += 1
        else:
            failures += 1
        print_progress(idx + 1, TOTAL_EXPERIMENTS, successes, failures, skipped)

    elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print(f"COMPLETE | total={TOTAL_EXPERIMENTS} ok={successes} fail={failures} "
          f"skip={skipped} elapsed={elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print("=" * 70)

    log_message(f"BATCH COMPLETE | total={TOTAL_EXPERIMENTS} ok={successes} "
                f"fail={failures} skip={skipped} elapsed={elapsed:.1f}s")
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
