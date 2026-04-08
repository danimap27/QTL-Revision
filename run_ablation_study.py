#!/usr/bin/env python3
"""
run_ablation_study.py -- Ablation study for qubit count and circuit depth.

Addresses MAJOR reviewer issue #7: systematic analysis of how the number of
qubits and circuit depth affect hybrid model performance.

Design:
    - Qubits:    [1, 2, 3, 4, 5, 6, 7, 8]
    - Depths:    [1, 2, 3, 4, 5]
    - Backbones: resnet18 (primary), optionally brain_tumor
    - Datasets:  hymenoptera (primary), brain_tumor
    - Framework: PennyLane ideal (cleanest signal)
    - Seeds:     [42, 123, 456, 789, 1024]
    - Total combinations for SLURM: 8 x 5 x 2 x 5 = 400

Usage examples:
    # SLURM array job (task IDs 0..399)
    python run_ablation_study.py --task-id $SLURM_ARRAY_TASK_ID

    # Run everything sequentially
    python run_ablation_study.py --run-all

    # Resume interrupted run (skip completed)
    python run_ablation_study.py --run-all --resume

    # Dry run to see what would execute
    python run_ablation_study.py --run-all --dry-run

    # Single task dry run
    python run_ablation_study.py --task-id 0 --dry-run
"""

import argparse
import itertools
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

QUBITS = [1, 2, 3, 4, 5, 6, 7, 8]
DEPTHS = [1, 2, 3, 4, 5]
DATASETS = ["hymenoptera", "brain_tumor"]
SEEDS = [42, 123, 456, 789, 1024]
BACKBONE = "resnet18"

# Build the full grid: qubits x depths x datasets x seeds
# Order: iterate seeds fastest so related configs are grouped together
GRID = list(itertools.product(QUBITS, DEPTHS, DATASETS, SEEDS))

TOTAL_JOBS = len(GRID)  # 8 * 5 * 2 * 5 = 400


def make_run_id(n_qubits, depth, dataset, seed):
    """Generate a unique run identifier."""
    return f"ablation_{n_qubits}q_depth{depth}_{dataset}_seed{seed}"


def result_csv_path(output_dir, n_qubits, depth, dataset, seed):
    """Path to the CSV file that signals a completed run."""
    run_id = make_run_id(n_qubits, depth, dataset, seed)
    return os.path.join(output_dir, f"{run_id}.csv")


def is_completed(output_dir, n_qubits, depth, dataset, seed):
    """Check whether a run has already been completed."""
    csv_path = result_csv_path(output_dir, n_qubits, depth, dataset, seed)
    return os.path.isfile(csv_path)


def build_command(n_qubits, depth, dataset, seed, output_dir):
    """Build the command list to dispatch a single ablation run."""
    run_id = make_run_id(n_qubits, depth, dataset, seed)
    cmd = [
        sys.executable, "train_cq_pennylane.py",
        "--dataset", dataset,
        "--model", BACKBONE,
        "--n-qubits", str(n_qubits),
        "--depth", str(depth),
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--id", run_id,
    ]
    return cmd


def run_single(task_id, output_dir, resume=False, dry_run=False):
    """Execute a single ablation configuration identified by task_id."""
    if task_id < 0 or task_id >= TOTAL_JOBS:
        print(f"ERROR: task-id {task_id} out of range [0, {TOTAL_JOBS - 1}]")
        return 1

    n_qubits, depth, dataset, seed = GRID[task_id]
    run_id = make_run_id(n_qubits, depth, dataset, seed)

    print(f"[{task_id}/{TOTAL_JOBS}] {run_id}")
    print(f"  qubits={n_qubits}  depth={depth}  dataset={dataset}  seed={seed}")

    # Resume check
    if resume and is_completed(output_dir, n_qubits, depth, dataset, seed):
        csv_path = result_csv_path(output_dir, n_qubits, depth, dataset, seed)
        print(f"  SKIP (already completed: {csv_path})")
        return 0

    cmd = build_command(n_qubits, depth, dataset, seed, output_dir)
    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"  DRY RUN: {cmd_str}")
        return 0

    print(f"  CMD: {cmd_str}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        print(f"  DONE in {elapsed:.1f}s (exit code {result.returncode})")
        return result.returncode
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"  FAILED after {elapsed:.1f}s (exit code {e.returncode})")
        return e.returncode
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1


def run_all(output_dir, resume=False, dry_run=False):
    """Run all ablation configurations sequentially."""
    print("=" * 70)
    print("ABLATION STUDY -- Qubit Count x Circuit Depth")
    print(f"Total configurations: {TOTAL_JOBS}")
    print(f"Qubits: {QUBITS}")
    print(f"Depths: {DEPTHS}")
    print(f"Datasets: {DATASETS}")
    print(f"Seeds: {SEEDS}")
    print(f"Backbone: {BACKBONE}")
    print(f"Output directory: {output_dir}")
    print(f"Resume: {resume}")
    print(f"Dry run: {dry_run}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    completed = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    for task_id in range(TOTAL_JOBS):
        n_qubits, depth, dataset, seed = GRID[task_id]

        if resume and is_completed(output_dir, n_qubits, depth, dataset, seed):
            skipped += 1
            continue

        rc = run_single(task_id, output_dir, resume=False, dry_run=dry_run)
        if rc == 0:
            completed += 1
        else:
            failed += 1

    elapsed = time.time() - t_start
    print("=" * 70)
    print("ABLATION STUDY COMPLETE")
    print(f"  Completed: {completed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 70)

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: qubit count x circuit depth (reviewer issue #7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task-id", type=int, default=None,
        help=f"SLURM array job index (0-{TOTAL_JOBS - 1})",
    )
    parser.add_argument(
        "--run-all", action="store_true",
        help="Run all configurations sequentially",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip configurations whose CSV output already exists",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/ablation",
        help="Directory for output files (default: results/ablation)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.task_id is None and not args.run_all:
        parser.error("Specify either --task-id N or --run-all")

    if args.task_id is not None and args.run_all:
        parser.error("--task-id and --run-all are mutually exclusive")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_all:
        return run_all(args.output_dir, resume=args.resume, dry_run=args.dry_run)
    else:
        return run_single(
            args.task_id, args.output_dir,
            resume=args.resume, dry_run=args.dry_run,
        )


if __name__ == "__main__":
    sys.exit(main())
