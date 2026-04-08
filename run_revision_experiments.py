#!/usr/bin/env python3
"""
run_revision_experiments.py -- Master orchestrator for revision experiments.

Addresses CRITICAL reviewer issue #1 (statistical validation) by running
all experiments with multiple seeds across 7 approaches, 4 backbones,
4 datasets, and 5 seeds (560 total runs).

Usage examples:
    # SLURM array job (task IDs 0..559)
    python run_revision_experiments.py --task-id $SLURM_ARRAY_TASK_ID

    # Run everything sequentially
    python run_revision_experiments.py --run-all

    # Resume interrupted run
    python run_revision_experiments.py --run-all --resume

    # Dry run to see what would execute
    python run_revision_experiments.py --run-all --dry-run

    # Single task dry run
    python run_revision_experiments.py --task-id 0 --dry-run
"""

import argparse
import csv
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

BACKBONES = ["resnet18", "mobilenetv2", "efficientnet_b0", "regnet_x_400mf"]

DATASETS = ["hymenoptera", "brain_tumor", "cats_dogs", "solar_dust"]

SEEDS = [42, 123, 456, 789, 1024]

TOTAL_EXPERIMENTS = len(APPROACHES) * len(BACKBONES) * len(DATASETS) * len(SEEDS)

# Estimated wall-clock seconds per approach (rough, for progress reporting)
_EST_SECS = {
    "classical": 120,
    "classical_mlp_matched": 120,
    "classical_mlp_standard": 120,
    "pennylane_ideal": 600,
    "pennylane_noisy": 900,
    "qiskit_ideal": 900,
    "qiskit_noisy": 1200,
}

# ---------------------------------------------------------------------------
# Index encoding / decoding
# ---------------------------------------------------------------------------

def decode_task_id(task_id: int):
    """Decode a linear task index into (approach, backbone, dataset, seed).

    Ordering (innermost to outermost):
        seed -> dataset -> backbone -> approach
    so that consecutive IDs share the same approach/backbone/dataset and only
    differ by seed -- friendly for filesystem caching.
    """
    n_seeds = len(SEEDS)
    n_datasets = len(DATASETS)
    n_backbones = len(BACKBONES)
    n_approaches = len(APPROACHES)

    if task_id < 0 or task_id >= TOTAL_EXPERIMENTS:
        raise ValueError(
            f"task-id {task_id} out of range [0, {TOTAL_EXPERIMENTS})"
        )

    seed_idx = task_id % n_seeds
    task_id //= n_seeds
    dataset_idx = task_id % n_datasets
    task_id //= n_datasets
    backbone_idx = task_id % n_backbones
    task_id //= n_backbones
    approach_idx = task_id % n_approaches

    return (
        APPROACHES[approach_idx],
        BACKBONES[backbone_idx],
        DATASETS[dataset_idx],
        SEEDS[seed_idx],
    )


def encode_task_id(approach: str, backbone: str, dataset: str, seed: int) -> int:
    """Inverse of decode_task_id -- useful for diagnostics."""
    a = APPROACHES.index(approach)
    b = BACKBONES.index(backbone)
    d = DATASETS.index(dataset)
    s = SEEDS.index(seed)
    n_seeds = len(SEEDS)
    n_datasets = len(DATASETS)
    n_backbones = len(BACKBONES)
    return ((a * n_backbones + b) * n_datasets + d) * n_seeds + s


def make_run_id(approach: str, backbone: str, dataset: str, seed: int) -> str:
    """Canonical run identifier used for filenames and logging."""
    return f"{approach}_{backbone}_{dataset}_seed{seed}"

# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

def result_csv_path(output_dir: str, run_id: str) -> str:
    """Path where the per-run result CSV is expected."""
    return os.path.join(output_dir, f"{run_id}.csv")


def is_completed(output_dir: str, run_id: str) -> bool:
    """Return True if a non-empty result CSV already exists for *run_id*."""
    path = result_csv_path(output_dir, run_id)
    if not os.path.isfile(path):
        return False
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_file_path() -> str:
    return os.path.join("results", "experiment_log.txt")


def log_message(msg: str) -> None:
    """Append a timestamped message to the shared experiment log."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    log_path = _log_file_path()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line)
    # Also print to stdout for console visibility
    print(line, end="", flush=True)

# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

def build_command(
    approach: str,
    backbone: str,
    dataset: str,
    seed: int,
    output_dir: str,
    run_id: str,
    epochs: int,
    batch_size: int,
    n_qubits: int,
    depth: int,
) -> list:
    """Return the subprocess command list for the given experiment."""

    common_args = [
        "--dataset", dataset,
        "--model", backbone,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--id", run_id,
    ]

    quantum_args = [
        "--n-qubits", str(n_qubits),
        "--depth", str(depth),
    ]

    python = sys.executable  # ensures we use the same interpreter

    if approach == "classical":
        return [python, "train_cc.py"] + common_args

    elif approach == "classical_mlp_matched":
        return [python, "train_cc_mlp.py"] + common_args + ["--head-type", "matched"]

    elif approach == "classical_mlp_standard":
        return [python, "train_cc_mlp.py"] + common_args + ["--head-type", "standard"]

    elif approach == "pennylane_ideal":
        return [python, "train_cq_pennylane.py"] + common_args + quantum_args

    elif approach == "pennylane_noisy":
        return [python, "train_cq_pennylane_noisy.py"] + common_args + quantum_args

    elif approach == "qiskit_ideal":
        return [python, "train_cq_qiskit.py"] + common_args + quantum_args

    elif approach == "qiskit_noisy":
        return [python, "train_cq_qiskit_noisy.py"] + common_args + quantum_args

    else:
        raise ValueError(f"Unknown approach: {approach}")

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_experiment(
    approach: str,
    backbone: str,
    dataset: str,
    seed: int,
    output_dir: str,
    epochs: int,
    batch_size: int,
    n_qubits: int,
    depth: int,
    dry_run: bool = False,
    resume: bool = False,
) -> bool:
    """Run a single experiment.  Returns True on success, False on failure."""

    run_id = make_run_id(approach, backbone, dataset, seed)

    # ------------------------------------------------------------------
    # Resume check
    # ------------------------------------------------------------------
    if resume and is_completed(output_dir, run_id):
        log_message(f"SKIP (completed) {run_id}")
        return True

    # ------------------------------------------------------------------
    # Build command
    # ------------------------------------------------------------------
    cmd = build_command(
        approach=approach,
        backbone=backbone,
        dataset=dataset,
        seed=seed,
        output_dir=output_dir,
        run_id=run_id,
        epochs=epochs,
        batch_size=batch_size,
        n_qubits=n_qubits,
        depth=depth,
    )

    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"[DRY-RUN] {cmd_str}")
        return True

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    log_message(f"START {run_id} | cmd: {cmd_str}")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=7200,  # 2-hour hard timeout per experiment
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            log_message(
                f"DONE  {run_id} | elapsed {elapsed:.1f}s | return code 0"
            )
            return True
        else:
            log_message(
                f"FAIL  {run_id} | elapsed {elapsed:.1f}s | return code {result.returncode}"
            )
            # Save stdout/stderr for debugging
            err_path = os.path.join(output_dir, f"{run_id}_ERROR.log")
            os.makedirs(output_dir, exist_ok=True)
            with open(err_path, "w", encoding="utf-8") as fh:
                fh.write(result.stdout or "")
            return False

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        log_message(f"TIMEOUT {run_id} | elapsed {elapsed:.1f}s (limit 7200s)")
        return False
    except Exception as exc:
        elapsed = time.time() - t0
        log_message(f"ERROR {run_id} | elapsed {elapsed:.1f}s | {exc}")
        return False

# ---------------------------------------------------------------------------
# Progress & summary helpers
# ---------------------------------------------------------------------------

def print_grid_summary() -> None:
    """Print a summary of the experiment grid."""
    print("=" * 70)
    print("REVISION EXPERIMENTS -- GRID SUMMARY")
    print("=" * 70)
    print(f"  Approaches ({len(APPROACHES)}): {', '.join(APPROACHES)}")
    print(f"  Backbones  ({len(BACKBONES)}):  {', '.join(BACKBONES)}")
    print(f"  Datasets   ({len(DATASETS)}):   {', '.join(DATASETS)}")
    print(f"  Seeds      ({len(SEEDS)}):      {', '.join(str(s) for s in SEEDS)}")
    print(f"  TOTAL RUNS: {TOTAL_EXPERIMENTS}")
    total_est = sum(
        _EST_SECS.get(a, 300)
        for a in APPROACHES
        for _ in BACKBONES
        for _ in DATASETS
        for _ in SEEDS
    )
    hours = total_est / 3600
    print(f"  Estimated sequential wall-clock: {hours:.1f} hours ({total_est} seconds)")
    print("=" * 70)


def print_progress(done: int, total: int, successes: int, failures: int, skipped: int) -> None:
    pct = done / total * 100 if total else 0
    bar_len = 40
    filled = int(bar_len * done / total) if total else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    print(
        f"\r  [{bar}] {done}/{total} ({pct:.1f}%) "
        f"| ok={successes} fail={failures} skip={skipped}",
        end="",
        flush=True,
    )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Master orchestrator for revision experiments. "
            "Runs 7 approaches x 4 backbones x 4 datasets x 5 seeds = 560 experiments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--task-id",
        type=int,
        default=None,
        help=(
            "Linear task index (0..559) for SLURM array jobs. "
            "Decoded into (approach, backbone, dataset, seed)."
        ),
    )
    mode.add_argument(
        "--run-all",
        action="store_true",
        default=False,
        help="Run all 560 experiments sequentially.",
    )

    p.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip experiments whose result CSV already exists in --output-dir.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("results", "seeds"),
        help="Directory for per-run result CSVs (default: results/seeds).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print commands that would run without executing them.",
    )
    p.add_argument(
        "--n-qubits",
        type=int,
        default=4,
        help="Number of qubits for quantum approaches (default: 4).",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Quantum circuit depth (default: 3).",
    )

    return p.parse_args(argv)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Single task mode (SLURM)
    # ------------------------------------------------------------------
    if args.task_id is not None:
        approach, backbone, dataset, seed = decode_task_id(args.task_id)
        run_id = make_run_id(approach, backbone, dataset, seed)

        print(f"Task ID {args.task_id} -> {run_id}")
        print(f"  approach={approach}  backbone={backbone}  dataset={dataset}  seed={seed}")

        ok = run_experiment(
            approach=approach,
            backbone=backbone,
            dataset=dataset,
            seed=seed,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_qubits=args.n_qubits,
            depth=args.depth,
            dry_run=args.dry_run,
            resume=args.resume,
        )
        return 0 if ok else 1

    # ------------------------------------------------------------------
    # Run-all mode
    # ------------------------------------------------------------------
    print_grid_summary()

    if args.dry_run:
        print("\n--- DRY RUN (no experiments will be executed) ---\n")

    successes = 0
    failures = 0
    skipped = 0
    t_start = time.time()

    for idx in range(TOTAL_EXPERIMENTS):
        approach, backbone, dataset, seed = decode_task_id(idx)
        run_id = make_run_id(approach, backbone, dataset, seed)

        # Check resume before counting
        if args.resume and is_completed(args.output_dir, run_id):
            skipped += 1
            print_progress(idx + 1, TOTAL_EXPERIMENTS, successes, failures, skipped)
            continue

        ok = run_experiment(
            approach=approach,
            backbone=backbone,
            dataset=dataset,
            seed=seed,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_qubits=args.n_qubits,
            depth=args.depth,
            dry_run=args.dry_run,
            resume=False,  # already checked above
        )

        if ok:
            successes += 1
        else:
            failures += 1

        print_progress(idx + 1, TOTAL_EXPERIMENTS, successes, failures, skipped)

    elapsed_total = time.time() - t_start
    print()  # newline after progress bar
    print("=" * 70)
    print("ALL EXPERIMENTS FINISHED")
    print(f"  Total:    {TOTAL_EXPERIMENTS}")
    print(f"  Success:  {successes}")
    print(f"  Failed:   {failures}")
    print(f"  Skipped:  {skipped}")
    print(f"  Elapsed:  {elapsed_total:.1f}s ({elapsed_total/3600:.2f}h)")
    print("=" * 70)

    log_message(
        f"BATCH COMPLETE | total={TOTAL_EXPERIMENTS} ok={successes} "
        f"fail={failures} skip={skipped} elapsed={elapsed_total:.1f}s"
    )

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
