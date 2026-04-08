#!/usr/bin/env python3
"""
run_gradient_analysis.py -- Barren plateau analysis via gradient variance.

Addresses MAJOR reviewer issue #5: demonstrate whether barren plateaus
affect the quantum circuits used in our hybrid models.

Method:
    For each (n_qubits, depth) configuration:
    1. Create PennyLane quantum circuit (same architecture as train_cq_pennylane.py)
    2. Initialize n_samples random parameter sets
    3. For each parameter set, compute the gradient of the cost function
       via the parameter-shift rule
    4. Compute Var(gradient) across all random initializations
    5. If Var decays exponentially with qubit count -> barren plateaus present

Configurations: qubits [2, 4, 6, 8] x depths [1, 2, 3, 4, 5] = 20 configs

The circuit architecture matches train_cq_pennylane.py exactly:
    - AngleEmbedding on all wires
    - BasicEntanglerLayers with shape (depth, n_qubits)
    - Measurement: expval(PauliZ(0))

Output:
    - results/gradient/bp_variance.csv
    - results/figures/barren_plateau_analysis.png

Usage examples:
    # SLURM array job (task IDs 0..19)
    python run_gradient_analysis.py --task-id $SLURM_ARRAY_TASK_ID

    # Run all configs sequentially
    python run_gradient_analysis.py --run-all

    # Custom number of samples
    python run_gradient_analysis.py --run-all --n-samples 500

    # Custom output directory
    python run_gradient_analysis.py --run-all --output-dir results/gradient
"""

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np
import pennylane as qml

# ---------------------------------------------------------------------------
# Configuration grid
# ---------------------------------------------------------------------------

QUBITS = [2, 4, 6, 8]
DEPTHS = [1, 2, 3, 4, 5]

GRID = list(itertools.product(QUBITS, DEPTHS))
TOTAL_CONFIGS = len(GRID)  # 4 * 5 = 20

DEFAULT_N_SAMPLES = 200


# ---------------------------------------------------------------------------
# Core analysis: gradient variance for a single configuration
# ---------------------------------------------------------------------------

def analyze_gradient_variance(n_qubits, depth, n_samples=DEFAULT_N_SAMPLES):
    """
    Compute the variance of the gradient of the cost function w.r.t. the
    first weight parameter across n_samples random initializations.

    The circuit matches the architecture in train_cq_pennylane.py:
        - qml.AngleEmbedding(inputs, wires=range(n_qubits))
        - qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        - return qml.expval(qml.PauliZ(0))
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    gradients = []
    for _ in range(n_samples):
        weights = np.random.uniform(-np.pi, np.pi, (depth, n_qubits))
        inputs = np.random.uniform(-np.pi / 2, np.pi / 2, n_qubits)

        # Parameter-shift rule for gradient w.r.t. weights[0, 0]
        shifted_plus = weights.copy()
        shifted_minus = weights.copy()
        shifted_plus[0, 0] += np.pi / 2
        shifted_minus[0, 0] -= np.pi / 2

        grad = (circuit(inputs, shifted_plus) - circuit(inputs, shifted_minus)) / 2
        gradients.append(float(grad))

    gradient_variance = float(np.var(gradients))
    gradient_mean = float(np.mean(gradients))

    return gradient_variance, gradient_mean


# ---------------------------------------------------------------------------
# Single configuration runner
# ---------------------------------------------------------------------------

def run_single_config(task_id, output_dir, n_samples, dry_run=False):
    """Run gradient analysis for a single (n_qubits, depth) configuration."""
    if task_id < 0 or task_id >= TOTAL_CONFIGS:
        print(f"ERROR: task-id {task_id} out of range [0, {TOTAL_CONFIGS - 1}]")
        return None

    n_qubits, depth = GRID[task_id]
    print(f"[{task_id}/{TOTAL_CONFIGS}] n_qubits={n_qubits}  depth={depth}  n_samples={n_samples}")

    if dry_run:
        print(f"  DRY RUN: would analyze gradient variance for {n_qubits} qubits, depth {depth}")
        return None

    t0 = time.time()
    variance, mean = analyze_gradient_variance(n_qubits, depth, n_samples)
    elapsed = time.time() - t0

    print(f"  Var(grad) = {variance:.6e}  Mean(grad) = {mean:.6e}  ({elapsed:.1f}s)")

    # Save individual result
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"bp_{n_qubits}q_depth{depth}.csv")
    with open(result_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_qubits", "depth", "gradient_variance", "gradient_mean", "n_samples"])
        writer.writerow([n_qubits, depth, variance, mean, n_samples])
    print(f"  Saved: {result_file}")

    return {
        "n_qubits": n_qubits,
        "depth": depth,
        "gradient_variance": variance,
        "gradient_mean": mean,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Aggregation: combine all results and generate figure
# ---------------------------------------------------------------------------

def aggregate_results(output_dir, fig_dir):
    """
    Combine individual CSV results into bp_variance.csv and generate
    the barren plateau analysis figure.
    """
    # Collect all individual results
    all_results = []
    for n_qubits, depth in GRID:
        result_file = os.path.join(output_dir, f"bp_{n_qubits}q_depth{depth}.csv")
        if os.path.isfile(result_file):
            with open(result_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_results.append(row)

    if not all_results:
        print("WARNING: No results found to aggregate.")
        return

    # Write combined CSV
    combined_csv = os.path.join(output_dir, "bp_variance.csv")
    with open(combined_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_qubits", "depth", "gradient_variance", "gradient_mean", "n_samples"])
        for r in all_results:
            writer.writerow([
                r["n_qubits"], r["depth"],
                r["gradient_variance"], r["gradient_mean"], r["n_samples"],
            ])
    print(f"Combined results saved: {combined_csv}")

    # Generate figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(fig_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 7))

        # Organize data by depth
        depth_data = {}
        for r in all_results:
            d = int(r["depth"])
            q = int(r["n_qubits"])
            v = float(r["gradient_variance"])
            if d not in depth_data:
                depth_data[d] = {"qubits": [], "variances": []}
            depth_data[d]["qubits"].append(q)
            depth_data[d]["variances"].append(v)

        markers = ["o", "s", "^", "D", "v"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, d in enumerate(sorted(depth_data.keys())):
            qubits = depth_data[d]["qubits"]
            variances = depth_data[d]["variances"]
            # Sort by qubit count
            sorted_pairs = sorted(zip(qubits, variances))
            q_sorted = [p[0] for p in sorted_pairs]
            v_sorted = [p[1] for p in sorted_pairs]

            # Use log scale for variance
            log_v = [np.log10(v) if v > 0 else -16 for v in v_sorted]

            ax.plot(
                q_sorted, log_v,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linewidth=2, markersize=8,
                label=f"Depth {d}",
            )

        ax.set_xlabel("Number of Qubits", fontsize=14)
        ax.set_ylabel(r"$\log_{10}$ Var($\partial C / \partial \theta$)", fontsize=14)
        ax.set_title("Barren Plateau Analysis: Gradient Variance vs Qubit Count", fontsize=15)
        ax.legend(fontsize=12, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(QUBITS)

        fig_path = os.path.join(fig_dir, "barren_plateau_analysis.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved: {fig_path}")

    except ImportError as e:
        print(f"WARNING: Could not generate figure ({e}). Install matplotlib.")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_all(output_dir, fig_dir, n_samples, dry_run=False):
    """Run gradient analysis for all configurations."""
    print("=" * 70)
    print("BARREN PLATEAU ANALYSIS -- Gradient Variance Study")
    print(f"Configurations: {TOTAL_CONFIGS} ({len(QUBITS)} qubit counts x {len(DEPTHS)} depths)")
    print(f"Qubits: {QUBITS}")
    print(f"Depths: {DEPTHS}")
    print(f"Samples per config: {n_samples}")
    print(f"Output directory: {output_dir}")
    print(f"Figure directory: {fig_dir}")
    print(f"Dry run: {dry_run}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    completed = 0
    failed = 0
    t_start = time.time()

    for task_id in range(TOTAL_CONFIGS):
        try:
            result = run_single_config(task_id, output_dir, n_samples, dry_run=dry_run)
            if result is not None:
                completed += 1
        except Exception as e:
            n_qubits, depth = GRID[task_id]
            print(f"  FAILED: n_qubits={n_qubits} depth={depth}: {e}")
            failed += 1

    elapsed = time.time() - t_start

    # Aggregate results and generate figure
    if not dry_run and completed > 0:
        print()
        print("-" * 70)
        print("Aggregating results and generating figure...")
        aggregate_results(output_dir, fig_dir)

    print("=" * 70)
    print("BARREN PLATEAU ANALYSIS COMPLETE")
    print(f"  Completed: {completed}")
    print(f"  Failed:    {failed}")
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 70)

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Barren plateau analysis via gradient variance (reviewer issue #5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task-id", type=int, default=None,
        help=f"SLURM array job index (0-{TOTAL_CONFIGS - 1})",
    )
    parser.add_argument(
        "--run-all", action="store_true",
        help="Run all configurations sequentially",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/gradient",
        help="Directory for output files (default: results/gradient)",
    )
    parser.add_argument(
        "--fig-dir", type=str, default="results/figures",
        help="Directory for figures (default: results/figures)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES,
        help=f"Number of random initializations per config (default: {DEFAULT_N_SAMPLES})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configurations without executing them",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.task_id is None and not args.run_all:
        parser.error("Specify either --task-id N or --run-all")

    if args.task_id is not None and args.run_all:
        parser.error("--task-id and --run-all are mutually exclusive")

    if args.run_all:
        return run_all(
            args.output_dir, args.fig_dir,
            n_samples=args.n_samples, dry_run=args.dry_run,
        )
    else:
        result = run_single_config(
            args.task_id, args.output_dir,
            n_samples=args.n_samples, dry_run=args.dry_run,
        )
        # If single task, also try to aggregate if all results exist
        if result is not None:
            all_exist = all(
                os.path.isfile(os.path.join(args.output_dir, f"bp_{q}q_depth{d}.csv"))
                for q, d in GRID
            )
            if all_exist:
                print("\nAll configs complete. Aggregating results...")
                aggregate_results(args.output_dir, args.fig_dir)
        return 0 if result is not None else 1


if __name__ == "__main__":
    sys.exit(main())
