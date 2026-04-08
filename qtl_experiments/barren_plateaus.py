#!/usr/bin/env python3
"""Barren plateau analysis: gradient variance vs qubits/depth (M2).

Measures gradient variance via parameter-shift rule for the PennyLane VQC
across different qubit counts and circuit depths.

Usage:
    python barren_plateaus.py --config config.yaml
    python barren_plateaus.py --config config.yaml --task-id 0  # SLURM single config
"""

import argparse
import csv
import itertools
import os
import time

import numpy as np
import pennylane as qml
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def create_circuit(n_qubits, depth):
    """Create the same VQC architecture used in the main experiments."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit


def compute_gradient_variance(n_qubits, depth, n_initializations=200):
    """Compute gradient variance for a given (n_qubits, depth) configuration.

    Returns:
        dict with per-parameter gradient variance and mean
    """
    circuit = create_circuit(n_qubits, depth)
    n_weight_params = depth * n_qubits

    all_grads = []  # shape will be (n_init, n_params)

    for init_idx in range(n_initializations):
        # Random inputs (simulating backbone features projected to qubits)
        inputs = np.random.uniform(-np.pi / 2, np.pi / 2, size=n_qubits)
        # Random weights
        weights = np.random.uniform(0, 2 * np.pi, size=(depth, n_qubits))

        # Compute gradient w.r.t. weights via parameter-shift
        grad_fn = qml.grad(lambda w: float(np.sum(circuit(inputs, w))))
        try:
            grads = grad_fn(weights)
            all_grads.append(grads.flatten())
        except Exception:
            continue

    if len(all_grads) < 10:
        return None

    all_grads = np.array(all_grads)  # (n_init, n_params)

    # Per-parameter statistics
    grad_var_per_param = np.var(all_grads, axis=0)
    grad_mean_per_param = np.mean(all_grads, axis=0)

    return {
        "n_qubits": n_qubits,
        "depth": depth,
        "n_initializations": len(all_grads),
        "mean_grad_variance": float(np.mean(grad_var_per_param)),
        "max_grad_variance": float(np.max(grad_var_per_param)),
        "min_grad_variance": float(np.min(grad_var_per_param)),
        "mean_grad_mean": float(np.mean(np.abs(grad_mean_per_param))),
        "per_param_variance": grad_var_per_param.tolist(),
        "per_param_mean": grad_mean_per_param.tolist(),
    }


BP_CSV_FIELDS = [
    "n_qubits", "depth", "n_initializations",
    "mean_grad_variance", "max_grad_variance", "min_grad_variance",
    "mean_grad_mean",
]

BP_DETAIL_FIELDS = [
    "n_qubits", "depth", "param_idx", "grad_variance", "grad_mean",
]


def main():
    parser = argparse.ArgumentParser(description="Barren Plateau Analysis")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    bp_cfg = config.get("barren_plateaus", {})
    if not bp_cfg.get("enabled"):
        print("Barren plateau analysis not enabled in config.")
        return

    qubits = bp_cfg.get("qubits", [2, 4, 6])
    depths = bp_cfg.get("depths", [1, 2, 3, 4, 5])
    n_init = bp_cfg.get("n_initializations", 200)
    output_dir = args.output_dir or os.path.join(config.get("output_dir", "./results"), "barren_plateaus")
    os.makedirs(output_dir, exist_ok=True)

    configs = list(itertools.product(qubits, depths))

    if args.task_id is not None:
        if args.task_id >= len(configs):
            print(f"task-id {args.task_id} out of range (max {len(configs)-1})")
            return
        configs = [configs[args.task_id]]

    summary_csv = os.path.join(output_dir, "bp_summary.csv")
    detail_csv = os.path.join(output_dir, "bp_detail.csv")

    for nq, dep in configs:
        print(f"[BP] n_qubits={nq}, depth={dep}, n_init={n_init} ...")
        t0 = time.perf_counter()
        result = compute_gradient_variance(nq, dep, n_init)
        elapsed = time.perf_counter() - t0

        if result is None:
            print(f"  FAILED (not enough valid gradients)")
            continue

        print(f"  mean_var={result['mean_grad_variance']:.6e}, time={elapsed:.1f}s")

        # Append summary
        write_header = not os.path.exists(summary_csv) or os.path.getsize(summary_csv) == 0
        with open(summary_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=BP_CSV_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow({k: result[k] for k in BP_CSV_FIELDS})

        # Append per-parameter detail
        write_header = not os.path.exists(detail_csv) or os.path.getsize(detail_csv) == 0
        with open(detail_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=BP_DETAIL_FIELDS)
            if write_header:
                w.writeheader()
            for pidx, (var, mean) in enumerate(
                zip(result["per_param_variance"], result["per_param_mean"])
            ):
                w.writerow({
                    "n_qubits": nq, "depth": dep,
                    "param_idx": pidx,
                    "grad_variance": f"{var:.8e}",
                    "grad_mean": f"{mean:.8e}",
                })

    print(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    main()
