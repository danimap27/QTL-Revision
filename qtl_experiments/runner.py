#!/usr/bin/env python3
"""Orchestrator: reads config.yaml, generates all run combinations, launches training.

Usage:
    python runner.py --config config.yaml                           # run all
    python runner.py --config config.yaml --dry-run                 # list runs
    python runner.py --config config.yaml --count                   # count only
    python runner.py --config config.yaml --task-id 0               # SLURM single task
    python runner.py --config config.yaml --dataset hymenoptera     # filter
    python runner.py --config config.yaml --head pl_ideal --seed 42
    python runner.py --config config.yaml --extensions-only         # only SPSA/ablation/noise
"""

import argparse
import copy
import itertools
import os
import sys
import traceback
import csv
from datetime import datetime

import yaml

from trainer import train_and_evaluate


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Run generation
# ---------------------------------------------------------------------------

def generate_main_runs(config, filters=None):
    """Generate main grid: (dataset x backbone x head x seed).

    For quantum heads, expands across qubit_configs x depth_configs.
    For classical heads, generates one run per (dataset, backbone, seed).
    """
    filters = filters or {}
    datasets = config["datasets"]
    backbones = config["backbones"]
    heads = config["heads"]
    seeds = config["seeds"]
    qubit_configs = config.get("qubit_configs", [4])
    depth_configs = config.get("depth_configs", [3])

    if filters.get("dataset"):
        datasets = [d for d in datasets if d["name"] in filters["dataset"]]
    if filters.get("backbone"):
        backbones = [b for b in backbones if b["name"] in filters["backbone"]]
    if filters.get("head"):
        heads = [h for h in heads if h["name"] in filters["head"]]
    if filters.get("seed"):
        seeds = [s for s in seeds if s in filters["seed"]]

    runs = []
    for ds, bb, hd, seed in itertools.product(datasets, backbones, heads, seeds):
        if hd["type"] == "classical":
            # Classical: no qubit/depth expansion
            run_id = f"{ds['name']}_{bb['name']}_{hd['name']}_{seed}"
            runs.append({
                "run_id": run_id,
                "seed": seed,
                "dataset": ds,
                "backbone": bb,
                "head": dict(hd),
            })
        else:
            # Quantum: expand across qubit x depth grid
            for nq, dep in itertools.product(qubit_configs, depth_configs):
                head_expanded = dict(hd)
                head_expanded["n_qubits"] = nq
                head_expanded["depth"] = dep
                run_id = f"{ds['name']}_{bb['name']}_{hd['name']}_{nq}q_d{dep}_{seed}"
                runs.append({
                    "run_id": run_id,
                    "seed": seed,
                    "dataset": ds,
                    "backbone": bb,
                    "head": head_expanded,
                })
    return runs


def generate_spsa_runs(config):
    """Generate SPSA optimizer control runs (M1)."""
    spsa = config.get("spsa_control", {})
    if not spsa.get("enabled"):
        return []

    runs = []
    ds_names = spsa["datasets"]
    bb_names = spsa["backbones"]
    nq = spsa.get("n_qubits", 4)
    dep = spsa.get("depth", 3)
    seeds = spsa.get("seeds", config["seeds"])

    datasets = [d for d in config["datasets"] if d["name"] in ds_names]
    backbones = [b for b in config["backbones"] if b["name"] in bb_names]

    for ds, bb, seed in itertools.product(datasets, backbones, seeds):
        run_id = f"{ds['name']}_{bb['name']}_pl_spsa_{nq}q_d{dep}_{seed}"
        runs.append({
            "run_id": run_id,
            "seed": seed,
            "dataset": ds,
            "backbone": bb,
            "head": {
                "name": "pl_spsa",
                "type": "pennylane",
                "backend": "default.qubit",
                "n_qubits": nq,
                "depth": dep,
                "noise": False,
                "optimizer_override": "spsa",
            },
        })
    return runs


def generate_ablation_runs(config):
    """Generate fine-grained ablation runs (M4)."""
    abl = config.get("ablation", {})
    if not abl.get("enabled"):
        return []

    runs = []
    ds_names = abl["datasets"]
    bb_names = abl["backbones"]
    hd_names = abl["heads"]
    qubits = abl["qubits"]
    depths = abl["depths"]
    seeds = abl.get("seeds", config["seeds"])

    datasets = [d for d in config["datasets"] if d["name"] in ds_names]
    backbones = [b for b in config["backbones"] if b["name"] in bb_names]
    heads = [h for h in config["heads"] if h["name"] in hd_names]

    for ds, bb, hd, nq, dep, seed in itertools.product(
        datasets, backbones, heads, qubits, depths, seeds
    ):
        # Skip combos already in main grid
        main_qubits = config.get("qubit_configs", [4])
        main_depths = config.get("depth_configs", [3])
        if nq in main_qubits and dep in main_depths:
            continue

        head_expanded = dict(hd)
        head_expanded["n_qubits"] = nq
        head_expanded["depth"] = dep
        run_id = f"abl_{ds['name']}_{bb['name']}_{hd['name']}_{nq}q_d{dep}_{seed}"
        runs.append({
            "run_id": run_id,
            "seed": seed,
            "dataset": ds,
            "backbone": bb,
            "head": head_expanded,
        })
    return runs


def generate_noise_decomp_runs(config):
    """Generate noise decomposition runs (M8)."""
    nd = config.get("noise_decomposition", {})
    if not nd.get("enabled"):
        return []

    runs = []
    ds_names = nd["datasets"]
    bb_names = nd["backbones"]
    nq = nd.get("n_qubits", 4)
    dep = nd.get("depth", 3)
    channels = nd["channels"]
    seeds = nd.get("seeds", config["seeds"])

    datasets = [d for d in config["datasets"] if d["name"] in ds_names]
    backbones = [b for b in config["backbones"] if b["name"] in bb_names]

    for ds, bb, ch, seed in itertools.product(datasets, backbones, channels, seeds):
        ch_name = ch["name"]
        run_id = f"noise_{ds['name']}_{bb['name']}_{ch_name}_{nq}q_d{dep}_{seed}"
        runs.append({
            "run_id": run_id,
            "seed": seed,
            "dataset": ds,
            "backbone": bb,
            "head": {
                "name": f"pl_noise_{ch_name}",
                "type": "pennylane",
                "backend": "default.mixed" if ch.get("noise") else "default.qubit",
                "n_qubits": nq,
                "depth": dep,
                "noise": ch.get("noise", False),
                "noise_params": ch.get("noise_params", None),
            },
        })
    return runs


def generate_all_runs(config, filters=None, extensions_only=False):
    """Generate all runs: main grid + extensions."""
    runs = []
    if not extensions_only:
        runs += generate_main_runs(config, filters)
    runs += generate_spsa_runs(config)
    runs += generate_ablation_runs(config)
    runs += generate_noise_decomp_runs(config)
    return runs


# ---------------------------------------------------------------------------
# Completed run tracking
# ---------------------------------------------------------------------------

def get_completed_run_ids(output_dir):
    runs_csv = os.path.join(output_dir, "runs.csv")
    if not os.path.exists(runs_csv):
        return set()
    completed = set()
    with open(runs_csv, newline="") as f:
        for row in csv.DictReader(f):
            completed.add(row["run_id"])
    return completed


def log_error(run_id, exc, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "errors.log"), "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{datetime.now().isoformat()}] {run_id}\n")
        f.write(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QTL Experiment Runner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", nargs="+", help="Filter by dataset(s)")
    parser.add_argument("--backbone", nargs="+", help="Filter by backbone(s)")
    parser.add_argument("--head", nargs="+", help="Filter by head(s)")
    parser.add_argument("--seed", type=int, nargs="+", help="Filter by seed(s)")
    parser.add_argument("--task-id", type=int, default=None,
                        help="SLURM: run only this task index")
    parser.add_argument("--dry-run", action="store_true",
                        help="List all runs without executing")
    parser.add_argument("--count", action="store_true",
                        help="Print run counts and exit")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run even if already in runs.csv")
    parser.add_argument("--extensions-only", action="store_true",
                        help="Only run extension experiments (SPSA, ablation, noise)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config.get("output_dir", "./results")

    filters = {}
    if args.dataset:
        filters["dataset"] = args.dataset
    if args.backbone:
        filters["backbone"] = args.backbone
    if args.head:
        filters["head"] = args.head
    if args.seed:
        filters["seed"] = args.seed

    all_runs = generate_all_runs(config, filters, args.extensions_only)

    if args.count:
        main_runs = generate_main_runs(config, filters)
        spsa_runs = generate_spsa_runs(config)
        abl_runs = generate_ablation_runs(config)
        noise_runs = generate_noise_decomp_runs(config)
        print(f"Main grid:          {len(main_runs):>5}")
        print(f"  Classical:        {sum(1 for r in main_runs if r['head']['type'] == 'classical'):>5}")
        print(f"  Quantum:          {sum(1 for r in main_runs if r['head']['type'] != 'classical'):>5}")
        print(f"SPSA control (M1):  {len(spsa_runs):>5}")
        print(f"Ablation (M4):      {len(abl_runs):>5}")
        print(f"Noise decomp (M8):  {len(noise_runs):>5}")
        print(f"{'─'*30}")
        print(f"TOTAL:              {len(main_runs) + len(spsa_runs) + len(abl_runs) + len(noise_runs):>5}")
        return

    if args.dry_run:
        completed = get_completed_run_ids(output_dir) if not args.no_skip else set()
        for i, r in enumerate(all_runs):
            status = "[DONE]" if r["run_id"] in completed else "[PEND]"
            head = r["head"]
            q_info = f" {head.get('n_qubits','?')}q d{head.get('depth','?')}" if head["type"] != "classical" else ""
            print(f"  {i:4d} {status} {r['run_id']}")
        pending = sum(1 for r in all_runs if r["run_id"] not in completed)
        print(f"\nTotal: {len(all_runs)} | Pending: {pending} | Done: {len(all_runs) - pending}")
        return

    # Select runs to execute
    if args.task_id is not None:
        if args.task_id >= len(all_runs):
            print(f"[ERROR] task-id {args.task_id} out of range (max {len(all_runs)-1})")
            sys.exit(1)
        runs_to_execute = [all_runs[args.task_id]]
    else:
        runs_to_execute = all_runs

    completed = get_completed_run_ids(output_dir) if not args.no_skip else set()
    total = len(runs_to_execute)
    done = 0
    failed = 0

    for i, run_cfg in enumerate(runs_to_execute, 1):
        rid = run_cfg["run_id"]
        if rid in completed:
            print(f"[SKIP {i}/{total}] {rid}")
            continue

        print(f"[RUN {i}/{total}] {rid} — started")
        try:
            result = train_and_evaluate(run_cfg, config)
            acc = result["test_accuracy"]
            t = result["train_time_s"]
            print(f"[DONE {i}/{total}] {rid} — acc={acc:.4f}, time={t:.1f}s")
            done += 1
        except Exception as e:
            print(f"[FAIL {i}/{total}] {rid} — {e}")
            log_error(rid, e, output_dir)
            failed += 1

    print(f"\nFinished: {done} done, {failed} failed. Results in {output_dir}/")


if __name__ == "__main__":
    main()
