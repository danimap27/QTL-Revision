#!/usr/bin/env python3
"""Multi-account SLURM distribution for Hercules.

Reads config.yaml, counts total runs, and distributes them across
multiple Hercules accounts as separate sbatch submissions.

Usage:
    python slurm/distribute.py --config config.yaml
    python slurm/distribute.py --config config.yaml --submit  # actually submit
"""

import argparse
import math
import os
import subprocess
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runner import load_config, generate_all_runs, get_completed_run_ids


def main():
    parser = argparse.ArgumentParser(description="Distribute SLURM jobs across accounts")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--submit", action="store_true", help="Actually submit (default: dry-run)")
    parser.add_argument("--max-concurrent", type=int, default=80, help="Max concurrent per account")
    args = parser.parse_args()

    config = load_config(args.config)
    all_runs = generate_all_runs(config)
    output_dir = config.get("output_dir", "./results")
    completed = get_completed_run_ids(output_dir)
    pending = [r for r in all_runs if r["run_id"] not in completed]

    accounts = config.get("hercules", {}).get("accounts", [])
    if not accounts:
        print("[ERROR] No accounts configured in config.yaml hercules.accounts")
        return

    resources = config.get("hercules", {}).get("default_resources", {})
    n_accounts = len(accounts)
    chunk_size = math.ceil(len(pending) / n_accounts)

    print(f"Total runs: {len(all_runs)}")
    print(f"Completed:  {len(completed)}")
    print(f"Pending:    {len(pending)}")
    print(f"Accounts:   {n_accounts}")
    print(f"Per account: ~{chunk_size}")
    print()

    # Create SLURM log directory
    os.makedirs("slurm_logs", exist_ok=True)

    # Map pending runs back to their global indices
    all_run_ids = [r["run_id"] for r in all_runs]
    pending_indices = [all_run_ids.index(r["run_id"]) for r in pending]

    for i, account in enumerate(accounts):
        chunk_indices = pending_indices[i * chunk_size: (i + 1) * chunk_size]
        if not chunk_indices:
            print(f"[{account['name']}] No tasks to submit")
            continue

        # Build array spec (ranges for efficiency)
        array_spec = _compress_indices(chunk_indices)
        max_conc = account.get("max_concurrent", args.max_concurrent)

        print(f"[{account['name']}] {len(chunk_indices)} tasks — array={array_spec}%{max_conc}")

        if args.submit:
            cmd = [
                "sbatch",
                f"--account={account.get('user', '')}",
                f"--partition={account.get('partition', 'gpu')}",
                f"--array={array_spec}%{max_conc}",
                f"--gres={resources.get('gres', 'gpu:a100:1')}",
                f"--mem={resources.get('mem', '32G')}",
                f"--cpus-per-task={resources.get('cpus_per_task', 8)}",
                f"--time={resources.get('time', '04:00:00')}",
                "slurm/run_experiments.sh",
            ]
            print(f"  CMD: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Submitted: {result.stdout.strip()}")
            else:
                print(f"  FAILED: {result.stderr.strip()}")
        else:
            print(f"  (dry-run, use --submit to actually submit)")

    if not args.submit:
        print("\nDry run complete. Add --submit to submit jobs.")


def _compress_indices(indices):
    """Compress [0,1,2,5,6,7,10] to '0-2,5-7,10'."""
    if not indices:
        return ""
    indices = sorted(indices)
    ranges = []
    start = indices[0]
    end = start
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = idx
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)


if __name__ == "__main__":
    main()
