#!/usr/bin/env python3
"""Statistical analysis pipeline for QTL revision.

Reads the 3 CSV outputs and produces:
1. Main results table (mean +/- std per approach/backbone/dataset)
2. Significance tests (Welch t-test / Mann-Whitney U, Bonferroni)
3. Energy comparison table
4. Ablation table (accuracy vs qubits/depth)
5. Noise decomposition table
6. LaTeX tables ready to paste into the manuscript
7. Publication figures (300 DPI)

Usage:
    python analyze_results.py --results-dir ./results
    python analyze_results.py --results-dir ./results --latex  # also generate LaTeX
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_runs(results_dir):
    path = os.path.join(results_dir, "runs.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No runs.csv in {results_dir}")
    df = pd.read_csv(path)
    # Parse numeric columns
    for col in ["test_accuracy", "test_precision", "test_recall", "test_f1",
                 "test_auc", "train_time_s", "energy_kwh", "n_trainable_params"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_training_log(results_dir):
    path = os.path.join(results_dir, "training_log.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_bp(results_dir):
    path = os.path.join(results_dir, "barren_plateaus", "bp_summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Table 1: Main results (C1 — mean +/- std)
# ---------------------------------------------------------------------------

def main_results_table(df):
    """Aggregate metrics by (head, backbone, dataset)."""
    group_cols = ["head", "backbone", "dataset"]
    # For quantum heads, also group by n_qubits and depth
    if "n_qubits" in df.columns:
        group_cols_q = group_cols + ["n_qubits", "depth"]
    else:
        group_cols_q = group_cols

    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc"]
    available = [m for m in metrics if m in df.columns]

    agg = {}
    for m in available:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
    agg["n_seeds"] = ("seed", "count")
    agg["train_time_mean"] = ("train_time_s", "mean")

    # Use the finer grouping only where quantum columns exist
    quantum_mask = df["head_type"].isin(["pennylane", "qiskit"])
    classical = df[~quantum_mask].groupby(group_cols, dropna=False).agg(**agg).reset_index()
    quantum = df[quantum_mask].groupby(group_cols_q, dropna=False).agg(**agg).reset_index()

    return pd.concat([classical, quantum], ignore_index=True)


# ---------------------------------------------------------------------------
# Table 2: Significance tests (C1)
# ---------------------------------------------------------------------------

def significance_tests(df):
    """Pairwise significance between each quantum head and mlp_a (matched baseline).

    For each (backbone, dataset, n_qubits, depth):
      compare quantum head accuracy vs mlp_a accuracy.
    """
    results = []
    mlp_a = df[df["head"] == "mlp_a"]
    quantum_heads = df[df["head_type"].isin(["pennylane", "qiskit"])]

    for (bb, ds), mlp_group in mlp_a.groupby(["backbone", "dataset"]):
        baseline_accs = mlp_group["test_accuracy"].dropna().values
        if len(baseline_accs) < 3:
            continue

        q_subset = quantum_heads[(quantum_heads["backbone"] == bb) &
                                  (quantum_heads["dataset"] == ds)]

        for (head, nq, dep), q_group in q_subset.groupby(["head", "n_qubits", "depth"]):
            q_accs = q_group["test_accuracy"].dropna().values
            if len(q_accs) < 3:
                continue

            # Normality test
            _, p_normal_base = stats.shapiro(baseline_accs) if len(baseline_accs) >= 3 else (0, 0)
            _, p_normal_q = stats.shapiro(q_accs) if len(q_accs) >= 3 else (0, 0)

            if p_normal_base > 0.05 and p_normal_q > 0.05:
                test_name = "welch_t"
                stat, p_val = stats.ttest_ind(q_accs, baseline_accs, equal_var=False)
            else:
                test_name = "mann_whitney"
                stat, p_val = stats.mannwhitneyu(q_accs, baseline_accs, alternative="two-sided")

            results.append({
                "backbone": bb, "dataset": ds,
                "head": head, "n_qubits": nq, "depth": dep,
                "test": test_name,
                "statistic": stat,
                "p_value": p_val,
                "q_mean": np.mean(q_accs),
                "q_std": np.std(q_accs),
                "baseline_mean": np.mean(baseline_accs),
                "baseline_std": np.std(baseline_accs),
                "delta": np.mean(q_accs) - np.mean(baseline_accs),
            })

    if not results:
        return pd.DataFrame()

    sig_df = pd.DataFrame(results)
    # Bonferroni correction
    n_tests = len(sig_df)
    sig_df["p_bonferroni"] = (sig_df["p_value"] * n_tests).clip(upper=1.0)
    sig_df["significance"] = sig_df["p_bonferroni"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    )
    return sig_df


# ---------------------------------------------------------------------------
# Table 3: Energy comparison (C3)
# ---------------------------------------------------------------------------

def energy_table(df):
    """Mean energy per head type."""
    if "energy_kwh" not in df.columns or df["energy_kwh"].isna().all():
        return pd.DataFrame()
    return df.groupby(["head", "head_type"]).agg(
        energy_mean=("energy_kwh", "mean"),
        energy_std=("energy_kwh", "std"),
        time_mean=("train_time_s", "mean"),
        time_std=("train_time_s", "std"),
        n_runs=("run_id", "count"),
    ).reset_index()


# ---------------------------------------------------------------------------
# Table 4: Ablation (M4)
# ---------------------------------------------------------------------------

def ablation_table(df):
    """Accuracy heatmap data: mean accuracy by (n_qubits, depth)."""
    quantum = df[df["head_type"].isin(["pennylane", "qiskit"])].copy()
    if quantum.empty:
        return pd.DataFrame()
    return quantum.groupby(["n_qubits", "depth", "head"]).agg(
        accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()


# ---------------------------------------------------------------------------
# Table 5: Noise decomposition (M8)
# ---------------------------------------------------------------------------

def noise_table(df):
    """Accuracy per noise channel."""
    noise_runs = df[df["run_id"].str.startswith("noise_")]
    if noise_runs.empty:
        return pd.DataFrame()

    # Extract channel name from head name (pl_noise_<channel>)
    noise_runs = noise_runs.copy()
    noise_runs["channel"] = noise_runs["head"].str.replace("pl_noise_", "", regex=False)
    return noise_runs.groupby(["channel", "dataset"]).agg(
        accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def to_latex_main(main_df, output_path):
    """Generate LaTeX table for main results."""
    with open(output_path, "w") as f:
        f.write("% Auto-generated by analyze_results.py\n")
        f.write("% Main results table (mean ± std)\n\n")

        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Main experimental results (mean $\\pm$ std over 5 seeds)}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{llllcccccc}\n")
        f.write("\\toprule\n")
        f.write("Head & Backbone & Dataset & Qubits & Depth & Accuracy & Precision & Recall & F1 & AUC \\\\\n")
        f.write("\\midrule\n")

        for _, row in main_df.iterrows():
            nq = int(row["n_qubits"]) if pd.notna(row.get("n_qubits")) else "--"
            dep = int(row["depth"]) if pd.notna(row.get("depth")) else "--"

            cols = []
            for m in ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc"]:
                mean_col = f"{m}_mean"
                std_col = f"{m}_std"
                if mean_col in row and pd.notna(row[mean_col]):
                    cols.append(f"${row[mean_col]:.3f} \\pm {row.get(std_col, 0):.3f}$")
                else:
                    cols.append("--")

            f.write(f"{row['head']} & {row['backbone']} & {row['dataset']} & "
                    f"{nq} & {dep} & {' & '.join(cols)} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}}\n")
        f.write("\\end{table}\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_ablation_heatmap(abl_df, output_dir):
    if not HAS_PLOT or abl_df.empty:
        return
    for head_name, hdf in abl_df.groupby("head"):
        pivot = hdf.pivot_table(index="n_qubits", columns="depth",
                                values="accuracy_mean", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    vmin=0.4, vmax=1.0)
        ax.set_title(f"Ablation: Accuracy vs Qubits/Depth ({head_name})")
        ax.set_xlabel("Circuit Depth")
        ax.set_ylabel("Number of Qubits")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"ablation_heatmap_{head_name}.png"), dpi=300)
        plt.close(fig)


def plot_convergence_curves(log_df, output_dir, max_runs=12):
    """Plot training curves for a representative subset of runs."""
    if not HAS_PLOT or log_df is None or log_df.empty:
        return
    run_ids = log_df["run_id"].unique()[:max_runs]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for rid in run_ids:
        sub = log_df[log_df["run_id"] == rid]
        label = rid.replace("_", " ")[:30]
        axes[0].plot(sub["epoch"], sub["train_loss"], alpha=0.7, label=label)
        axes[1].plot(sub["epoch"], sub["val_accuracy"], alpha=0.7, label=label)

    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Convergence Curves (representative runs)")
    axes[0].legend(fontsize=6, ncol=3)
    axes[1].set_ylabel("Val Accuracy")
    axes[1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "convergence_curves.png"), dpi=300)
    plt.close(fig)


def plot_energy_comparison(energy_df, output_dir):
    if not HAS_PLOT or energy_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    energy_df_sorted = energy_df.sort_values("energy_mean")
    bars = ax.barh(energy_df_sorted["head"], energy_df_sorted["energy_mean"],
                   xerr=energy_df_sorted["energy_std"], capsize=3, color="steelblue")
    ax.set_xlabel("Energy (kWh)")
    ax.set_title("Energy Consumption per Head Type")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "energy_comparison.png"), dpi=300)
    plt.close(fig)


def plot_noise_decomposition(noise_df, output_dir):
    if not HAS_PLOT or noise_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for ds, sub in noise_df.groupby("dataset"):
        sub_sorted = sub.sort_values("accuracy_mean", ascending=False)
        ax.barh([f"{c} ({ds})" for c in sub_sorted["channel"]],
                sub_sorted["accuracy_mean"],
                xerr=sub_sorted["accuracy_std"], capsize=3, alpha=0.8, label=ds)
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Noise Decomposition: Impact of Individual Noise Channels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "noise_decomposition.png"), dpi=300)
    plt.close(fig)


def plot_barren_plateaus(bp_df, output_dir):
    if not HAS_PLOT or bp_df is None or bp_df.empty:
        return
    pivot = bp_df.pivot_table(index="n_qubits", columns="depth",
                               values="mean_grad_variance", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2e", cmap="viridis_r", ax=ax)
    ax.set_title("Barren Plateaus: Gradient Variance vs Qubits/Depth")
    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("Number of Qubits")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "barren_plateaus_heatmap.png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QTL Results Analysis")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = os.path.join(results_dir, "analysis")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("Loading data...")
    df = load_runs(results_dir)
    log_df = load_training_log(results_dir)
    bp_df = load_bp(results_dir)

    print(f"  runs.csv: {len(df)} runs")
    if log_df is not None:
        print(f"  training_log.csv: {len(log_df)} rows")
    if bp_df is not None:
        print(f"  bp_summary.csv: {len(bp_df)} configs")

    # Table 1: Main results
    print("\nGenerating main results table...")
    main_df = main_results_table(df)
    main_df.to_csv(os.path.join(output_dir, "main_results.csv"), index=False)
    print(f"  Saved: {len(main_df)} aggregated rows")

    # Table 2: Significance
    print("Running significance tests...")
    sig_df = significance_tests(df)
    if not sig_df.empty:
        sig_df.to_csv(os.path.join(output_dir, "significance_tests.csv"), index=False)
        n_sig = (sig_df["significance"] != "ns").sum()
        print(f"  {len(sig_df)} tests, {n_sig} significant after Bonferroni")
    else:
        print("  Skipped (not enough data)")

    # Table 3: Energy
    print("Generating energy table...")
    e_df = energy_table(df)
    if not e_df.empty:
        e_df.to_csv(os.path.join(output_dir, "energy_comparison.csv"), index=False)
        print(f"  {len(e_df)} rows")

    # Table 4: Ablation
    print("Generating ablation table...")
    abl_df = ablation_table(df)
    if not abl_df.empty:
        abl_df.to_csv(os.path.join(output_dir, "ablation_table.csv"), index=False)
        print(f"  {len(abl_df)} rows")

    # Table 5: Noise decomposition
    print("Generating noise decomposition table...")
    n_df = noise_table(df)
    if not n_df.empty:
        n_df.to_csv(os.path.join(output_dir, "noise_decomposition.csv"), index=False)
        print(f"  {len(n_df)} rows")

    # LaTeX
    if args.latex:
        print("\nGenerating LaTeX tables...")
        latex_dir = os.path.join(output_dir, "latex")
        os.makedirs(latex_dir, exist_ok=True)
        to_latex_main(main_df, os.path.join(latex_dir, "main_results.tex"))
        print("  Saved LaTeX tables")

    # Figures
    print("\nGenerating figures...")
    plot_ablation_heatmap(abl_df, figures_dir)
    plot_convergence_curves(log_df, figures_dir)
    plot_energy_comparison(e_df, figures_dir)
    plot_noise_decomposition(n_df, figures_dir)
    plot_barren_plateaus(bp_df, figures_dir)
    print(f"  Figures saved to {figures_dir}/")

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    heads = df["head"].unique()
    print(f"Heads tested: {', '.join(sorted(heads))}")
    print(f"Backbones: {', '.join(sorted(df['backbone'].unique()))}")
    print(f"Datasets: {', '.join(sorted(df['dataset'].unique()))}")
    print(f"Seeds per config: {df.groupby(['head', 'backbone', 'dataset'])['seed'].nunique().mode().values[0] if len(df) > 0 else 0}")
    print(f"\nAll outputs in: {output_dir}/")
    print(f"Figures in:     {figures_dir}/")


if __name__ == "__main__":
    main()
