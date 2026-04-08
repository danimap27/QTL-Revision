#!/usr/bin/env python3
"""
analyze_results.py - Comprehensive Statistical Analysis Pipeline

Addresses CRITICAL reviewer issue #1: no statistical validation.

Reads CSV result files from:
  - results/seeds/       (repeated runs with different random seeds)
  - results/ablation/    (ablation study over n_qubits x depth)
  - results/gradient/    (barren plateau / gradient variance data)
  - results/noise/       (noise decomposition data)

Produces:
  1. Main results table with mean +/- std across seeds
  2. Pairwise significance tests (Shapiro-Wilk + Welch t / Mann-Whitney U + Bonferroni)
  3. Energy comparison table
  4. Ablation table
  5. LaTeX tables ready for the manuscript
  6. Publication-quality figures (300 DPI)

Usage:
    python analyze_results.py
    python analyze_results.py --results-dir results --format both
    python analyze_results.py --format latex
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers / CI
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRIC_COLS = [
    "test_accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "auc_roc_weighted",
    "train_time_s",
    "test_time_s",
    "energy_kwh",
]

# Fallback column name mappings so the pipeline works with CSVs from both
# run_paper_experiments.py and run_complete_benchmark.py.
COLUMN_ALIASES = {
    # From run_paper_experiments.py
    "Model_Type": "approach",
    "Backbone": "backbone",
    "Dataset": "dataset",
    "Test_Accuracy": "test_accuracy",
    "Train_Time_s": "train_time_s",
    "Test_Time_s": "test_time_s",
    # From run_complete_benchmark.py
    "architecture": "approach",
    "test_accuracy": "test_accuracy",
    "train_time": "train_time_s",
    "test_time": "test_time_s",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1": "f1_weighted",
    "auc": "auc_roc_weighted",
    "elapsed_time": "train_time_s",
}

# Pretty labels for approaches (used in figures / LaTeX)
APPROACH_LABELS = {
    "classical": "Classical (CC)",
    "Classical": "Classical (CC)",
    "pennylane": "PennyLane",
    "PennyLane_Standard": "PennyLane",
    "pennylane_noisy": "PennyLane-Noisy",
    "PennyLane_Noisy": "PennyLane-Noisy",
    "qiskit": "Qiskit",
    "Qiskit_Standard": "Qiskit",
    "qiskit_noisy": "Qiskit-Noisy",
    "Qiskit_Noisy": "Qiskit-Noisy",
    "ensemble_voting": "Ensemble (Vote)",
    "ensemble_stacking": "Ensemble (Stack)",
}

# Seaborn publication style
SNS_STYLE = {
    "style": "whitegrid",
    "context": "paper",
    "font_scale": 1.3,
    "rc": {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
    },
}

# Colour palette (colour-blind friendly)
PALETTE = sns.color_palette("colorblind", n_colors=10)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _setup_style():
    """Apply seaborn publication style globally."""
    sns.set_theme(**SNS_STYLE)


def _ensure_dirs(*dirs):
    """Create directories if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _significance_marker(p: float) -> str:
    """Return LaTeX-compatible significance marker."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def _fmt_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    """Format mean +/- std for plain text."""
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def _fmt_mean_std_latex(mean: float, std: float, decimals: int = 3) -> str:
    """Format mean +/- std for LaTeX."""
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def _pretty_approach(name: str) -> str:
    """Map raw approach name to a human-readable label."""
    return APPROACH_LABELS.get(name, name)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using COLUMN_ALIASES."""
    rename_map = {}
    for old, new in COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            rename_map[old] = new
    df = df.rename(columns=rename_map)

    # Ensure approach column exists
    if "approach" not in df.columns:
        for candidate in ("model_type", "architecture", "Model_Type"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "approach"})
                break

    # Lower-case approach values for consistency
    if "approach" in df.columns:
        df["approach"] = df["approach"].astype(str).str.strip()

    return df


def _load_csvs_from_dir(directory: str) -> pd.DataFrame:
    """Recursively load all CSV files under *directory* into one DataFrame."""
    frames = []
    directory = Path(directory)
    if not directory.exists():
        return pd.DataFrame()
    for csv_path in sorted(directory.rglob("*.csv")):
        try:
            tmp = pd.read_csv(csv_path)
            tmp["_source_file"] = str(csv_path)
            frames.append(tmp)
        except Exception as exc:
            print(f"  [WARN] Could not read {csv_path}: {exc}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return _normalise_columns(df)


def load_seed_results(results_dir: str) -> pd.DataFrame:
    """Load repeated-seed experiment results."""
    df = _load_csvs_from_dir(os.path.join(results_dir, "seeds"))
    if df.empty:
        print("  [INFO] No seed results found in results/seeds/. Trying root results CSV...")
        # Fallback: try to load from any CSVs at the root level or benchmark_results
        for fallback in [results_dir, os.path.join(results_dir, "..", "paper_results"),
                         os.path.join(results_dir, "..", "benchmark_results")]:
            df = _load_csvs_from_dir(fallback)
            if not df.empty:
                break
    # Filter out error rows
    if "Status" in df.columns:
        df = df[df["Status"] != "ERROR"]
    if "success" in df.columns:
        df = df[df["success"].astype(str).str.lower().isin(["true", "1", "yes"])]
    # Coerce metric columns to numeric
    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_ablation_results(results_dir: str) -> pd.DataFrame:
    """Load ablation study results."""
    df = _load_csvs_from_dir(os.path.join(results_dir, "ablation"))
    for col in ["n_qubits", "depth", "quantum_depth", "test_accuracy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Normalise depth column name
    if "quantum_depth" in df.columns and "depth" not in df.columns:
        df = df.rename(columns={"quantum_depth": "depth"})
    if "Quantum_Depth" in df.columns and "depth" not in df.columns:
        df = df.rename(columns={"Quantum_Depth": "depth"})
    if "N_Qubits" in df.columns and "n_qubits" not in df.columns:
        df = df.rename(columns={"N_Qubits": "n_qubits"})
    return df


def load_gradient_results(results_dir: str) -> pd.DataFrame:
    """Load barren plateau / gradient variance data."""
    bp_path = os.path.join(results_dir, "gradient", "bp_variance.csv")
    if os.path.isfile(bp_path):
        try:
            df = pd.read_csv(bp_path)
            for col in ["n_qubits", "depth", "variance", "log_variance", "mean_gradient"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as exc:
            print(f"  [WARN] Could not read {bp_path}: {exc}")
    # Fallback: try loading all CSVs in gradient/
    return _load_csvs_from_dir(os.path.join(results_dir, "gradient"))


def load_noise_results(results_dir: str) -> pd.DataFrame:
    """Load noise decomposition results."""
    return _load_csvs_from_dir(os.path.join(results_dir, "noise"))


# ---------------------------------------------------------------------------
# 1. Main results table
# ---------------------------------------------------------------------------

def compute_main_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std of all metrics grouped by (approach, backbone, dataset)."""
    if df.empty:
        print("  [SKIP] No data for main results table.")
        return pd.DataFrame()

    group_cols = [c for c in ["approach", "backbone", "dataset"] if c in df.columns]
    if not group_cols:
        print("  [SKIP] Missing grouping columns (approach/backbone/dataset).")
        return pd.DataFrame()

    available_metrics = [m for m in METRIC_COLS if m in df.columns]
    if not available_metrics:
        print("  [SKIP] No recognised metric columns found.")
        return pd.DataFrame()

    agg_funcs = {}
    for m in available_metrics:
        agg_funcs[f"{m}_mean"] = pd.NamedAgg(column=m, aggfunc="mean")
        agg_funcs[f"{m}_std"] = pd.NamedAgg(column=m, aggfunc="std")
        agg_funcs[f"{m}_n"] = pd.NamedAgg(column=m, aggfunc="count")

    result = df.groupby(group_cols, dropna=False).agg(**agg_funcs).reset_index()
    return result


# ---------------------------------------------------------------------------
# 2. Significance tests
# ---------------------------------------------------------------------------

def compute_significance_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise significance tests between approaches on test_accuracy.

    For each (backbone, dataset) group:
      - All pairwise combinations of approaches
      - Shapiro-Wilk normality check on each sample (n >= 3)
      - Welch t-test if both normal, Mann-Whitney U otherwise
      - Bonferroni correction within each group
    """
    if df.empty or "test_accuracy" not in df.columns:
        print("  [SKIP] No data for significance tests.")
        return pd.DataFrame()

    group_cols = [c for c in ["backbone", "dataset"] if c in df.columns]
    if not group_cols:
        group_cols = ["dataset"] if "dataset" in df.columns else []
    if "approach" not in df.columns:
        print("  [SKIP] 'approach' column missing.")
        return pd.DataFrame()

    records = []

    for grp_key, grp_df in df.groupby(group_cols, dropna=False):
        if not isinstance(grp_key, tuple):
            grp_key = (grp_key,)

        approaches = sorted(grp_df["approach"].unique())
        n_pairs = len(list(combinations(approaches, 2)))
        if n_pairs == 0:
            continue

        bonferroni_factor = max(n_pairs, 1)

        for a1, a2 in combinations(approaches, 2):
            vals1 = grp_df.loc[grp_df["approach"] == a1, "test_accuracy"].dropna().values
            vals2 = grp_df.loc[grp_df["approach"] == a2, "test_accuracy"].dropna().values

            n1, n2 = len(vals1), len(vals2)
            if n1 < 2 or n2 < 2:
                continue

            # Normality tests (Shapiro-Wilk requires n >= 3)
            if n1 >= 3:
                sw1_stat, sw1_p = stats.shapiro(vals1)
            else:
                sw1_stat, sw1_p = np.nan, 1.0
            if n2 >= 3:
                sw2_stat, sw2_p = stats.shapiro(vals2)
            else:
                sw2_stat, sw2_p = np.nan, 1.0

            both_normal = (sw1_p > 0.05) and (sw2_p > 0.05)

            if both_normal:
                test_name = "Welch t-test"
                t_stat, p_raw = stats.ttest_ind(vals1, vals2, equal_var=False)
            else:
                test_name = "Mann-Whitney U"
                try:
                    u_stat, p_raw = stats.mannwhitneyu(
                        vals1, vals2, alternative="two-sided"
                    )
                    t_stat = u_stat
                except ValueError:
                    t_stat, p_raw = np.nan, 1.0

            p_corrected = min(p_raw * bonferroni_factor, 1.0)
            sig = _significance_marker(p_corrected)

            rec = {}
            for i, col in enumerate(group_cols):
                rec[col] = grp_key[i]
            rec.update(
                {
                    "approach_1": a1,
                    "approach_2": a2,
                    "n_1": n1,
                    "n_2": n2,
                    "mean_1": np.mean(vals1),
                    "mean_2": np.mean(vals2),
                    "std_1": np.std(vals1, ddof=1),
                    "std_2": np.std(vals2, ddof=1),
                    "shapiro_p_1": sw1_p,
                    "shapiro_p_2": sw2_p,
                    "both_normal": both_normal,
                    "test_used": test_name,
                    "statistic": t_stat,
                    "p_value_raw": p_raw,
                    "bonferroni_factor": bonferroni_factor,
                    "p_value_corrected": p_corrected,
                    "significance": sig,
                }
            )
            records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. Energy comparison
# ---------------------------------------------------------------------------

def compute_energy_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std energy per (approach, backbone, dataset)."""
    if df.empty or "energy_kwh" not in df.columns:
        print("  [SKIP] No energy data found.")
        return pd.DataFrame()

    group_cols = [c for c in ["approach", "backbone", "dataset"] if c in df.columns]
    if not group_cols:
        return pd.DataFrame()

    result = (
        df.groupby(group_cols, dropna=False)["energy_kwh"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "energy_kwh_mean", "std": "energy_kwh_std", "count": "n"})
    )
    return result


# ---------------------------------------------------------------------------
# 4. Ablation table
# ---------------------------------------------------------------------------

def compute_ablation_table(df_ablation: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy by (n_qubits, depth) with std."""
    if df_ablation.empty:
        print("  [SKIP] No ablation data found.")
        return pd.DataFrame()

    required = {"n_qubits", "depth", "test_accuracy"}
    # Also accept the column after normalisation
    if "test_accuracy" not in df_ablation.columns and "Test_Accuracy" in df_ablation.columns:
        df_ablation = df_ablation.rename(columns={"Test_Accuracy": "test_accuracy"})
        df_ablation["test_accuracy"] = pd.to_numeric(df_ablation["test_accuracy"], errors="coerce")

    if not required.issubset(df_ablation.columns):
        print(f"  [SKIP] Ablation data missing columns. Have: {list(df_ablation.columns)}")
        return pd.DataFrame()

    result = (
        df_ablation.groupby(["n_qubits", "depth"], dropna=False)["test_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy_mean", "std": "accuracy_std", "count": "n"})
    )
    return result


# ---------------------------------------------------------------------------
# 5. LaTeX tables
# ---------------------------------------------------------------------------

def _latex_escape(s: str) -> str:
    """Escape special LaTeX characters in a string."""
    for ch in ["_", "&", "%", "#"]:
        s = s.replace(ch, f"\\{ch}")
    return s


def generate_latex_tables(
    main_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    energy_df: pd.DataFrame,
    output_path: str,
):
    """Write all LaTeX table code to a single .tex file."""
    lines = []
    lines.append("% Auto-generated LaTeX tables from analyze_results.py")
    lines.append("% Paste these directly into your manuscript.\n")

    # --- Main results table ---
    lines.append("%" + "=" * 70)
    lines.append("% TABLE 1: Main Results (Mean +/- Std)")
    lines.append("%" + "=" * 70)

    if not main_df.empty:
        # Determine which metrics are available
        available = [m for m in METRIC_COLS if f"{m}_mean" in main_df.columns]
        # Build a condensed table: approach | backbone | dataset | metrics...
        header_metrics = " & ".join([_latex_escape(m) for m in available])
        n_cols = 3 + len(available)
        col_spec = "l l l " + " ".join(["c"] * len(available))

        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Main experimental results (mean $\\pm$ std over 5 seeds).}")
        lines.append("\\label{tab:main_results}")
        lines.append(f"\\resizebox{{\\textwidth}}{{!}}{{%")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        lines.append(f"Approach & Backbone & Dataset & {header_metrics} \\\\")
        lines.append("\\midrule")

        for _, row in main_df.iterrows():
            approach_str = _latex_escape(str(row.get("approach", "")))
            backbone_str = _latex_escape(str(row.get("backbone", "")))
            dataset_str = _latex_escape(str(row.get("dataset", "")))
            cells = [approach_str, backbone_str, dataset_str]
            for m in available:
                mean_val = row.get(f"{m}_mean", np.nan)
                std_val = row.get(f"{m}_std", np.nan)
                if pd.notna(mean_val):
                    dec = 4 if "time" not in m and "energy" not in m else 2
                    cells.append(_fmt_mean_std_latex(mean_val, std_val if pd.notna(std_val) else 0.0, dec))
                else:
                    cells.append("--")
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}}")
        lines.append("\\end{table}\n")
    else:
        lines.append("% [No main results data available]\n")

    # --- Significance table ---
    lines.append("%" + "=" * 70)
    lines.append("% TABLE 2: Pairwise Significance Tests")
    lines.append("%" + "=" * 70)

    if not sig_df.empty:
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Pairwise significance tests on test accuracy (Bonferroni-corrected). "
                      "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}")
        lines.append("\\label{tab:significance}")
        lines.append("\\resizebox{\\textwidth}{!}{%")
        lines.append("\\begin{tabular}{l l l l c c c c l}")
        lines.append("\\toprule")
        lines.append("Backbone & Dataset & Approach A & Approach B & Mean A & Mean B & $p$-corrected & Test & Sig. \\\\")
        lines.append("\\midrule")

        for _, row in sig_df.iterrows():
            bb = _latex_escape(str(row.get("backbone", "")))
            ds = _latex_escape(str(row.get("dataset", "")))
            a1 = _latex_escape(str(row.get("approach_1", "")))
            a2 = _latex_escape(str(row.get("approach_2", "")))
            m1 = f"{row.get('mean_1', 0):.4f}"
            m2 = f"{row.get('mean_2', 0):.4f}"
            pc = row.get("p_value_corrected", 1.0)
            if pc < 0.001:
                pc_str = "$<0.001$"
            else:
                pc_str = f"${pc:.4f}$"
            test_str = _latex_escape(str(row.get("test_used", "")))
            sig_str = row.get("significance", "")
            lines.append(
                f"{bb} & {ds} & {a1} & {a2} & {m1} & {m2} & {pc_str} & {test_str} & {sig_str} \\\\"
            )

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}}")
        lines.append("\\end{table}\n")
    else:
        lines.append("% [No significance test data available]\n")

    # --- Ablation table ---
    lines.append("%" + "=" * 70)
    lines.append("% TABLE 3: Ablation Study (n_qubits x depth)")
    lines.append("%" + "=" * 70)

    if not ablation_df.empty:
        depths = sorted(ablation_df["depth"].dropna().unique())
        qubits = sorted(ablation_df["n_qubits"].dropna().unique())

        depth_headers = " & ".join([f"$d={int(d)}$" for d in depths])
        col_spec = "c " + " ".join(["c"] * len(depths))

        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Ablation study: test accuracy by number of qubits and circuit depth.}")
        lines.append("\\label{tab:ablation}")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        lines.append(f"$n_{{qubits}}$ & {depth_headers} \\\\")
        lines.append("\\midrule")

        for q in qubits:
            cells = [f"${int(q)}$"]
            for d in depths:
                match = ablation_df[
                    (ablation_df["n_qubits"] == q) & (ablation_df["depth"] == d)
                ]
                if not match.empty:
                    row = match.iloc[0]
                    cells.append(
                        _fmt_mean_std_latex(
                            row["accuracy_mean"],
                            row["accuracy_std"] if pd.notna(row["accuracy_std"]) else 0.0,
                            3,
                        )
                    )
                else:
                    cells.append("--")
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}\n")
    else:
        lines.append("% [No ablation data available]\n")

    # --- Energy table ---
    lines.append("%" + "=" * 70)
    lines.append("% TABLE 4: Energy Comparison (kWh)")
    lines.append("%" + "=" * 70)

    if not energy_df.empty:
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Energy consumption per approach (kWh, mean $\\pm$ std).}")
        lines.append("\\label{tab:energy}")
        group_cols = [c for c in ["approach", "backbone", "dataset"] if c in energy_df.columns]
        n_gc = len(group_cols)
        col_spec = " ".join(["l"] * n_gc) + " c c"
        header = " & ".join([_latex_escape(c) for c in group_cols]) + " & Energy (kWh) & $n$ \\\\"

        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        lines.append(header)
        lines.append("\\midrule")

        for _, row in energy_df.iterrows():
            cells = [_latex_escape(str(row.get(c, ""))) for c in group_cols]
            m = row.get("energy_kwh_mean", np.nan)
            s = row.get("energy_kwh_std", np.nan)
            if pd.notna(m):
                cells.append(_fmt_mean_std_latex(m, s if pd.notna(s) else 0.0, 4))
            else:
                cells.append("--")
            cells.append(str(int(row.get("n", 0))))
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}\n")
    else:
        lines.append("% [No energy data available]\n")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [OK] LaTeX tables written to {output_path}")


# ---------------------------------------------------------------------------
# 6. Publication figures
# ---------------------------------------------------------------------------

def plot_ablation_heatmap(ablation_df: pd.DataFrame, figures_dir: str):
    """Heatmap of accuracy by (n_qubits, depth)."""
    if ablation_df.empty:
        print("  [SKIP] No ablation data for heatmap.")
        return

    pivot = ablation_df.pivot_table(
        index="n_qubits", columns="depth", values="accuracy_mean"
    )
    if pivot.empty:
        print("  [SKIP] Could not pivot ablation data.")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Test Accuracy"},
    )
    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("Number of Qubits")
    ax.set_title("Ablation Study: Accuracy by Qubits and Depth")
    fig.tight_layout()
    out_path = os.path.join(figures_dir, "ablation_heatmap.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {out_path}")


def plot_barren_plateau(gradient_df: pd.DataFrame, figures_dir: str):
    """Plot log(Var) vs n_qubits for each depth."""
    if gradient_df.empty:
        print("  [SKIP] No gradient data for barren plateau plot.")
        return

    # Determine variance column
    var_col = None
    for candidate in ["variance", "log_variance", "grad_variance"]:
        if candidate in gradient_df.columns:
            var_col = candidate
            break
    if var_col is None:
        print("  [SKIP] No variance column in gradient data.")
        return

    if "n_qubits" not in gradient_df.columns:
        print("  [SKIP] 'n_qubits' column missing in gradient data.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    depth_col = "depth" if "depth" in gradient_df.columns else None

    if depth_col:
        for depth, sub in gradient_df.groupby(depth_col):
            sub_sorted = sub.sort_values("n_qubits")
            y = sub_sorted[var_col].values
            # Compute log if not already log
            if var_col != "log_variance":
                y = np.log10(np.clip(y, 1e-30, None))
            ax.plot(
                sub_sorted["n_qubits"].values,
                y,
                marker="o",
                label=f"depth={int(depth)}",
            )
    else:
        sub_sorted = gradient_df.sort_values("n_qubits")
        y = sub_sorted[var_col].values
        if var_col != "log_variance":
            y = np.log10(np.clip(y, 1e-30, None))
        ax.plot(sub_sorted["n_qubits"].values, y, marker="o")

    ax.set_xlabel("Number of Qubits ($n$)")
    ax.set_ylabel("$\\log_{10}(\\mathrm{Var}[\\partial \\theta])$")
    ax.set_title("Barren Plateau Analysis: Gradient Variance vs. Qubits")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(figures_dir, "barren_plateau_variance.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {out_path}")


def plot_energy_comparison(energy_df: pd.DataFrame, figures_dir: str):
    """Grouped bar chart of energy per approach."""
    if energy_df.empty:
        print("  [SKIP] No energy data for bar chart.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    # Aggregate over backbone/dataset to get per-approach mean
    if "approach" in energy_df.columns:
        agg = (
            energy_df.groupby("approach")["energy_kwh_mean"]
            .mean()
            .sort_values()
            .reset_index()
        )
        agg["label"] = agg["approach"].apply(_pretty_approach)
        bars = ax.barh(agg["label"], agg["energy_kwh_mean"], color=PALETTE[:len(agg)])
        ax.set_xlabel("Energy (kWh)")
        ax.set_title("Energy Consumption by Approach")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    else:
        print("  [SKIP] 'approach' column missing in energy data.")
        plt.close(fig)
        return

    fig.tight_layout()
    out_path = os.path.join(figures_dir, "energy_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {out_path}")


def plot_convergence_curves(seed_df: pd.DataFrame, results_dir: str, figures_dir: str):
    """Plot representative training loss curves.

    Looks for per-epoch loss CSVs in results/seeds/ subdirectories,
    or falls back to showing available training history.
    """
    # Try to find training history files
    loss_dir = Path(results_dir) / "seeds"
    history_files = sorted(loss_dir.rglob("*history*.csv")) + sorted(loss_dir.rglob("*loss*.csv"))

    if not history_files:
        # Also try root results
        for alt in [Path(results_dir), Path(results_dir).parent / "paper_results"]:
            history_files = sorted(alt.rglob("*history*.csv")) + sorted(alt.rglob("*loss*.csv"))
            if history_files:
                break

    if not history_files:
        print("  [SKIP] No training history / loss CSVs found for convergence plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    # Pick up to 4 diverse files
    selected = history_files[: min(4, len(history_files))]

    for fpath in selected:
        try:
            hdf = pd.read_csv(fpath)
            # Find epoch and loss columns
            epoch_col = None
            loss_col = None
            for c in hdf.columns:
                cl = c.lower()
                if "epoch" in cl:
                    epoch_col = c
                if "loss" in cl and "val" not in cl:
                    loss_col = c
            if loss_col is None:
                continue
            if epoch_col is None:
                hdf["_epoch"] = range(1, len(hdf) + 1)
                epoch_col = "_epoch"
            label = Path(fpath).stem[:40]
            ax.plot(hdf[epoch_col], hdf[loss_col], label=label, alpha=0.8)
        except Exception:
            continue

    if not ax.get_lines():
        print("  [SKIP] Could not parse any convergence data.")
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Representative Convergence Curves")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(figures_dir, "convergence_curves.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {out_path}")


def plot_accuracy_boxplots(seed_df: pd.DataFrame, figures_dir: str):
    """Box plots of accuracy distribution per approach."""
    if seed_df.empty or "test_accuracy" not in seed_df.columns or "approach" not in seed_df.columns:
        print("  [SKIP] No data for accuracy box plots.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Order approaches by median accuracy
    order = (
        seed_df.groupby("approach")["test_accuracy"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    sns.boxplot(
        data=seed_df,
        x="approach",
        y="test_accuracy",
        order=order,
        palette="colorblind",
        ax=ax,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "red",
            "markeredgecolor": "black",
            "markersize": 6,
        },
    )
    # Overlay strip plot for individual points
    sns.stripplot(
        data=seed_df,
        x="approach",
        y="test_accuracy",
        order=order,
        color="black",
        alpha=0.35,
        size=3,
        jitter=True,
        ax=ax,
    )

    ax.set_xlabel("Approach")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy Distribution per Approach (all datasets/backbones)")
    ax.set_xticklabels([_pretty_approach(t.get_text()) for t in ax.get_xticklabels()], rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(figures_dir, "accuracy_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {out_path}")


def plot_noise_decomposition(noise_df: pd.DataFrame, figures_dir: str):
    """Bar chart showing accuracy impact of each noise component."""
    if noise_df.empty:
        print("  [SKIP] No noise decomposition data.")
        return

    # Expect columns like: noise_component, accuracy or accuracy_delta
    acc_col = None
    for candidate in ["accuracy", "test_accuracy", "accuracy_delta", "delta"]:
        if candidate in noise_df.columns:
            acc_col = candidate
            break
    comp_col = None
    for candidate in ["noise_component", "component", "noise_type", "type", "label"]:
        if candidate in noise_df.columns:
            comp_col = candidate
            break

    if acc_col is None or comp_col is None:
        # Fallback: compare noisy vs ideal approaches
        if "approach" in noise_df.columns and "test_accuracy" in noise_df.columns:
            comp_col = "approach"
            acc_col = "test_accuracy"
        else:
            print("  [SKIP] Cannot determine noise decomposition columns.")
            return

    agg = noise_df.groupby(comp_col)[acc_col].agg(["mean", "std"]).reset_index()
    agg = agg.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        agg[comp_col].astype(str),
        agg["mean"],
        xerr=agg["std"].fillna(0),
        color=PALETTE[: len(agg)],
        capsize=3,
    )
    ax.set_xlabel("Test Accuracy" if "delta" not in acc_col else "Accuracy Delta")
    ax.set_title("Noise Decomposition: Impact of Each Component")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(figures_dir, "noise_decomposition.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {out_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(results_dir: str, output_dir: str, figures_dir: str, fmt: str):
    """Execute the full analysis pipeline."""

    _setup_style()
    _ensure_dirs(output_dir, figures_dir)

    print("=" * 70)
    print("  STATISTICAL ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"  Results dir : {results_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Figures dir : {figures_dir}")
    print(f"  Format      : {fmt}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    seed_df = load_seed_results(results_dir)
    print(f"  Seed results : {len(seed_df)} rows")

    ablation_raw = load_ablation_results(results_dir)
    print(f"  Ablation data: {len(ablation_raw)} rows")

    gradient_df = load_gradient_results(results_dir)
    print(f"  Gradient data: {len(gradient_df)} rows")

    noise_df = load_noise_results(results_dir)
    print(f"  Noise data   : {len(noise_df)} rows")

    # ------------------------------------------------------------------
    # 1. Main results table
    # ------------------------------------------------------------------
    print("\n[2/6] Computing main results table...")
    main_df = compute_main_results(seed_df)
    if not main_df.empty and fmt in ("csv", "both"):
        out_path = os.path.join(output_dir, "main_results.csv")
        main_df.to_csv(out_path, index=False)
        print(f"  [OK] Saved {out_path} ({len(main_df)} rows)")

    # ------------------------------------------------------------------
    # 2. Significance tests
    # ------------------------------------------------------------------
    print("\n[3/6] Running significance tests...")
    sig_df = compute_significance_tests(seed_df)
    if not sig_df.empty and fmt in ("csv", "both"):
        out_path = os.path.join(output_dir, "significance_tests.csv")
        sig_df.to_csv(out_path, index=False)
        n_sig = (sig_df["significance"] != "").sum() if "significance" in sig_df.columns else 0
        print(f"  [OK] Saved {out_path} ({len(sig_df)} comparisons, {n_sig} significant)")

    # ------------------------------------------------------------------
    # 3. Energy comparison
    # ------------------------------------------------------------------
    print("\n[4/6] Computing energy comparison...")
    energy_df = compute_energy_comparison(seed_df)
    if not energy_df.empty and fmt in ("csv", "both"):
        out_path = os.path.join(output_dir, "energy_comparison.csv")
        energy_df.to_csv(out_path, index=False)
        print(f"  [OK] Saved {out_path}")

    # ------------------------------------------------------------------
    # 4. Ablation table
    # ------------------------------------------------------------------
    print("\n[5/6] Computing ablation table...")
    ablation_df = compute_ablation_table(ablation_raw)
    if not ablation_df.empty and fmt in ("csv", "both"):
        out_path = os.path.join(output_dir, "ablation_table.csv")
        ablation_df.to_csv(out_path, index=False)
        print(f"  [OK] Saved {out_path}")

    # ------------------------------------------------------------------
    # 5. LaTeX tables
    # ------------------------------------------------------------------
    if fmt in ("latex", "both"):
        print("\n[6a/6] Generating LaTeX tables...")
        latex_path = os.path.join(output_dir, "latex_tables.tex")
        generate_latex_tables(main_df, sig_df, ablation_df, energy_df, latex_path)

    # ------------------------------------------------------------------
    # 6. Publication figures
    # ------------------------------------------------------------------
    print("\n[6b/6] Generating publication figures...")
    plot_ablation_heatmap(ablation_df, figures_dir)
    plot_barren_plateau(gradient_df, figures_dir)
    plot_energy_comparison(energy_df, figures_dir)
    plot_convergence_curves(seed_df, results_dir, figures_dir)
    plot_accuracy_boxplots(seed_df, figures_dir)
    plot_noise_decomposition(noise_df, figures_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)

    outputs_created = []
    for dirpath in [output_dir, figures_dir]:
        for fname in sorted(os.listdir(dirpath)):
            fpath = os.path.join(dirpath, fname)
            if os.path.isfile(fpath):
                size_kb = os.path.getsize(fpath) / 1024
                outputs_created.append((fpath, size_kb))
    if outputs_created:
        print("  Generated files:")
        for fpath, size_kb in outputs_created:
            print(f"    {fpath} ({size_kb:.1f} KB)")
    else:
        print("  [WARN] No output files were generated. Run experiments first to populate results/seeds/, etc.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive statistical analysis pipeline for QTL review.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_results.py
  python analyze_results.py --results-dir results --format both
  python analyze_results.py --format latex --output-dir results/aggregated
        """,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base directory containing seeds/, ablation/, gradient/, noise/ subdirs (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/aggregated",
        help="Where to save aggregated CSV/LaTeX tables (default: results/aggregated)",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="results/figures",
        help="Where to save publication figures (default: results/figures)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "latex", "both"],
        default="both",
        help="Output format: csv, latex, or both (default: both)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        fmt=args.format,
    )


if __name__ == "__main__":
    main()
