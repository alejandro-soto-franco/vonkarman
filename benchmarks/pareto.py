#!/usr/bin/env python3
"""Pareto / speed-accuracy plotter for vonkarman benchmark CSVs.

Reads one or more per-repeat timing CSVs (schema: solver, backend, resolution,
repeat_index, walltime_s, steps, steps_per_s, end_time, notes) and an optional
per-run accuracy CSV, then produces:

  (a) A walltime distribution / speed table with error bars (mean +/- std across
      repeats). When accuracy data are absent the accuracy axis is left as a
      placeholder labelled "accuracy data not yet available".

  (b) An accuracy-vs-walltime scatter when accuracy rows are supplied via
      --accuracy-csv. Accuracy column is named "error" (lower is better).

Figures are saved to benchmarks/results/figures/.

Usage:
    uv run python benchmarks/pareto.py timing_vonkarman_cpu_n64.csv [more.csv ...]
    uv run python benchmarks/pareto.py *.csv --accuracy-csv accuracy.csv
    uv run python benchmarks/pareto.py --results-dir benchmarks/results

The script discovers all timing_*.csv files in --results-dir when no positional
CSV paths are given.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="*", help="Timing CSV files (one row per repeat)")
    p.add_argument("--results-dir", default=RESULTS_DIR,
                   help="Directory to scan for timing_*.csv when no CSVs given")
    p.add_argument("--accuracy-csv", default=None,
                   help="CSV with accuracy metrics (columns: solver, backend, resolution, error)")
    p.add_argument("--figures-dir", default=FIGURES_DIR,
                   help="Output directory for PNG figures")
    p.add_argument("--no-show", action="store_true", default=True,
                   help="Do not display figures interactively (default on headless)")
    return p.parse_args()


def discover_csvs(results_dir: str) -> list[str]:
    """Find all timing_*.csv files in results_dir."""
    if not os.path.isdir(results_dir):
        return []
    found = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("timing_") and fname.endswith(".csv"):
            found.append(os.path.join(results_dir, fname))
    return found


def load_timing_csvs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not os.path.exists(p):
            print(f"WARNING: {p} not found, skipping", file=sys.stderr)
            continue
        df = pd.read_csv(p)
        # Validate required columns.
        required = {"solver", "backend", "resolution", "repeat_index", "walltime_s"}
        missing = required - set(df.columns)
        if missing:
            print(f"WARNING: {p} missing columns {missing}, skipping", file=sys.stderr)
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (solver, backend, resolution), compute mean/std of walltime."""
    # Drop rows that failed (NaN walltime).
    valid = df.dropna(subset=["walltime_s"])
    if valid.empty:
        return pd.DataFrame()

    grouped = valid.groupby(["solver", "backend", "resolution"])
    agg = grouped["walltime_s"].agg(
        walltime_mean="mean",
        walltime_std="std",
        n_repeats="count",
    ).reset_index()

    # Also aggregate steps_per_s if present.
    if "steps_per_s" in valid.columns:
        sps = valid.groupby(["solver", "backend", "resolution"])["steps_per_s"].agg(
            sps_mean="mean",
            sps_std="std",
        ).reset_index()
        agg = agg.merge(sps, on=["solver", "backend", "resolution"], how="left")

    # Fill std=0 for single-repeat runs (NaN std from a single sample).
    agg["walltime_std"] = agg["walltime_std"].fillna(0.0)
    if "sps_std" in agg.columns:
        agg["sps_std"] = agg["sps_std"].fillna(0.0)

    return agg


def label_for(row: pd.Series) -> str:
    return f"{row['solver']}/{row['backend']} N={row['resolution']}"


def plot_speed_table(agg: pd.DataFrame, figures_dir: str) -> str:
    """Bar chart of mean walltime with std error bars. Returns saved path."""
    fig, ax = plt.subplots(figsize=(max(6, len(agg) * 1.2), 5))

    labels = [label_for(r) for _, r in agg.iterrows()]
    x = np.arange(len(labels))
    means = agg["walltime_mean"].values
    stds = agg["walltime_std"].values
    n_repeats = agg["n_repeats"].values

    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#4878cf", alpha=0.82, error_kw={"elinewidth": 1.5})

    # Annotate single-repeat bars.
    for i, (n, std) in enumerate(zip(n_repeats, stds)):
        if n == 1:
            ax.annotate("(1 repeat, no std)", xy=(x[i], means[i]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7, color="#555555")
        else:
            ax.annotate(f"n={n}", xy=(x[i], means[i] + stds[i]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Solver speed (mean walltime, error bars = std across repeats)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(figures_dir, "speed_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_steps_per_s(agg: pd.DataFrame, figures_dir: str) -> str | None:
    """Bar chart of steps/s. Returns saved path or None if no data."""
    if "sps_mean" not in agg.columns:
        return None

    valid = agg.dropna(subset=["sps_mean"])
    if valid.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(valid) * 1.2), 5))
    labels = [label_for(r) for _, r in valid.iterrows()]
    x = np.arange(len(labels))
    means = valid["sps_mean"].values
    stds = valid["sps_std"].values if "sps_std" in valid.columns else np.zeros_like(means)
    n_repeats = valid["n_repeats"].values

    ax.bar(x, means, yerr=stds, capsize=5, color="#6acc65", alpha=0.82, error_kw={"elinewidth": 1.5})

    for i, n in enumerate(n_repeats):
        label = f"n={n}" if n > 1 else "(1 repeat)"
        ax.annotate(label, xy=(x[i], means[i] + stds[i]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Steps per second")
    ax.set_title("Solver throughput (steps/s, mean across repeats)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(figures_dir, "steps_per_s.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pareto(agg: pd.DataFrame, acc_df: pd.DataFrame | None, figures_dir: str) -> str:
    """Accuracy-vs-walltime scatter. If no accuracy data, plots a placeholder."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if acc_df is not None and not acc_df.empty:
        merged = agg.merge(acc_df, on=["solver", "backend", "resolution"], how="inner")
        if merged.empty:
            print("WARNING: accuracy CSV provided but no rows matched timing data", file=sys.stderr)
        else:
            for _, row in merged.iterrows():
                lbl = label_for(row)
                ax.errorbar(
                    row["walltime_mean"],
                    row["error"],
                    xerr=row.get("walltime_std", 0.0),
                    fmt="o",
                    label=lbl,
                    capsize=4,
                )
            ax.set_xlabel("Wall-clock time (s)")
            ax.set_ylabel("Error (lower is better)")
            ax.set_title("Accuracy vs speed (Pareto front)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    else:
        # Placeholder: scatter walltime distribution, accuracy axis blank.
        for _, row in agg.iterrows():
            lbl = label_for(row)
            ax.errorbar(
                row["walltime_mean"],
                0.5,
                xerr=row.get("walltime_std", 0.0),
                fmt="o",
                label=lbl,
                capsize=4,
            )
        ax.set_xlabel("Wall-clock time (s)")
        ax.set_ylabel("Accuracy (not yet available)")
        ax.set_yticks([])
        ax.set_title(
            "Accuracy vs speed (Pareto)\n"
            "[accuracy axis: placeholder, to be filled in during final gamut]"
        )
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.text(
            0.5, 0.05,
            "Accuracy data not yet supplied. Re-run with --accuracy-csv once\n"
            "per-run error metrics are available from the final gamut.",
            transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8, color="#888888",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8),
        )

    fig.tight_layout()
    path = os.path.join(figures_dir, "pareto.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def print_summary_table(agg: pd.DataFrame) -> None:
    print("\n=== Speed summary ===")
    print(f"{'Solver/Backend/N':<30} {'mean_s':>8} {'std_s':>8} {'n':>4} {'steps/s':>10}")
    print("-" * 66)
    for _, row in agg.iterrows():
        lbl = label_for(row)
        std_s = row.get("walltime_std", 0.0)
        sps = row.get("sps_mean", float("nan"))
        print(f"{lbl:<30} {row['walltime_mean']:>8.3f} {std_s:>8.3f} {int(row['n_repeats']):>4} {sps:>10.1f}")
    print()


def main() -> None:
    args = parse_args()

    csv_paths = list(args.csvs)

    # Resolve relative paths against cwd, then also try benchmarks/results/.
    resolved = []
    for p in csv_paths:
        if os.path.exists(p):
            resolved.append(os.path.abspath(p))
        else:
            alt = os.path.join(args.results_dir, p)
            if os.path.exists(alt):
                resolved.append(os.path.abspath(alt))
            else:
                print(f"WARNING: {p} not found (tried {p} and {alt})", file=sys.stderr)

    if not resolved:
        # Auto-discover.
        resolved = discover_csvs(args.results_dir)
        if resolved:
            print(f"Auto-discovered {len(resolved)} CSV(s) in {args.results_dir}")
        else:
            print("ERROR: no timing CSVs found. Pass CSV paths explicitly or run bench_runner.py first.", file=sys.stderr)
            sys.exit(1)

    df = load_timing_csvs(resolved)
    if df.empty:
        print("ERROR: no usable timing data loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} repeat rows from {len(resolved)} CSV(s)")

    agg = aggregate(df)
    if agg.empty:
        print("ERROR: no valid (non-NaN) walltime rows after aggregation.", file=sys.stderr)
        sys.exit(1)

    print_summary_table(agg)

    acc_df = None
    if args.accuracy_csv and os.path.exists(args.accuracy_csv):
        acc_df = pd.read_csv(args.accuracy_csv)
        print(f"Loaded accuracy data: {len(acc_df)} rows from {args.accuracy_csv}")

    figures_dir = args.figures_dir
    os.makedirs(figures_dir, exist_ok=True)

    p1 = plot_speed_table(agg, figures_dir)
    print(f"Saved: {p1}")

    p2 = plot_steps_per_s(agg, figures_dir)
    if p2:
        print(f"Saved: {p2}")

    p3 = plot_pareto(agg, acc_df, figures_dir)
    print(f"Saved: {p3}")


if __name__ == "__main__":
    main()
