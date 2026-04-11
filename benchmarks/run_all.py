#!/usr/bin/env python3
"""Cross-solver benchmark orchestrator.

Usage:
    python run_all.py build     # Build/install all reference solvers
    python run_all.py run       # Run canonical test case on all solvers
    python run_all.py compare   # Compare results numerically
    python run_all.py report    # Generate figures and markdown report
    python run_all.py all       # All of the above
"""
import os
import sys
import subprocess
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REF_DIR = os.path.join(SCRIPT_DIR, "reference_data")

SOLVERS = {
    "vonkarman": {
        "build": None,
        "run": os.path.join(SCRIPT_DIR, "vonkarman", "run_tg.sh"),
        "parse": os.path.join(SCRIPT_DIR, "vonkarman", "parse_output.py"),
        "csv": "vonkarman_cpu.csv",
    },
    "hit3d": {
        "build": os.path.join(SCRIPT_DIR, "hit3d", "build.sh"),
        "run": os.path.join(SCRIPT_DIR, "hit3d", "run.sh"),
        "parse": os.path.join(SCRIPT_DIR, "hit3d", "parse_output.py"),
        "csv": "hit3d.csv",
    },
    "spectraldns": {
        "build": os.path.join(SCRIPT_DIR, "spectraldns", "setup.sh"),
        "run": os.path.join(SCRIPT_DIR, "spectraldns", "run_tg.py"),
        "parse": os.path.join(SCRIPT_DIR, "spectraldns", "parse_output.py"),
        "csv": "spectraldns.csv",
    },
    "dedalus": {
        "build": os.path.join(SCRIPT_DIR, "dedalus", "setup.sh"),
        "run": os.path.join(SCRIPT_DIR, "dedalus", "run_tg.py"),
        "parse": os.path.join(SCRIPT_DIR, "dedalus", "parse_output.py"),
        "csv": "dedalus.csv",
    },
}


def run_cmd(cmd, label, cwd=None):
    """Run a command, print output on failure."""
    print(f"  [{label}] {cmd}")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s):")
        if result.stdout:
            print(result.stdout[-500:])
        if result.stderr:
            print(result.stderr[-500:])
        return False
    print(f"  OK ({elapsed:.1f}s)")
    return True


def cmd_build():
    """Build all reference solvers."""
    print("=== Building reference solvers ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for name, cfg in SOLVERS.items():
        if cfg["build"]:
            ok = run_cmd(f"bash {cfg['build']}", f"build:{name}")
            if not ok:
                print(f"  WARNING: {name} build failed, skipping")


def cmd_run():
    """Run all solvers."""
    print("=== Running solvers ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for name, cfg in SOLVERS.items():
        run_script = cfg["run"]
        if not run_script or not os.path.exists(run_script):
            print(f"  [{name}] no run script, skipping")
            continue

        if run_script.endswith(".py"):
            cmd = f"python3 {run_script}"
        else:
            cmd = f"bash {run_script}"

        ok = run_cmd(cmd, f"run:{name}")
        if not ok:
            continue

        # Parse output
        parse_script = cfg["parse"]
        if parse_script and os.path.exists(parse_script):
            run_cmd(f"python3 {parse_script}", f"parse:{name}")


def cmd_compare():
    """Compare results numerically."""
    print("=== Comparing results ===")
    try:
        import pandas as pd
    except ImportError:
        print("  pandas not installed. pip install pandas", file=sys.stderr)
        return

    # Load all available CSVs
    data = {}
    for name, cfg in SOLVERS.items():
        csv_path = os.path.join(RESULTS_DIR, cfg["csv"])
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            data[name] = df
            print(
                f"  Loaded {name}: {len(df)} rows, "
                f"t=[{df['t'].min():.2f}, {df['t'].max():.2f}]"
            )
        else:
            print(f"  {name}: no results (run first)")

    if len(data) < 2:
        print("  Need at least 2 solver results to compare")
        return

    # Compare enstrophy peaks
    print("\n  Enstrophy peaks:")
    for name, df in data.items():
        if "Omega" in df.columns and df["Omega"].max() > 0:
            idx = df["Omega"].idxmax()
            print(
                f"    {name:>15}: peak Omega = {df['Omega'].iloc[idx]:.4e} "
                f"at t = {df['t'].iloc[idx]:.3f}"
            )

    # Compare energy at t=10
    print("\n  Energy at t=10:")
    for name, df in data.items():
        if "E" in df.columns:
            t10 = df.loc[(df["t"] - 10.0).abs().idxmin()]
            print(f"    {name:>15}: E(10) = {t10['E']:.8e}")

    # Pairwise L2 comparison against vonkarman
    if "vonkarman" in data:
        ref = data["vonkarman"]
        print("\n  L2 deviation from vonkarman:")
        for name, df in data.items():
            if name == "vonkarman" or "E" not in df.columns:
                continue
            t_max = min(ref["t"].max(), df["t"].max())
            t_common = np.linspace(0, t_max, 200)
            E_ref = np.interp(t_common, ref["t"], ref["E"])
            E_other = np.interp(t_common, df["t"], df["E"])
            l2 = np.sqrt(np.mean((E_ref - E_other) ** 2)) / np.mean(np.abs(E_ref))
            print(f"    {name:>15}: L2 = {l2:.6e}")


def cmd_report():
    """Generate markdown report and figures."""
    print("=== Generating report ===")
    try:
        import pandas as pd
    except ImportError:
        print("  pandas not installed", file=sys.stderr)
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figures")
        plt = None

    fig_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    data = {}
    for name, cfg in SOLVERS.items():
        csv_path = os.path.join(RESULTS_DIR, cfg["csv"])
        if os.path.exists(csv_path):
            data[name] = pd.read_csv(csv_path)

    if plt and data:
        # Energy decay
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, df in data.items():
            if "E" in df.columns:
                ax.plot(df["t"], df["E"], label=name)
        ax.set_xlabel("t")
        ax.set_ylabel("E(t)")
        ax.set_title("Energy Decay: Taylor-Green Re=1600 N=128")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(
            os.path.join(fig_dir, "energy_decay.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"  Saved: {fig_dir}/energy_decay.png")

        # Enstrophy
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, df in data.items():
            if "Omega" in df.columns and df["Omega"].max() > 0:
                ax.plot(df["t"], df["Omega"], label=name)
        brachet_path = os.path.join(REF_DIR, "brachet1983.csv")
        if os.path.exists(brachet_path):
            ref = pd.read_csv(brachet_path)
            ax.plot(
                ref["t"], ref["enstrophy"], "ko--", label="Brachet 1983", markersize=4
            )
        ax.set_xlabel("t")
        ax.set_ylabel("Enstrophy")
        ax.set_title("Enstrophy: Taylor-Green Re=1600 N=128")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(
            os.path.join(fig_dir, "enstrophy.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"  Saved: {fig_dir}/enstrophy.png")

        # Dissipation rate
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, df in data.items():
            if "epsilon" in df.columns and df["epsilon"].max() > 0:
                ax.plot(df["t"], df["epsilon"], label=name)
        vanrees_path = os.path.join(REF_DIR, "vanrees2011.csv")
        if os.path.exists(vanrees_path):
            ref = pd.read_csv(vanrees_path)
            ax.plot(
                ref["t"],
                ref["epsilon"],
                "ko--",
                label="van Rees 2011",
                markersize=4,
            )
        ax.set_xlabel("t")
        ax.set_ylabel("epsilon(t)")
        ax.set_title("Dissipation Rate: Taylor-Green Re=1600 N=128")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(
            os.path.join(fig_dir, "dissipation.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"  Saved: {fig_dir}/dissipation.png")

    # Markdown report
    report_path = os.path.join(RESULTS_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write("# Cross-Solver Benchmark Report\n\n")
        f.write("Taylor-Green vortex, Re=1600, N=128, t in [0, 10]\n\n")
        f.write("## Solvers\n\n")
        f.write("| Solver | Results |\n|--------|--------|\n")
        for name in SOLVERS:
            status = "available" if name in data else "missing"
            f.write(f"| {name} | {status} |\n")

        if data:
            f.write("\n## Enstrophy Peaks\n\n")
            f.write(
                "| Solver | Peak Enstrophy | Peak Time |\n"
                "|--------|---------------|----------|\n"
            )
            for name, df in data.items():
                if "Omega" in df.columns and df["Omega"].max() > 0:
                    idx = df["Omega"].idxmax()
                    f.write(
                        f"| {name} | {df['Omega'].iloc[idx]:.4e} "
                        f"| {df['t'].iloc[idx]:.3f} |\n"
                    )

        f.write("\n## Figures\n\n")
        f.write("![Energy](figures/energy_decay.png)\n\n")
        f.write("![Enstrophy](figures/enstrophy.png)\n\n")
        f.write("![Dissipation](figures/dissipation.png)\n\n")

    print(f"  Report: {report_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    commands = {
        "build": cmd_build,
        "run": cmd_run,
        "compare": cmd_compare,
        "report": cmd_report,
        "all": lambda: (cmd_build(), cmd_run(), cmd_compare(), cmd_report()),
    }

    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(commands)}")
        sys.exit(1)

    commands[cmd]()


if __name__ == "__main__":
    main()
