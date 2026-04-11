#!/usr/bin/env python3
"""Parse hit3d stat1.gp diagnostic output into normalised CSV.

stat1.gp columns: itime, time, energy, dissipation(eps_v), eta, enstrophy, Re_lambda
Format: i8, 20e15.6

Output: ../results/hit3d.csv with columns: t,E,Omega,epsilon,max_omega
"""
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")


def main():
    stat_path = os.path.join(SCRIPT_DIR, "stat1.gp")
    if not os.path.exists(stat_path):
        print(f"stat1.gp not found in {SCRIPT_DIR}", file=sys.stderr)
        print("Run run.sh first.", file=sys.stderr)
        sys.exit(1)

    # Parse stat1.gp: skip comment lines, read fixed-width columns
    rows = []
    with open(stat_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    itime = int(parts[0])
                    t = float(parts[1])
                    E = float(parts[2])
                    eps = float(parts[3])
                    eta = float(parts[4])
                    Omega = float(parts[5])
                    rows.append((t, E, Omega, eps, 0.0))
                except (ValueError, IndexError):
                    continue

    if not rows:
        print("No valid data rows found in stat1.gp", file=sys.stderr)
        sys.exit(1)

    data = np.array(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "hit3d.csv")
    np.savetxt(
        out_path,
        data,
        delimiter=",",
        header="t,E,Omega,epsilon,max_omega",
        comments="",
    )
    print(f"Written: {out_path} ({len(data)} rows)")
    print(f"  t = [{data[0,0]:.3f}, {data[-1,0]:.3f}]")
    print(f"  E = [{data[:,1].min():.6e}, {data[:,1].max():.6e}]")
    print(f"  peak Omega = {data[:,2].max():.6e} at t = {data[data[:,2].argmax(), 0]:.3f}")


if __name__ == "__main__":
    main()
