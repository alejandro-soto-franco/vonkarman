#!/usr/bin/env python3
"""Parse hit3d diagnostic output into normalised CSV.

hit3d writes diagnostic data to ASCII files. The exact format depends
on the version. This script will need adaptation after inspecting actual output.

Output: ../results/hit3d.csv with columns: t,E,Omega,epsilon,max_omega
"""
import sys
import os
import glob
import numpy as np


def parse_hit3d_output(run_dir: str) -> np.ndarray:
    """Parse hit3d output files and return (t, E, Omega, epsilon, max_omega) array."""
    # Look for diagnostic output files (common names across hit3d versions)
    candidates = glob.glob(os.path.join(run_dir, "*.txt"))
    candidates += glob.glob(os.path.join(run_dir, "*.dat"))

    if not candidates:
        print(f"No output files found in {run_dir}", file=sys.stderr)
        print(f"Available files: {os.listdir(run_dir)}", file=sys.stderr)
        sys.exit(1)

    print(f"Found output files: {[os.path.basename(f) for f in candidates]}")

    # Try to find the main diagnostics file
    for path in candidates:
        try:
            data = np.loadtxt(path, comments="#")
            if data.ndim == 2 and data.shape[1] >= 2:
                print(f"Parsed: {path} ({data.shape[0]} rows, {data.shape[1]} columns)")
                # Pad to 5 columns if needed
                if data.shape[1] < 5:
                    padded = np.zeros((data.shape[0], 5))
                    padded[:, : data.shape[1]] = data
                    data = padded
                return data[:, :5]
        except (ValueError, OSError):
            continue

    print("Could not parse any output file", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    run_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(run_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    data = parse_hit3d_output(run_dir)

    out_path = os.path.join(results_dir, "hit3d.csv")
    np.savetxt(
        out_path,
        data,
        delimiter=",",
        header="t,E,Omega,epsilon,max_omega",
        comments="",
    )
    print(f"Written: {out_path}")
