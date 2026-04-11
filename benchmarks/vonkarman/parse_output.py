#!/usr/bin/env python3
"""Parse vonkarman Parquet diagnostics into normalised CSV."""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")


def main():
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed. pip install pandas", file=sys.stderr)
        sys.exit(1)

    parquet_dir = os.path.join(RESULTS_DIR, "vonkarman_output")
    parquet_path = os.path.join(parquet_dir, "diagnostics.parquet")

    if not os.path.exists(parquet_path):
        print(f"Parquet not found: {parquet_path}", file=sys.stderr)
        print("Run run_tg.sh first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    nu = 6.25e-4

    out = pd.DataFrame(
        {
            "t": df["time"],
            "E": df["energy"],
            "Omega": df["enstrophy"],
            "epsilon": 2 * nu * df["enstrophy"],
            "max_omega": df["max_vorticity"],
        }
    )

    out_path = os.path.join(RESULTS_DIR, "vonkarman_cpu.csv")
    out.to_csv(out_path, index=False)
    print(f"Written: {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
