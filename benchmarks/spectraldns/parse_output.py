#!/usr/bin/env python3
"""Parse spectralDNS output into normalised CSV."""
import os
import sys
import glob
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "spectraldns.csv")

    # Look for HDF5 output
    h5_files = glob.glob(os.path.join(SCRIPT_DIR, "spectralDNS", "demo", "*.h5"))
    h5_files += glob.glob(os.path.join(SCRIPT_DIR, "*.h5"))

    if h5_files:
        try:
            import h5py

            with h5py.File(h5_files[0], "r") as f:
                print(f"HDF5 keys: {list(f.keys())}")
                # Attempt common spectralDNS output layouts
                if "Energy" in f:
                    t = np.arange(len(f["Energy"])) * 0.001  # approximate
                    E = f["Energy"][:]
                    data = np.column_stack([t, E, np.zeros_like(E), np.zeros_like(E), np.zeros_like(E)])
                    np.savetxt(out_path, data, delimiter=",",
                               header="t,E,Omega,epsilon,max_omega", comments="")
                    print(f"Written: {out_path}")
                    return
        except ImportError:
            print("h5py not available", file=sys.stderr)

    # Look for text output
    txt_files = glob.glob(os.path.join(SCRIPT_DIR, "spectralDNS", "demo", "*.txt"))
    for path in txt_files:
        try:
            data = np.loadtxt(path)
            if data.ndim == 2 and data.shape[1] >= 2:
                padded = np.zeros((data.shape[0], 5))
                padded[:, :data.shape[1]] = data
                np.savetxt(out_path, padded, delimiter=",",
                           header="t,E,Omega,epsilon,max_omega", comments="")
                print(f"Written: {out_path} from {path}")
                return
        except (ValueError, OSError):
            continue

    print("No parseable output found. Run run_tg.py first.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
