#!/usr/bin/env python3
"""Generate von Karman spectrum IC using TurboGenPY and export E(k)
for comparison with vonkarman's random_isotropic IC.

Falls back to analytical von Karman spectrum if TurboGenPY import fails.
"""
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")


def analytical_von_karman(N, k_peak=4.0):
    """Generate analytical von Karman spectrum E(k) ~ k^4 / (1 + (k/kp)^2)^(17/6)."""
    k = np.arange(1, N // 2 + 1, dtype=float)
    Ek = k**4 / (1 + (k / k_peak) ** 2) ** (17.0 / 6.0)
    # Normalise to total energy = 0.5
    Ek *= 0.5 / np.trapezoid(Ek, k)
    return k, Ek


def main():
    N = 128
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sys.path.insert(0, os.path.join(SCRIPT_DIR, "TurboGenPY"))

    try:
        from tgpy import TurboGen

        tg = TurboGen(N=N, L=2 * np.pi, spectrum="vkp", k_peak=4.0)
        tg.generate()
        k, Ek = tg.compute_spectrum()
        source = "TurboGenPY"
    except (ImportError, Exception) as e:
        print(f"TurboGenPY import failed ({e}), using analytical spectrum", file=sys.stderr)
        k, Ek = analytical_von_karman(N)
        source = "analytical von Karman"

    out_path = os.path.join(RESULTS_DIR, "turbogenpy_spectrum.csv")
    np.savetxt(
        out_path,
        np.column_stack([k, Ek]),
        delimiter=",",
        header="k,Ek",
        comments="",
    )
    print(f"Written: {out_path} (source: {source})")


if __name__ == "__main__":
    main()
