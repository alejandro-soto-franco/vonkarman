#!/usr/bin/env python3
"""Run Taylor-Green Re=1600 using spectralDNS and export diagnostics.

spectralDNS API varies by version. This script attempts the common patterns.
May need adjustment after installation.
"""
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(SCRIPT_DIR, ".venv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(VENV):
        print("Virtual environment not found. Run setup.sh first.", file=sys.stderr)
        sys.exit(1)

    python = os.path.join(VENV, "bin", "python3")

    # Try running the built-in TG demo
    tg_script = os.path.join(SCRIPT_DIR, "spectralDNS", "demo", "TG.py")
    if not os.path.exists(tg_script):
        print(f"TG.py not found at {tg_script}", file=sys.stderr)
        print("spectralDNS may not be cloned. Run setup.sh first.", file=sys.stderr)
        sys.exit(1)

    print("Running spectralDNS Taylor-Green Re=1600 N=128...")
    result = subprocess.run(
        [
            python,
            tg_script,
            "--M", "7", "7", "7",  # N = 2^7 = 128
            "--nu", "6.25e-4",
            "--T", "10.0",
            "--dealias", "3/2-rule",
            "NS",
        ],
        cwd=os.path.join(SCRIPT_DIR, "spectralDNS", "demo"),
        capture_output=True,
        text=True,
    )

    print(result.stdout[-1000:] if result.stdout else "(no stdout)")
    if result.returncode != 0:
        print(f"Failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr[-500:] if result.stderr else "(no stderr)", file=sys.stderr)
        sys.exit(1)

    print("Done. Run parse_output.py to extract CSV.")


if __name__ == "__main__":
    main()
