#!/usr/bin/env python3
"""Parse Dedalus output (CSV already written by run_tg.py)."""
import os
import sys

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
csv_path = os.path.join(results_dir, "dedalus.csv")

if os.path.exists(csv_path):
    print(f"Dedalus results at: {csv_path}")
else:
    print("No results found. Run run_tg.py first.", file=sys.stderr)
    sys.exit(1)
