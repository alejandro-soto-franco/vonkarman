#!/usr/bin/env python3
"""vonkarman timing harness: wall-clock capture per repeat, one CSV row per run.

Usage:
    uv run python benchmarks/vonkarman/bench_runner.py [OPTIONS]

Options:
    --config PATH       TOML config file for vonkarman (relative to this script)
    --backend STR       Backend override: cpu, cufft, auto  (default: cpu)
    --resolution N      Grid resolution N (overrides config via generated TOML)
    --end-time T        Physical end-time (overrides config)
    --max-steps K       Max steps (overrides config; takes precedence over end-time)
    --repeats N         Number of timed repeats after one warm-up (default: 3)
    --out PATH          Output CSV path (default: benchmarks/results/timing_<solver>.csv)
    --append            Append to existing CSV instead of overwriting
    --solver NAME       Solver label written to CSV (default: vonkarman)
    --notes STR         Free-form notes column value

Wall-clock is measured with time.perf_counter() around the subprocess call.
One warm-up run is performed and discarded (GPU initialisation, OS caching).
Every repeat is written as its own CSV row so variance is visible at plot time.
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BINARY = os.path.join(WORKSPACE, "target", "release", "vonkarman")
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results"))

CSV_COLUMNS = [
    "solver",
    "backend",
    "resolution",
    "repeat_index",
    "walltime_s",
    "steps",
    "steps_per_s",
    "end_time",
    "notes",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(SCRIPT_DIR, "tg_re1600.toml"))
    p.add_argument("--backend", default="cpu", choices=["cpu", "cufft", "auto"])
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--end-time", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default=None)
    p.add_argument("--append", action="store_true")
    p.add_argument("--solver", default="vonkarman")
    p.add_argument("--notes", default="")
    return p.parse_args()


def load_toml_text(path: str) -> str:
    with open(path) as f:
        return f.read()


def patch_toml(toml_text: str, backend: str, resolution: int | None, end_time: float | None, max_steps: int | None, output_dir: str) -> str:
    """Produce a patched TOML string with overrides applied.

    Rather than pulling in a full TOML library, we do line-by-line substitution
    on the relevant scalar fields. The TOML files here are simple enough that
    this is robust.
    """
    import re

    lines = toml_text.splitlines()
    patched = []
    section = ""
    for line in lines:
        m = re.match(r"^\[(\w+)\]", line.strip())
        if m:
            section = m.group(1)

        if section == "domain" and re.match(r"\s*backend\s*=", line):
            line = f'backend = "{backend}"'
        elif section == "domain" and resolution is not None and re.match(r"\s*n\s*=", line):
            line = f"n = {resolution}"
        elif section == "termination" and end_time is not None and re.match(r"\s*max_time\s*=", line):
            line = f"max_time = {end_time}"
        elif section == "termination" and max_steps is not None and re.match(r"\s*max_steps\s*=", line):
            line = f"max_steps = {max_steps}"
        elif section == "run" and re.match(r"\s*output_dir\s*=", line):
            line = f'output_dir = "{output_dir}"'

        patched.append(line)

    result = "\n".join(patched)

    # Add missing fields that were not present in the original.
    if section != "termination":
        # No termination section found; cannot add. The original TOML should have it.
        pass
    if max_steps is not None and "max_steps" not in result:
        result += f"\nmax_steps = {max_steps}\n"
    if end_time is not None and "max_time" not in result:
        result += f"\nmax_time = {end_time}\n"

    return result


def run_once(toml_path: str, backend: str, output_dir: str) -> tuple[float, int, float]:
    """Run vonkarman once, return (walltime_s, steps, sim_end_time).

    Wall-clock is from perf_counter around the subprocess. Steps and sim_end_time
    are read from the diagnostics parquet written by the binary.
    """
    t0 = time.perf_counter()
    result = subprocess.run(
        [BINARY, "run", "--config", toml_path],
        capture_output=True,
        text=True,
    )
    walltime = time.perf_counter() - t0

    if result.returncode != 0:
        msg = result.stderr[-1000:] if result.stderr else "(no stderr)"
        raise RuntimeError(f"vonkarman exited {result.returncode}:\n{msg}")

    # Read diagnostics parquet to get step count and final sim time.
    parquet_path = os.path.join(output_dir, "diagnostics.parquet")
    steps = 0
    sim_end_time = 0.0
    if os.path.exists(parquet_path):
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_path)
            n_rows = table.num_rows
            # Rows include initial state (step 0), so steps = n_rows - 1 if > 0.
            steps = max(0, n_rows - 1)
            if "time" in table.schema.names and n_rows > 0:
                col = table.column("time")
                sim_end_time = float(col[-1].as_py())
        except Exception as e:
            print(f"  WARNING: could not read parquet: {e}", file=sys.stderr)

    return walltime, steps, sim_end_time


def check_binary() -> bool:
    if not os.path.isfile(BINARY):
        print(f"ERROR: vonkarman binary not found at {BINARY}", file=sys.stderr)
        print("Build with: RUSTFLAGS='-C target-cpu=native' cargo build --release -p vonkarman-bin", file=sys.stderr)
        return False
    return True


def main() -> None:
    args = parse_args()

    if not check_binary():
        sys.exit(1)

    # Resolve config path.
    config_path = args.config if os.path.isabs(args.config) else os.path.join(SCRIPT_DIR, args.config)
    if not os.path.exists(config_path):
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    out_csv = args.out
    if out_csv is None:
        label = f"{args.solver}_{args.backend}"
        if args.resolution:
            label += f"_n{args.resolution}"
        out_csv = os.path.join(RESULTS_DIR, f"timing_{label}.csv")

    resolution = args.resolution
    backend = args.backend
    end_time = args.end_time
    max_steps = args.max_steps

    # Determine resolution from config if not overridden (for CSV metadata).
    if resolution is None:
        import re
        with open(config_path) as f:
            txt = f.read()
        m = re.search(r"^\s*n\s*=\s*(\d+)", txt, re.MULTILINE)
        resolution = int(m.group(1)) if m else 0

    print(f"vonkarman bench_runner")
    print(f"  binary   : {BINARY}")
    print(f"  config   : {config_path}")
    print(f"  backend  : {backend}")
    print(f"  N        : {resolution}")
    print(f"  repeats  : {args.repeats} (+ 1 warm-up discarded)")
    print(f"  output   : {out_csv}")
    print()

    # Build patched TOML once (shared across warm-up and all repeats).
    toml_text = load_toml_text(config_path)
    run_output_dir = os.path.join(RESULTS_DIR, f"vonkarman_bench_{backend}_n{resolution}")

    patched = patch_toml(toml_text, backend=backend, resolution=resolution if args.resolution else None,
                         end_time=end_time, max_steps=max_steps, output_dir=run_output_dir)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False, dir=RESULTS_DIR) as tf:
        tf.write(patched)
        tmp_toml = tf.name

    try:
        # Warm-up run: initialises CUDA context, warms OS page cache.
        print("  [warm-up] running (discarded)...")
        try:
            wt, _, _ = run_once(tmp_toml, backend, run_output_dir)
            print(f"  [warm-up] done in {wt:.2f}s")
        except RuntimeError as e:
            # If warm-up fails (e.g. cufft unavailable), record failure and exit.
            note = f"backend_init_failed: {str(e)[:200]}"
            print(f"  [warm-up] FAILED: {note}", file=sys.stderr)
            _write_failure_row(out_csv, args, backend, resolution, note, args.append)
            return

        # Timed repeats.
        rows = []
        for i in range(args.repeats):
            print(f"  [repeat {i+1}/{args.repeats}] running...", end=" ", flush=True)
            try:
                wt, steps, sim_end = run_once(tmp_toml, backend, run_output_dir)
                sps = steps / wt if wt > 0 else 0.0
                print(f"{wt:.3f}s  ({steps} steps, {sps:.1f} steps/s, sim_t={sim_end:.4f})")
                rows.append({
                    "solver": args.solver,
                    "backend": backend,
                    "resolution": resolution,
                    "repeat_index": i,
                    "walltime_s": round(wt, 6),
                    "steps": steps,
                    "steps_per_s": round(sps, 3),
                    "end_time": round(sim_end, 6),
                    "notes": args.notes,
                })
            except RuntimeError as e:
                note = f"run_failed: {str(e)[:200]}"
                print(f"FAILED: {note}", file=sys.stderr)
                rows.append({
                    "solver": args.solver,
                    "backend": backend,
                    "resolution": resolution,
                    "repeat_index": i,
                    "walltime_s": float("nan"),
                    "steps": 0,
                    "steps_per_s": float("nan"),
                    "end_time": float("nan"),
                    "notes": note,
                })

        # Write CSV (one row per repeat).
        mode = "a" if (args.append and os.path.exists(out_csv)) else "w"
        write_header = mode == "w" or not os.path.exists(out_csv)
        with open(out_csv, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        print()
        print(f"  Written {len(rows)} rows to {out_csv}")

    finally:
        os.unlink(tmp_toml)


def _write_failure_row(out_csv: str, args: argparse.Namespace, backend: str, resolution: int, note: str, append: bool) -> None:
    mode = "a" if (append and os.path.exists(out_csv)) else "w"
    write_header = mode == "w" or not os.path.exists(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "solver": args.solver,
            "backend": backend,
            "resolution": resolution,
            "repeat_index": -1,
            "walltime_s": float("nan"),
            "steps": 0,
            "steps_per_s": float("nan"),
            "end_time": float("nan"),
            "notes": note,
        })
    print(f"  Failure row written to {out_csv}")


if __name__ == "__main__":
    main()
