#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$WORKSPACE"
RUSTFLAGS="-C target-cpu=native" cargo build --release -p vonkarman-bin 2>&1 | tail -1

cd "$SCRIPT_DIR"
echo "Running vonkarman TG Re=1600 N=128 (cuFFT GPU)..."
"$WORKSPACE/target/release/vonkarman" run --config tg_re1600.toml --backend cufft
echo "Done."
