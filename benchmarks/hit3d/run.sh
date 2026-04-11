#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="/usr/lib64/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}"

BINARY="$SCRIPT_DIR/src/hit3d.x"
if [ ! -f "$BINARY" ]; then
    echo "hit3d binary not found at $BINARY. Run build.sh first."
    exit 1
fi

# Run name must be exactly 10 characters; input file is <run_name>.in
RUN_NAME="tg_re01600"

echo "Running hit3d Taylor-Green Re=1600 N=128..."
echo "  binary: $BINARY"
echo "  input:  ${RUN_NAME}.in"

mpirun --oversubscribe -np 1 "$BINARY" "$RUN_NAME" 2>&1 | tee run.log

echo "Done. Diagnostics in stat1.gp, stat2.gp, es.gp"
