#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

module load mpi/openmpi-x86_64 2>/dev/null || true
export PATH="/usr/lib64/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}"

if [ ! -f hit3d ]; then
    echo "hit3d binary not found. Run build.sh first."
    exit 1
fi

echo "Running hit3d Taylor-Green Re=1600 N=128..."
mpirun --oversubscribe -np 1 ./hit3d 2>&1 | tee run.log

echo "Done. Output in $(pwd)/"
