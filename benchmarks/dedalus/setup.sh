#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

module load mpi/openmpi-x86_64 2>/dev/null || true
export PATH="/usr/lib64/openmpi/bin:$PATH"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --quiet numpy mpi4py cython

# Install Dedalus with system FFTW
FFTW_PATH=/usr pip install --quiet dedalus 2>&1 || {
    echo "Dedalus pip install failed. Trying with explicit FFTW flags..."
    FFTW_PATH=/usr FFTW_STATIC=0 pip install --quiet dedalus 2>&1 || {
        echo "WARNING: Dedalus installation failed. May need manual setup."
        exit 1
    }
}

echo "Dedalus installed in $(pwd)/.venv"
python3 -c "import dedalus; print(f'Dedalus version: {dedalus.__version__}')" 2>/dev/null || echo "Dedalus import check failed"
