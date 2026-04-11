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
pip install --quiet numpy mpi4py pyfftw h5py
pip install --quiet shenfun mpi4py-fft

# Clone spectralDNS
if [ ! -d "spectralDNS" ]; then
    git clone https://github.com/spectralDNS/spectralDNS.git
fi

cd spectralDNS
pip install --quiet -e .
cd ..

echo "spectralDNS installed in $(pwd)/.venv"
