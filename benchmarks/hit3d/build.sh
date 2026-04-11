#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load MPI
module load mpi/openmpi-x86_64 2>/dev/null || true
export PATH="/usr/lib64/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}"

# Clone if needed
if [ ! -d "src" ]; then
    echo "Cloning hit3d..."
    git clone https://github.com/cpraveen/hit3d.git src
fi

cd src

# Build with gfortran 15 flags
echo "Building hit3d from source..."
SRC_FILES=$(find . -name '*.f90' -not -path './tests/*' | sort | tr '\n' ' ')

mpif90 -O3 -fdefault-real-8 -fdefault-integer-8 -fallow-argument-mismatch \
    -I/usr/include -o ../hit3d $SRC_FILES -lfftw3 2>&1 || {
    echo "Direct build failed, trying file-by-file compilation..."
    # Some versions need compilation in dependency order
    for f in $SRC_FILES; do
        mpif90 -O3 -fdefault-real-8 -fdefault-integer-8 -fallow-argument-mismatch \
            -I/usr/include -c "$f" 2>/dev/null || echo "  skip: $f"
    done
    mpif90 -O3 -o ../hit3d *.o -lfftw3 2>&1 || echo "Linking failed. Check source compatibility."
}

cd ..
if [ -f hit3d ]; then
    echo "hit3d built successfully: $(pwd)/hit3d"
    ls -la hit3d
else
    echo "WARNING: hit3d build failed. May need manual Makefile adjustment."
    echo "Check src/ for the actual build system."
fi
