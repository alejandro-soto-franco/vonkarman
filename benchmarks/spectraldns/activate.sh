#!/bin/bash
# Activate spectralDNS venv with MPI paths
source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
export PATH="/usr/lib64/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}"
