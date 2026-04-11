#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "TurboGenPY" ]; then
    git clone https://github.com/saadgroup/TurboGenPY.git
fi

echo "TurboGenPY cloned to $(pwd)/TurboGenPY"
