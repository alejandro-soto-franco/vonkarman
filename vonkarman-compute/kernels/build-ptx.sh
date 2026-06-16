#!/usr/bin/env bash
# Compile the vonkarman GPU kernels to PTX with the FMA-capable cuda-oxide
# fork backend and refresh the checked-in copy used by the library at runtime.
#
# Prerequisites:
#   - the FMA-capable codegen backend .so (a STALE cache emits no fma.rn.f64).
#     Build it once with:
#       cd <cuda-oxide>/crates/rustc-codegen-cuda \
#         && CARGO_TARGET_DIR=/home/cargo-targets/cuda-oxide-codegen \
#            cargo +nightly-2026-04-03 build
#   - cargo-oxide on PATH and a CUDA toolkit at $CUDA_TOOLKIT_PATH (build only).
#
# Usage:  CUDA_OXIDE_BACKEND=/path/to/librustc_codegen_cuda.so ./build-ptx.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
backend="${CUDA_OXIDE_BACKEND:-/home/cargo-targets/cuda-oxide-codegen/debug/librustc_codegen_cuda.so}"

if [[ ! -f "$backend" ]]; then
  echo "error: codegen backend not found at $backend" >&2
  echo "set CUDA_OXIDE_BACKEND to the FMA-capable librustc_codegen_cuda.so" >&2
  exit 1
fi

export CUDA_OXIDE_BACKEND="$backend"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/home/cargo-targets/vk-kernels}"

echo "building kernels with backend: $backend"
( cd "$here" && cargo oxide build )

ptx="$here/vonkarman_kernels.ptx"
fma="$(grep -c 'fma.rn.f64' "$ptx" || true)"
echo "emitted $ptx (fma.rn.f64 count: $fma)"
if [[ "$fma" -eq 0 ]]; then
  echo "warning: no fma.rn.f64 in PTX; backend is likely STALE (un-fused)" >&2
fi

dest="$here/../src/ptx/vonkarman_kernels.ptx"
cp "$ptx" "$dest"
echo "refreshed checked-in PTX: $dest"
