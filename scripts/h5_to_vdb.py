#!/usr/bin/env python
"""Convert vonkarman checkpoint HDF5 files to an OpenVDB sequence of |omega|.

Checkpoints store the spectral velocity (/spectral/u_hat_{re,im}_0..2, rFFT
layout) plus grid metadata. We inverse-FFT to physical velocity, take the
spectral curl for vorticity, and write the vorticity magnitude as a dense
FloatGrid named "density" that Blender imports directly as a volume. Only the
field STRUCTURE matters for the render, so a global FFT-normalisation constant
is irrelevant (we rescale by the global max anyway).

Run it on the run's checkpoint_*.h5 (set checkpoint_interval to your frame
cadence in the .toml).

Deps (conda-forge, off-snapshot via pixi):
    pixi add h5py numpy openvdb
Run:
    pixi run python scripts/h5_to_vdb.py output/antiparallel vdb_out
    # -> vdb_out/frame_0001.vdb ... numbered by step for Blender.
"""
import sys
import glob
import os
import numpy as np
import h5py
import openvdb as vdb


def vort_mag(h5) -> np.ndarray:
    uhat = [
        h5[f"spectral/u_hat_re_{c}"][:] + 1j * h5[f"spectral/u_hat_im_{c}"][:]
        for c in range(3)
    ]
    # Dimensions straight from the rFFT array shape (last axis = nz//2+1), so we
    # never depend on the metadata layout. Box length L (cubic) from attrs, else
    # the default 2*pi; a wrong-but-uniform L is a global scale we normalise out.
    snx, sny, snz = uhat[0].shape
    nx, ny, nz = snx, sny, (snz - 1) * 2
    attrs = h5["metadata"].attrs if "metadata" in h5 else {}
    lx = float(attrs.get("lx", 2 * np.pi))
    ly = float(attrs.get("ly", 2 * np.pi))
    lz = float(attrs.get("lz", 2 * np.pi))
    # Angular wavenumbers for the rFFT layout (last axis halved).
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    kz = 2 * np.pi * np.fft.rfftfreq(nz, d=lz / nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    # omega_hat = i k x u_hat
    ox = 1j * (KY * uhat[2] - KZ * uhat[1])
    oy = 1j * (KZ * uhat[0] - KX * uhat[2])
    oz = 1j * (KX * uhat[1] - KY * uhat[0])
    s = (nx, ny, nz)
    wx = np.fft.irfftn(ox, s=s)
    wy = np.fft.irfftn(oy, s=s)
    wz = np.fft.irfftn(oz, s=s)
    return np.sqrt(wx * wx + wy * wy + wz * wz).astype(np.float32)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: h5_to_vdb.py <checkpoint_dir> <vdb_out_dir>")
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(os.path.join(in_dir, "**", "checkpoint_*.h5"), recursive=True)
    if not files:
        sys.exit(f"no checkpoint_*.h5 under {in_dir} (set checkpoint_interval in the .toml)")

    frames, gmax = [], 0.0
    for f in files:
        with h5py.File(f, "r") as h5:
            attrs = h5["metadata"].attrs if "metadata" in h5 else {}
            step = int(attrs.get("step_count", "".join(filter(str.isdigit, os.path.basename(f))) or 0))
            wmag = vort_mag(h5)
        gmax = max(gmax, float(wmag.max()))
        frames.append((step, wmag))
    frames.sort(key=lambda t: t[0])
    scale = 1.0 / gmax if gmax > 0 else 1.0
    print(f"{len(frames)} frames, global max |omega| = {gmax:.4g}")

    for i, (step, wmag) in enumerate(frames, start=1):
        grid = vdb.FloatGrid()
        grid.copyFromArray(np.ascontiguousarray(wmag * scale))
        grid.name = "density"
        out = os.path.join(out_dir, f"frame_{i:04d}.vdb")
        vdb.write(out, grids=[grid])
        print(f"  step {step:>6} -> {out}")


if __name__ == "__main__":
    main()
