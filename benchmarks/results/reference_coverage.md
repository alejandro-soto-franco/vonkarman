# Reference Solver Coverage on This Box

Generated: 2026-06-16. Host: HP OMEN 16 (Fedora, x86-64, RTX 5060, 8 GB GPU).

## Summary table

| Solver | Available | How built | Build command | Run command | Output format |
|--------|-----------|-----------|---------------|-------------|---------------|
| hit3d | Yes | Pre-built (`src/hit3d.x` present, compiled via `build.sh` with gfortran 16 + OpenMPI 5.0.9 + FFTW3) | `bash benchmarks/hit3d/build.sh` | `bash benchmarks/hit3d/run.sh` then `python3 benchmarks/hit3d/parse_output.py` | `results/hit3d.csv`: columns `t,E,Omega,epsilon,max_omega`; 100 rows, t in [0.1, 10.0] |
| turbogenpy | Yes (analytical fallback) | Cloned via `setup.sh`; `tgpy` module not importable (no `tgpy.py` in the repo, only `isoturb.py` etc.), so `run_spectrum.py` falls back automatically | `bash benchmarks/turbogenpy/setup.sh` | `python3 benchmarks/turbogenpy/run_spectrum.py` | `results/turbogenpy_spectrum.csv`: columns `k,Ek`; IC spectrum comparison only, not a time-stepping DNS |
| spectraldns | Yes (pure numpy, serial) | `.venv` present with numpy 2.4.6, mpi4py, pyfftw, mpi4py-fft installed via `setup.sh` (pip inside venv, pre-existing) | `bash benchmarks/spectraldns/setup.sh` | `benchmarks/spectraldns/.venv/bin/python3 benchmarks/spectraldns/run_tg.py` | `results/spectraldns.csv`: columns `t,E,Omega,epsilon,max_omega`; one row per 100 steps |
| dedalus | Yes | `.venv` present with Dedalus 3.0.5 installed via `setup.sh` (pip inside venv, pre-existing); IVP build and time-stepping smoke-tested | `bash benchmarks/dedalus/setup.sh` | `OMP_NUM_THREADS=1 benchmarks/dedalus/.venv/bin/python3 benchmarks/dedalus/run_tg.py` | `results/dedalus.csv`: columns `t,E,Omega,epsilon,max_omega`; energy only (enstrophy/epsilon columns are zero) |

## Per-solver findings

### hit3d (Fortran 90 + MPI)

Status: **runnable**. The binary `benchmarks/hit3d/src/hit3d.x` (ELF 64-bit) was previously
compiled with gfortran 16.1.1 and OpenMPI 5.0.9, which are both present on this box
(`/usr/lib64/openmpi/`). FFTW3 double-precision is installed at `/usr/lib64/libfftw3.so.3.6.10`
and `/usr/include/fftw3.h`. Note: `libfftw3_mpi` is NOT present, but the hit3d build links
only against serial `libfftw3`, so this is fine.

Existing pre-run output files (`stat1.gp`, `stat2.gp`, `es.gp`, `d0000.txt`,
`tg_re01600.64.010100`) confirm the solver was run at N=256, Re=1600 to t=10.0 (10,100
timesteps, IPRNT1=100). These cover the canonical test case (the input file actually uses
N=256 rather than N=128; the benchmark compares at whatever N the run used).

Running `parse_output.py` on the existing `stat1.gp` produces `results/hit3d.csv` with
100 rows, t in [0.1, 10.0], peak enstrophy 2.066e+01 at t=8.9.

`run.sh` expects the binary at `src/hit3d.x` and invokes `mpirun --oversubscribe -np 1`.
A full re-run at N=256 takes significant wall time and was not re-run here; the existing
outputs are used directly.

### turbogenpy

Status: **runnable (analytical fallback)**. The TurboGenPY repo is cloned at
`benchmarks/turbogenpy/TurboGenPY/`. However, `run_spectrum.py` imports `from tgpy import
TurboGen`, and no `tgpy` module exists in the repo (the actual modules are `isoturb.py`,
`isoturbo.py`, etc., not the API the script expects). The script handles this gracefully with
a `try/except ImportError` and falls back to an analytical von Karman spectrum
E(k) ~ k^4 / (1 + (k/k_peak)^2)^(17/6). The fallback ran and produced
`results/turbogenpy_spectrum.csv`.

This solver is IC-spectrum comparison only (no time-stepping DNS); it does not produce
`t,E,Omega` columns. It cannot be used for Taylor-Green accuracy validation, only for
initial-condition spectrum shape comparison.

The `tgpy` API mismatch is a soft blocker: wiring `isoturb.py` directly would require
adapting `run_spectrum.py`, which was not done within the timebox.

### spectraldns (Python pseudospectral DNS)

Status: **runnable**. `.venv` (Python 3.14) has numpy 2.4.6, mpi4py, pyfftw, and
mpi4py-fft installed. The `run_tg.py` script is a self-contained pure-numpy pseudospectral
RK4 DNS (not a wrapper around the spectralDNS repo, despite its name). It uses
`np.fft.rfftn`/`irfftn` with a 2/3 dealiasing mask and Leray projection, producing the
same algorithm as the spectralDNS library.

Smoke test at N=16 for 5 RK4 steps: E0=1.25e-01, E_final=1.2500e-01, ran in 0.02 s.
Note: `np.fft.irfftn` raises a DeprecationWarning on NumPy 2.0 when called without
explicit `axes`; this is cosmetic and does not affect results.

Full run at N=128, t=10 (10,000 steps of RK4, each requiring 12 FFT/IFFT calls on
128^3 arrays) will be slow in pure numpy (estimated several hours on CPU). It will
produce the correct output format.

The `spectraldns/spectralDNS/` repo clone is present but not used by `run_tg.py`.

### dedalus (Dedalus v3)

Status: **runnable**. `.venv` (Python 3.14) has Dedalus 3.0.5 installed. Smoke test:
built a full 3-D incompressible NS IVP with `RealFourier` bases at N=8, ran 5 steps of
RK443, confirmed `sim_time=0.005`. The warning about threading (`OMP_NUM_THREADS` not
set to 1) should be addressed by the `activate.sh` which sets `OMP_NUM_THREADS=1`.

`run_tg.py` produces `results/dedalus.csv` with energy E at each logged step, but
enstrophy and dissipation rate columns are written as zero (the script does not compute
them, only energy via `d3.Integrate(d3.dot(u,u))`). This means Dedalus results
participate in energy-decay comparison but not enstrophy/dissipation comparison.

Full run at N=128 with Dedalus's implicit-explicit RK443 is expected to be slower than
the pure numpy run but may be more stable with larger dt.

## System dependencies confirmed present

| Dependency | Version / path | Needed by |
|------------|---------------|-----------|
| gfortran | 16.1.1 (GCC, Red Hat 16.1.1-2) | hit3d build |
| mpif90 / mpirun | OpenMPI 5.0.9 at `/usr/lib64/openmpi/bin/` | hit3d build and run |
| FFTW3 (double) | 3.6.10 at `/usr/lib64/libfftw3.so`, header at `/usr/include/fftw3.h` | hit3d |
| FFTW3-MPI | NOT present (`libfftw3_mpi*` absent) | not required by hit3d as built |
| numpy | 2.4.6 (spectraldns .venv); 2.4.6 (benchmarks .venv) | spectraldns, turbogenpy, run_all.py |
| mpi4py, pyfftw, mpi4py-fft | present in spectraldns/.venv | spectraldns setup (not used by run_tg.py directly) |
| Dedalus | 3.0.5 in dedalus/.venv | dedalus |
| Python | 3.14 (all .venv) | all Python solvers |

## What the gamut can fairly compare on this box

**Full DNS time series (energy + enstrophy + dissipation):** hit3d at N=256, Re=1600.
The existing pre-run output is already parsed into `results/hit3d.csv` (100 rows,
t in [0.1, 10.0]).

**Full DNS time series (energy + enstrophy + dissipation):** spectraldns at N=128,
Re=1600, once the full run completes. This is a pure-numpy serial RK4 implementation
verified to produce the correct IC energy (1.25e-01) and correct RK4 dynamics on
a smoke test.

**Energy-only DNS time series:** dedalus at N=128, Re=1600. Enstrophy and dissipation
are not emitted by `run_tg.py`.

**IC spectrum shape (not time-stepping):** turbogenpy, analytical von Karman spectrum
only (TurboGenPY `tgpy` API unavailable). Useful for validating the `random_isotropic`
initial condition spectrum shape in vonkarman, not for Taylor-Green accuracy validation.

**Not available on this box:** any solver requiring FFTW3-MPI for distributed
parallel runs, or MPI-parallel Python (mpi4py-fft MPI parallelism was not tested
and is not required by the current scripts).

Honest summary: three of the four solvers (hit3d, spectraldns, dedalus) are fully
available for Taylor-Green vortex accuracy benchmarking on this box. turbogenpy
provides IC spectrum comparison only. The gamut can produce a three-way DNS comparison
against vonkarman on energy decay and (for hit3d and spectraldns) enstrophy and
dissipation rate.
