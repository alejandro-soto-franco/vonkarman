# vonkarman

[![CI](https://github.com/alejandro-soto-franco/vonkarman/actions/workflows/ci.yml/badge.svg)](https://github.com/alejandro-soto-franco/vonkarman/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-2024_edition-orange.svg)](https://www.rust-lang.org/)
![Crates](https://img.shields.io/badge/crates-6-blue)

Multi-precision pseudospectral Navier-Stokes solver with topological diagnostics.

## Overview

vonkarman is a Rust DNS (direct numerical simulation) solver for the incompressible 3D Navier-Stokes equations on the periodic torus. It targets the quantitative diagnostics defined in a 3D Navier-Stokes regularity project, computing energy spectra, conservation laws, and (in future phases) vortex line topology and reconnection dominance ratios.

The solver uses a pseudospectral method with ETD-RK4 time integration, dealiased via $3/2$ zero-padding, with adaptive CFL timestepping.

## Architecture

```
vonkarman/
  vonkarman-core/       Float trait, field types, SpectralOps, Domain trait
  vonkarman-fft/        FFT backends (ndrustfft CPU; cuFFT/VkFFT/FFTW planned)
  vonkarman-periodic/   Periodic 3D pseudospectral solver (ETD-RK4 + RK4)
  vonkarman-diag/       Scalar diagnostics, energy spectrum, conservation audit
  vonkarman-io/         HDF5 snapshots
  vonkarman-bin/        CLI binary (TOML config, Parquet output)
  vonkarman-2d/         2D pseudospectral vorticity-streamfunction solver, dyes, penalised bluff bodies
```

## Quick start

```bash
cargo build --release

# Create an experiment config
cat > tg_re1600.toml << 'EOF'
[run]
name = "tg-re1600"
output_dir = "./output/tg-re1600"

[domain]
type = "periodic3d"
n = 128

[physics]
nu = 6.25e-4

[initial_condition]
type = "taylor-green"

[termination]
max_steps = 10000
max_time = 10.0
EOF

# Run the simulation
./target/release/vonkarman run --config tg_re1600.toml
```

Output goes to `./output/tg-re1600/diagnostics.parquet`, queryable with DuckDB or Polars.

## Initial conditions

| IC | Description | Key use |
|----|-------------|---------|
| Taylor-Green | $\sin(x)\cos(y)\cos(z)$ standard benchmark | Validation, energy decay |
| ABC/Beltrami | Eigenfunction of curl, $\nabla \times \mathbf{u} = \mathbf{u}$ | Helicity conservation |
| Anti-parallel tubes | Two Lamb-Oseen tubes with perturbation | Vortex reconnection |
| Kida-Pelz | Octahedral symmetry | Depletion, high-symmetry blowup |
| Random isotropic | Von Karman spectrum, Leray-projected | Developed turbulence |

### Vorticity nulls on mesh points

The unperturbed Taylor-Green datum places its vorticity nulls exactly on
the mesh points of a uniform periodic grid with even `n`. Every diagnostic
built from the vorticity direction field divides by $|\omega|$ and is
singular at a null, so a node-collocated null is evaluated AT the
singularity. At $32^3$ the direction energy $\langle|\nabla\xi|^2\rangle$
reads `1.96e29` there against `5.55` for the same field sampled a half cell
away.

`shift` samples the field at $x_i = (i + \mathrm{shift})h$, in cells, which
moves the mesh off the null planes and leaves the flow alone:

```toml
[initial_condition]
type = "taylor-green"
shift = 0.5
```

`shift` defaults to `0`, which reproduces the canonical benchmark, nulls
and all. Every frame reports `min_vorticity`, `null_cell_margin` and
`null_fraction`, and a run whose margin falls below `1e-3` cells logs a
warning naming the condition once.

The hazard generalises to any symmetric datum whose critical points land on
mesh points. Taylor-Green is the common one.

## Time integration

Two integrators:

1. **ETD-RK4** (default): Exponential time differencing with Kassam-Trefethen contour integral $\varphi$-functions. Treats viscous term exactly, eliminating stiffness at high wavenumbers.

2. **Classical RK4**: Standard 4th-order Runge-Kutta for cross-validation. Requires smaller timesteps due to stiff viscous term.

## Diagnostics

Every timestep (Tier 1):
- Kinetic energy, enstrophy, helicity, superhelicity
- Maximum vorticity magnitude $\|\boldsymbol{\omega}\|_\infty$
- Energy and helicity dissipation rates
- CFL number

Periodic (Tier 2):
- Shell-averaged energy spectrum $E(k)$
- Compensated spectrum $k^{5/3} E(k)$
- Dissipation spectrum $2\nu k^2 E(k)$

Conservation audit runs automatically, flagging energy increases or NaN.

## Snapshots

HDF5 snapshots contain full solver state:
- Physical-space velocity and vorticity
- Spectral coefficients (exact restart)
- Grid and physics metadata

## Precision

Currently `f64`. The `Float` trait is designed for multi-precision: double-double (~31 digits), quad-double (~62 digits), and MPFR (arbitrary) are planned.

## Dependencies

- Rust 2024 edition
- ndarray, ndrustfft (pure-Rust FFT)
- Arrow + Parquet (diagnostics output)
- HDF5 (snapshots, via hdf5-metno)
- clap (CLI), toml (config), tracing (logging)

## Tests

```bash
# Unit tests (~75s)
cargo test --lib

# Full suite including integration tests (>10 min; use -p <crate> for a single crate)
cargo test
```

55+ tests covering FFT roundtrip/Parseval, spectral operators, IC analytical values, energy decay, dissipation rate consistency, Parquet I/O, HDF5 snapshots, and end-to-end CLI.

## License

MIT
