# Changelog

## Unreleased

### 2D bluff bodies
- `Spectral2D` generalised from a square `n x n` box over `[0, 2 pi)^2` to `nx x ny` over `[0, lx) x [0, ly)`, with physical wavenumbers and the 2/3 dealias mask kept in mode-index space. `Spectral2D::new_square` preserves the old constructor.
- `Penalisation`: Brinkman volume penalisation of a stationary circular cylinder with a `tanh`-smoothed mask, applied as an exact exponential substep Strang split around IF-RK4, so the penalisation parameter is free of the step size. Vorticity is reformed by a spectral curl, which doubles as the Helmholtz projection.
- Downstream fringe relaxing vorticity to zero, so a wake does not re-enter the periodic box as inflow.
- `Sim::set_mean_flow` for a uniform stream. It carries no vorticity, so it is held outside the state and folded in where it acts: advection, the penalisation substep (where the mask gradient crossed with the stream is what generates vorticity at the body), `Sim::total_velocity` and `Sim::streamfunction`.
- `Sim::streamfunction` returning the Poisson solve plus the analytic `u_mean * y` mean part.
- `.npy` frame export (`psi`, `omega`, `speed`, `mask`) with a `meta.json` sidecar, and a `cylinder` driver binary taking a TOML config.
- Validation against published benchmarks: `C_d` and recirculation length at Re 40, Strouhal number at Re 100, both ignored by default.
- Fixed a latent index transposition in `write_dye_png` that was masked by the grid being square.
- `cylinder` can now checkpoint and resume, so a long run survives an interruption. Three changes affect any reader of a run's output directory:
  - `meta.json` gains a `complete` key (boolean). It is `false` from the moment the run starts until the process reaches its own configured step count, then `true`. A reader that builds a fixed-field record from `meta.json` and does not expect this key will raise a type error on it; add the field.
  - While `complete` is `false`, `frames` is a planned count, not the number of files actually written yet: an interrupted run can leave fewer `.npy` files than `frames` claims. Trust the files present in the directory, not `frames`, until `complete` is `true`.
  - A new `checkpoint.bin` file may appear in the output directory. It is solver state for resuming, not a rendered field, and nothing downstream needs to read it. Delete it once the run you intended has finished, not merely once this process's own `complete` reads `true`: a checkpoint-and-resume split writes `complete: true` at the end of its first half too, meaning only that the first process reached its own configured step count, and deleting `checkpoint.bin` at that point discards the state the second half needs to finish the run.

### Resident GPU memory
- ETD-RK4 stage fold: the resident solver now holds three spectral stage triples instead of four by folding `n23 = n2 + n3` in place after the stage-4 combination and reusing the freed buffer for `n4`. Two new `#[cuda_module]` kernels (`cplx_add_assign`, a bit-identical `add.f64`; `etd_final_folded`) regenerated into the checked-in PTX via `cargo oxide build`. The folded final reproduces the four-buffer `etd_final` value to the CPU FMA reference at the same tolerance (not bit-identical to the GPU `etd_final` kernel, which the backend regroups sub-ULP); the full resident step still matches the CPU solver to 1.52e-15. Buffer-only footprint drops ~387 MiB at 256^3 (6.420 to 6.042 GiB) and ~48 MiB at 128^3.

## 0.2.0 (2026-04-11)

### cuFFT GPU backend
- `CufftBackend`: runtime-loaded cuFFT/CUDA via `libloading` (no compile-time CUDA dependency)
- `BackendMode` (Auto/Cufft/Cpu) with `create_backend` factory and auto-fallback
- `Periodic3D` refactored to `Box<dyn FftBackend<f64>>` for backend polymorphism
- `--backend` CLI flag and TOML `backend` config field

### Operational guarantees
- Checkpoint write/read (HDF5) with bitwise-exact roundtrip
- `Periodic3D::from_checkpoint` constructor for exact solver restart
- `--restart` CLI flag for checkpoint restart
- SIGINT/SIGTERM graceful shutdown with emergency checkpoint
- Input config validation (power-of-2 grid, nu>0, termination, backend, CFL)
- Periodic checkpoint writing at configurable intervals

### Hardened conservation monitoring
- `AuditConfig` with configurable tolerances for energy budget, divergence-free, and Parseval identity checks
- `halt_on_violation` option for strict mode
- TOML-configurable tolerance fields

### Precision tests
- Spectral convergence (Taylor-Green N=8..64, energy error < 5e-4)
- Parseval identity verification (residual < 1e-12 at N=8,16,32)
- Energy budget closure (midpoint enstrophy, 100% steps within tolerance)
- Helicity conservation (ABC flow viscous decay)
- Checkpoint-restart bitwise identity
- Reference data validation (Taylor-Green Re=1600 N=128 vs Brachet 1983, ignored/long-running)

### Benchmark infrastructure
- Cross-solver harness: hit3d (Fortran), spectralDNS (Python), Dedalus (Python), TurboGenPY (Python)
- `run_all.py` orchestrator with build/run/compare/report subcommands
- Digitised Brachet 1983 and van Rees 2011 reference data
- Matplotlib figure generation (energy, enstrophy, dissipation, performance)

## 0.1.0 (2026-04-06)

### Phase 1: Core solver

- Cargo workspace with 6 crates (`core`, `fft`, `periodic`, `diag`, `io`, `bin`)
- `Float` trait with f64 implementation (multi-precision ready)
- Complex number utilities, Kahan-compensated summation
- `GridSpec`, `ScalarField`, `VectorField` field types
- `SpectralOps`: wavenumber arrays, spectral curl, Leray projection, viscous operator
- `Domain` trait and `PhysicsParams` for solver abstraction
- `FftBackend` trait with pure-Rust ndrustfft CPU backend
- Dealiased cross product via 3/2 zero-padding
- ETD-RK4 time integrator with Kassam-Trefethen contour integral phi-functions
- Taylor-Green vortex initial condition
- Nonlinear term in rotation form (spectral curl + dealiased cross + Leray)
- `Periodic3D` solver with adaptive CFL timestepping
- Scalar diagnostics (energy, enstrophy, helicity, superhelicity, max vorticity)
- Conservation audit (energy monotonicity, NaN detection)
- TOML experiment configuration parsing
- Parquet diagnostics time series writer (Arrow + ZSTD compression)
- `vonkarman run` CLI subcommand with termination conditions
- Integration tests: energy decay, exponential short-time agreement, CLI end-to-end

### Phase 2: Extended ICs, diagnostics, I/O

- Shell-averaged energy spectrum E(k) with compensated and dissipation spectra
- ABC/Beltrami flow IC (eigenfunction of curl, helicity conservation test)
- Anti-parallel vortex tubes IC (Kerr/Hou-Li reconnection geometry)
- Kida-Pelz high-symmetry IC (octahedral group, depletion studies)
- Random isotropic turbulence IC (von Karman spectrum, Leray-projected, seeded RNG)
- Classical RK4 integrator (cross-validation against ETD-RK4)
- `vonkarman-io` crate with HDF5 snapshot writer/reader (velocity, vorticity, spectral state, metadata)
