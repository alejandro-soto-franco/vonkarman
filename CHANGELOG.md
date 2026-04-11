# Changelog

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
