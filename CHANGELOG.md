# Changelog

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
