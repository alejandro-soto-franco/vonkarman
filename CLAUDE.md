# vonkarman

Multi-precision pseudospectral Navier-Stokes DNS solver on the periodic torus T^3.

## Build & Test

```bash
cargo build --release                        # full workspace build
cargo test --lib                             # unit tests (~25s)
cargo test --workspace                       # full suite incl. integration (~45s)
cargo clippy --workspace -- -D warnings      # lint (zero warnings required)
cargo fmt --check                            # format check
```

## Run a simulation

```bash
./target/release/vonkarman run --config experiment.toml
```

Output: `<output_dir>/diagnostics.parquet` (queryable with DuckDB/Polars).

## Architecture

6-crate workspace:

| Crate | Purpose |
|-------|---------|
| `vonkarman-core` | Float trait, GridSpec, ScalarField/VectorField, SpectralOps, Domain |
| `vonkarman-fft` | FftBackend trait, NdrustfftBackend (CPU), dealiased cross product |
| `vonkarman-periodic` | Periodic3D solver (ETD-RK4 + RK4), 5 initial conditions |
| `vonkarman-diag` | ScalarDiagnostics, EnergySpectrum, ConservationAudit |
| `vonkarman-io` | HDF5 snapshot writer/reader |
| `vonkarman-bin` | CLI (clap), TOML config, Parquet writer |

Solver state lives in **spectral space** (`u_hat`). Physical-space fields are derived via inverse FFT. Dealiasing uses the 3/2 rule with a separate padded FFT backend.

## Key conventions

- Year in all docs and headers: **2026** (never 2025)
- Edition: Rust 2024
- Error handling: `thiserror` preferred, `anyhow` secondary
- Heavy doc comments on every public function
- Conventional commit messages: `feat/`, `fix/`, `patch/`, `chore/`, `docs/`
- No Co-Authored-By lines in commits
- No em dashes in any output
- Integration tests in `<crate>/tests/` directories
- Check `.wolf/buglog.json` before fixing bugs; log fixes after

## Current development

**Next milestone**: cuFFT GPU backend (v0.2.0)
- Plan: `docs/superpowers/plans/2026-04-08-cufft-backend.md` (6 tasks, all unchecked)
- Runtime-loads `libcufft.so` + `libcudart.so` via `libloading` (no compile-time CUDA dep)
- Auto-fallback to CPU when CUDA unavailable
- Refactors `Periodic3D` from concrete `NdrustfftBackend` to `Box<dyn FftBackend<f64>>`
- Adds `--backend` CLI flag and TOML `backend` field

## Wolf

Check `.wolf/anatomy.md` before reading project files. Update `.wolf/` after changes.
