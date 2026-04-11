# vonkarman Anatomy

## Workspace Layout (6 crates)

```
vonkarman/
  Cargo.toml                    # Workspace manifest (resolver 2, edition 2024)
  vonkarman-core/src/
    lib.rs                      # Re-exports: float, field, spectral_ops, domain, complex_utils, kahan
    float.rs                    # Float trait (f64 impl; multi-precision ready)
    field.rs                    # GridSpec, ScalarField, VectorField
    spectral_ops.rs             # SpectralOps: wavenumber arrays, curl, Leray projection, viscous op
    domain.rs                   # Domain trait, DomainType, PhysicsParams, Snapshot
    complex_utils.rs            # Complex number helpers
    kahan.rs                    # Kahan-compensated summation
  vonkarman-fft/src/
    lib.rs                      # Re-exports: FftBackend, NdrustfftBackend, CufftBackend, BackendMode, create_backend
    backend.rs                  # FftBackend<F: Float> trait (r2c_3d, c2r_3d, name, precision_digits)
    ndrustfft_backend.rs        # NdrustfftBackend: pure-Rust CPU FFT via ndrustfft
    cufft.rs                    # CufftBackend: GPU FFT via runtime-loaded libcufft.so/libcudart.so
    select.rs                   # BackendMode (Auto/Cufft/Cpu) + create_backend factory
    dealiased.rs                # dealiased_cross_product (3/2 zero-padding)
  vonkarman-periodic/src/
    lib.rs                      # Re-exports: Periodic3D, IcType
    solver.rs                   # Periodic3D struct (ETD-RK4, adaptive CFL dt)
    etd.rs                      # EtdCoeffs: Kassam-Trefethen contour integral phi-functions
    nonlinear.rs                # compute_nonlinear (rotation form: curl + cross + Leray)
    rk4.rs                      # Classical RK4 integrator
    ic/
      mod.rs                    # IcType enum, IC generators
      taylor_green.rs           # Taylor-Green vortex
      abc.rs                    # ABC/Beltrami flow
      anti_parallel_tubes.rs    # Anti-parallel Lamb-Oseen tubes
      kida_pelz.rs              # Kida-Pelz high-symmetry
      random_isotropic.rs       # Random isotropic (von Karman spectrum)
  vonkarman-diag/src/
    lib.rs                      # ScalarDiagnostics, EnergySpectrum, ConservationAudit
  vonkarman-io/src/
    lib.rs                      # HDF5 snapshot writer/reader
  vonkarman-bin/src/
    main.rs                     # CLI entry (clap): `vonkarman run --config <toml>`
    config.rs                   # ExperimentConfig TOML deserialization
    run.rs                      # Simulation orchestration loop
    diagnostics_writer.rs       # Parquet (Arrow + ZSTD) time-series writer
  vonkarman-bin/src/
    validate.rs                 # Input config validation (n power-of-2, nu>0, backend, termination)
  benchmarks/
    run_all.py                  # Orchestrator: build/run/compare/report across all solvers
    requirements.txt            # Python deps (numpy, pandas, matplotlib, h5py)
    reference_data/             # Digitised Brachet 1983 + van Rees 2011 curves
    hit3d/                      # JHU Fortran DNS: build.sh, run.sh, parse_output.py
    spectraldns/                # Mortensen Python DNS: setup.sh, run_tg.py
    dedalus/                    # Dedalus spectral PDE: setup.sh, run_tg.py
    turbogenpy/                 # Saad IC generator: setup.sh, run_spectrum.py
    vonkarman/                  # Our solver: tg_re1600.toml, run_tg.sh, run_tg_gpu.sh
  docs/superpowers/plans/
    2026-04-05-vonkarman-design.md
    2026-04-06-vonkarman-phase1.md
    2026-04-08-cufft-backend.md
    2026-04-11-operational-precision-benchmarks.md
  docs/superpowers/specs/
    2026-04-11-operational-precision-benchmarks-design.md
```

## Key Types

- `Float` trait (vonkarman-core): abstraction over f64, future dd/qd/mpfr
- `GridSpec`: nx/ny/nz, dx, spectral_shape(), padded_3half()
- `SpectralOps<F>`: kx/ky/kz arrays, k_mag_sq, spectral_curl, leray_project
- `FftBackend<F>` trait: r2c_3d, c2r_3d (NdrustfftBackend + CufftBackend)
- `BackendMode`: Auto/Cufft/Cpu; `create_backend()` factory with auto-fallback
- `CufftBackend`: runtime-loaded CUDA FFT (libloading, no compile-time CUDA dep)
- `Periodic3D`: solver struct holding u_hat[3], ops, etd_coeffs, fft, fft_padded (Box<dyn FftBackend>)
- `EtdCoeffs`: phi_1/phi_2/phi_3 per wavenumber for ETD-RK4
- `IcType`: enum of initial conditions
- `ScalarDiagnostics`: energy, enstrophy, helicity, superhelicity, max_vorticity, CFL
- `EnergySpectrum`: shell-averaged E(k), compensated, dissipation spectrum

## Data Flow

1. Config (TOML) -> ExperimentConfig -> GridSpec + PhysicsParams + IcType
2. IC generates physical velocity -> FFT -> u_hat (spectral state)
3. Each timestep: ETD-RK4 or RK4 advances u_hat
   - Nonlinear: spectral curl -> physical vorticity, physical velocity -> dealiased cross -> Leray project
   - Linear: exact exponential integration (ETD) or explicit (RK4)
4. Diagnostics computed from u_hat every step -> Parquet
5. HDF5 snapshots at configured intervals

## Critical Invariants

- Solver state lives in spectral space (u_hat); physical space is derived
- Dealiasing uses 3/2 rule (separate fft_padded backend)
- ETD coefficients depend on dt; recomputed when dt changes (adaptive CFL)
- cuFFT inverse is unnormalized; 1/N^3 applied in c2r_3d
