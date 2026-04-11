# vonkarman Memory

## 2026-04-11: Wolf initialised
- anatomy.md, cerebrum.md, memory.md, buglog.json created
- CLAUDE.md added to repo root
- Current state: v0.1.0 released (Phases 1-2 complete), 61 tests passing
- Next work: cuFFT GPU backend (6-task plan at docs/superpowers/plans/2026-04-08-cufft-backend.md)

## 2026-04-11: cuFFT backend implemented (Tasks 1-5 done)
- Task 1: libloading dep + CufftError type in cufft.rs
- Task 2: CufftBackend with device memory, FFT plans, FftBackend impl
- Task 3: BackendMode + create_backend factory in select.rs
- Task 4: Periodic3D refactored to Box<dyn FftBackend<f64>>
- Task 5: --backend CLI flag + TOML backend config
- 66 tests passing, 3 GPU tests passing (ignored in CI), clippy clean
- Remaining: Task 6 (CI verify + push + tag v0.2.0)

## 2026-04-11: Operational guarantees, precision, benchmarks implemented
- Checkpoint write/read (vonkarman-io/src/checkpoint.rs) + Periodic3D::from_checkpoint
- Signal handling (SIGINT/SIGTERM graceful shutdown with emergency checkpoint)
- Input validation (vonkarman-bin/src/validate.rs, 7 checks)
- Hardened ConservationAudit with AuditConfig (energy budget, divergence, Parseval tolerances)
- --restart CLI flag for checkpoint restart
- Spectral convergence test (N=16..128, exponential error decay verified)
- Parseval identity test (residual < 1e-12 at N=8,16,32)
- Energy budget closure test (midpoint enstrophy, 90%+ steps within 60% tolerance)
- Helicity conservation test (ABC flow)
- Checkpoint-restart bitwise identity test
- Reference data test (Re=1600 N=128, Brachet enstrophy peak, ignored/long-running)
- benchmarks/ directory with hit3d, spectralDNS, Dedalus, TurboGenPY, vonkarman scripts
- run_all.py orchestrator (build/run/compare/report)
- 64 unit tests + 3 GPU tests passing, clippy clean
