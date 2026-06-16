//! Brachet / van Rees Re=1600 Taylor-Green validation on the DEVICE-RESIDENT
//! GPU solver.
//!
//! Reproduces the same Taylor-Green vortex, the same adaptive CFL schedule, and
//! the same acceptance gates as the CPU `reference_data` test, but time-steps
//! entirely on the resident GPU path (`ResidentSolver`). The per-step nonlinear,
//! FFTs, ETD-RK4 combination AND the CFL timestep run on device; only a periodic
//! diagnostic download of `u_hat` crosses the bus (allowed). Gated behind the
//! `cuda` feature and SKIPS gracefully when no GPU / CUDA toolkit is present.
//!
//! Run on the RTX 5060 with:
//!
//! ```text
//! CUDA_TOOLKIT_PATH=/usr/local/cuda \
//!   cargo +nightly-2026-04-03 test -p vonkarman-periodic --features cuda \
//!   --test resident_reference -- --nocapture --ignored
//! ```

#![cfg(feature = "cuda")]
// The spectral diagnostics loop over the 3 components and the (ix, iy, iz)
// triple-nested ranges, indexing several arrays per iteration; an iterator
// rewrite would obscure the index arithmetic, so keep the explicit ranges.
#![allow(clippy::needless_range_loop)]

use num_complex::Complex;
use vonkarman_compute::CudaBackend;
use vonkarman_core::domain::Domain;
use vonkarman_core::field::GridSpec;
use vonkarman_core::spectral_ops::SpectralOps;
use vonkarman_fft::BackendMode;
use vonkarman_periodic::{IcType, Periodic3D, ResidentSolver};

/// Energy `E = (1/2)(1/N^3) sum_w |u_hat|^2` with the R2C half-spectrum weight,
/// identical to `Periodic3D::energy`, from a host copy of `u_hat`.
fn energy_from_uhat(u_hat: &[Vec<Complex<f64>>; 3], grid: &GridSpec) -> f64 {
    let (snx, sny, snz) = grid.spectral_shape();
    let ntot = (grid.nx * grid.ny * grid.nz) as f64;
    let mut e = 0.0_f64;
    for comp in u_hat.iter() {
        let mut idx = 0;
        for _ix in 0..snx {
            for _iy in 0..sny {
                for iz in 0..snz {
                    let z = comp[idx];
                    let mag2 = z.re * z.re + z.im * z.im;
                    let weight = if iz == 0 || iz == grid.nz / 2 {
                        1.0
                    } else {
                        2.0
                    };
                    e += weight * mag2;
                    idx += 1;
                }
            }
        }
    }
    0.5 * e / (ntot * ntot)
}

/// Enstrophy `Z = (1/N^3) sum_w |omega_hat|^2`, omega = curl(u), identical to
/// `Periodic3D::enstrophy`, from a host copy of `u_hat`.
fn enstrophy_from_uhat(
    u_hat: &[Vec<Complex<f64>>; 3],
    ops: &SpectralOps<f64>,
    grid: &GridSpec,
) -> f64 {
    use ndarray::Array3;
    let (snx, sny, snz) = grid.spectral_shape();
    let shape = (snx, sny, snz);
    // Rebuild Array3 from the flat host vectors (C-order [ix, iy, iz]).
    let mut uh: [Array3<Complex<f64>>; 3] = [
        Array3::zeros(shape),
        Array3::zeros(shape),
        Array3::zeros(shape),
    ];
    for c in 0..3 {
        let mut idx = 0;
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    uh[c][[ix, iy, iz]] = u_hat[c][idx];
                    idx += 1;
                }
            }
        }
    }
    let zero = Complex { re: 0.0, im: 0.0 };
    let mut omega = [
        Array3::from_elem(shape, zero),
        Array3::from_elem(shape, zero),
        Array3::from_elem(shape, zero),
    ];
    ops.curl(&uh, &mut omega);
    let ntot = (grid.nx * grid.ny * grid.nz) as f64;
    let mut ens = 0.0_f64;
    for c in 0..3 {
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    let o = omega[c][[ix, iy, iz]];
                    let mag2 = o.re * o.re + o.im * o.im;
                    let weight = if iz == 0 || iz == grid.nz / 2 {
                        1.0
                    } else {
                        2.0
                    };
                    ens += weight * mag2;
                }
            }
        }
    }
    ens / (ntot * ntot)
}

/// Resident-vs-CPU TG Re1600 equivalence at N=48: the GPU resident solver and
/// the CPU `Periodic3D` are each adaptively stepped from the same Taylor-Green
/// state, and their energy and enstrophy are compared at checkpoints. This
/// proves the resident path reproduces the validated CPU physics over a real
/// (not single-step) integration. It is the rigorous cross-validation behind the
/// minutes-long N=128 Brachet run.
///
/// `#[ignore]`: each backend kernel call currently reloads the PTX module, so a
/// 60-step GPU march plus a parallel CPU march is too slow for the default
/// suite. Run explicitly with `--ignored --nocapture`.
#[test]
#[ignore]
fn resident_matches_cpu_trajectory_48() {
    let backend = match CudaBackend::new(0) {
        Ok(be) => be,
        Err(e) => {
            eprintln!("skipping resident-vs-CPU trajectory: CudaBackend::new(0) failed: {e}");
            return;
        }
    };

    let n = 48;
    let nu = 6.25e-4; // Re = 1600 (the reference physics)
    let cfl_safety = 0.5;
    let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
    let ops = SpectralOps::<f64>::new(&grid);
    let cplx_len = {
        let (a, b, c) = grid.spectral_shape();
        a * b * c
    };

    // CPU oracle and resident solver from the SAME TG initial state.
    let mut cpu = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
    let u0 = cpu.u_hat();
    let u_hat_flat: [Vec<Complex<f64>>; 3] = [
        u0[0].iter().copied().collect(),
        u0[1].iter().copied().collect(),
        u0[2].iter().copied().collect(),
    ];
    let mut solver = ResidentSolver::new(backend, grid, nu, &u_hat_flat);

    let mut host: [Vec<Complex<f64>>; 3] = [
        vec![Complex::new(0.0, 0.0); cplx_len],
        vec![Complex::new(0.0, 0.0); cplx_len],
        vec![Complex::new(0.0, 0.0); cplx_len],
    ];

    // Each integrator advances with its own adaptive dt (the resident CFL for the
    // GPU, the CPU's own CFL for the CPU). They track because both integrate the
    // same PDE from the same state with the same ETD-RK4 scheme and near-equal
    // dt; the per-step kernels already match the CPU to 1.5e-15.
    let mut max_e_rel = 0.0_f64;
    let mut max_z_rel = 0.0_f64;
    for k in 0u64..60 {
        let dt = solver.cfl_dt(cfl_safety, nu);
        solver.step(dt);
        cpu.step();

        if (k + 1).is_multiple_of(10) {
            solver.download_u_hat_all(&mut host);
            let e_gpu = energy_from_uhat(&host, &grid);
            let z_gpu = enstrophy_from_uhat(&host, &ops, &grid);
            let e_cpu = cpu.energy();
            let z_cpu = cpu.enstrophy();
            let e_rel = (e_gpu - e_cpu).abs() / e_cpu.abs().max(1e-30);
            let z_rel = (z_gpu - z_cpu).abs() / z_cpu.abs().max(1e-30);
            max_e_rel = max_e_rel.max(e_rel);
            max_z_rel = max_z_rel.max(z_rel);
            eprintln!(
                "step {}: E gpu {e_gpu:.8e} cpu {e_cpu:.8e} (rel {e_rel:e}); Z gpu {z_gpu:.6e} cpu {z_cpu:.6e} (rel {z_rel:e})",
                k + 1
            );
        }
    }
    // The resident and CPU trajectories agree to a tight relative tolerance over
    // 60 steps (both adaptive, same scheme, same IC). 1e-6 absorbs the dt-
    // schedule and FMA-rounding differences accumulated over the run.
    assert!(
        max_e_rel < 1e-6 && max_z_rel < 1e-6,
        "resident vs CPU trajectory diverged: max energy rel {max_e_rel:e}, max enstrophy rel {max_z_rel:e}"
    );
    eprintln!(
        "resident vs CPU agree over 60 steps: max E rel {max_e_rel:e}, max Z rel {max_z_rel:e}"
    );
}

/// Fast resident smoke at N=32: the resident CFL dt must match the CPU
/// construction dt, and a short resident march must keep energy monotone with no
/// NaN. Catches CFL / stepping regressions without the minutes-long N=128 run.
#[test]
fn resident_cfl_and_decay_smoke_32() {
    let backend = match CudaBackend::new(0) {
        Ok(be) => be,
        Err(e) => {
            eprintln!("skipping resident smoke: CudaBackend::new(0) failed: {e}");
            return;
        }
    };

    let n = 32;
    let nu = 0.01;
    let cfl_safety = 0.5;
    let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
    let ops = SpectralOps::<f64>::new(&grid);
    let cplx_len = {
        let (a, b, c) = grid.spectral_shape();
        a * b * c
    };

    let cpu = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
    let dt0_cpu = cpu.dt();
    let u0 = cpu.u_hat();
    let u_hat_flat: [Vec<Complex<f64>>; 3] = [
        u0[0].iter().copied().collect(),
        u0[1].iter().copied().collect(),
        u0[2].iter().copied().collect(),
    ];
    let e0 = energy_from_uhat(&u_hat_flat, &grid);

    let mut solver = ResidentSolver::new(backend, grid, nu, &u_hat_flat);
    let dt0_gpu = solver.cfl_dt(cfl_safety, nu);
    let rel = (dt0_gpu - dt0_cpu).abs() / dt0_cpu;
    eprintln!("smoke CFL dt: gpu {dt0_gpu:.6e} vs cpu {dt0_cpu:.6e} (rel {rel:e})");
    assert!(rel < 1e-9, "resident CFL dt mismatch at t=0: rel {rel:e}");

    let mut host: [Vec<Complex<f64>>; 3] = [
        vec![Complex::new(0.0, 0.0); cplx_len],
        vec![Complex::new(0.0, 0.0); cplx_len],
        vec![Complex::new(0.0, 0.0); cplx_len],
    ];
    let mut prev = e0;
    let mut dt = dt0_gpu;
    for _ in 0..40 {
        let new_dt = solver.cfl_dt(cfl_safety, nu);
        if (new_dt - dt).abs() / dt.max(1e-30) > 0.01 {
            dt = new_dt;
        }
        solver.step(dt);
        solver.download_u_hat_all(&mut host);
        let e = energy_from_uhat(&host, &grid);
        let z = enstrophy_from_uhat(&host, &ops, &grid);
        assert!(e.is_finite() && z.is_finite(), "NaN in resident smoke");
        assert!(
            e <= prev + 1e-12 * prev.abs().max(1e-30),
            "energy increased in resident smoke: {prev} -> {e}"
        );
        prev = e;
    }
    assert!(
        prev < e0,
        "resident smoke energy did not decay: {e0} -> {prev}"
    );
    eprintln!("resident smoke OK: E0 {e0:.6e} -> E {prev:.6e}");
}

/// Full Brachet / van Rees Re=1600 Taylor-Green validation on the resident GPU
/// path at N=128, asserting the SAME gates as the CPU `reference_data` test.
/// Reports the measured peak enstrophy, peak epsilon, and peak time.
#[test]
#[ignore] // GPU + minutes at N=128; run with --ignored --nocapture
fn taylor_green_re1600_resident_128() {
    let backend = match CudaBackend::new(0) {
        Ok(be) => be,
        Err(e) => {
            eprintln!("skipping resident reference test: CudaBackend::new(0) failed: {e}");
            return;
        }
    };

    let n = 128;
    let nu = 6.25e-4; // Re = 1600
    let t_final = 10.0;
    let cfl_safety = 0.5;
    let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
    let ops = SpectralOps::<f64>::new(&grid);
    let cplx_len = {
        let (a, b, c) = grid.spectral_shape();
        a * b * c
    };

    // CPU solver only to generate the TG initial spectral state and the
    // construction-time dt (identical IC path to reference_data). We do NOT step
    // the CPU; the resident GPU solver carries the integration.
    let cpu = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
    let dt0_cpu = cpu.dt();
    let u0 = cpu.u_hat();
    let u_hat_flat: [Vec<Complex<f64>>; 3] = [
        u0[0].iter().copied().collect(),
        u0[1].iter().copied().collect(),
        u0[2].iter().copied().collect(),
    ];
    let e0 = energy_from_uhat(&u_hat_flat, &grid);

    let mut solver = ResidentSolver::new(backend, grid, nu, &u_hat_flat);

    // Sanity: the resident CFL dt at t=0 must match the CPU construction dt (the
    // same scheme on the same field), to a tight relative tolerance.
    let dt0_gpu = solver.cfl_dt(cfl_safety, nu);
    let dt0_rel = (dt0_gpu - dt0_cpu).abs() / dt0_cpu;
    eprintln!("resident CFL dt at t=0: gpu {dt0_gpu:.6e} vs cpu {dt0_cpu:.6e} (rel {dt0_rel:e})");
    assert!(
        dt0_rel < 1e-9,
        "resident CFL dt diverged from CPU at t=0: {dt0_gpu} vs {dt0_cpu} (rel {dt0_rel:e})"
    );

    // Adaptive ETD-RK4 march on the resident path, mirroring the CPU schedule:
    // recompute dt from the resident CFL each step and rebuild ETD coefficients
    // only when dt changes by more than 1% (the CPU `Domain::step` rule).
    let mut time = 0.0_f64;
    let mut dt = dt0_gpu;
    let mut step = 0u64;

    let mut peak_enstrophy = 0.0_f64;
    let mut peak_time = 0.0_f64;
    let mut peak_epsilon = 0.0_f64;
    let mut prev_energy = e0;

    let mut host: [Vec<Complex<f64>>; 3] = [
        vec![Complex::new(0.0, 0.0); cplx_len],
        vec![Complex::new(0.0, 0.0); cplx_len],
        vec![Complex::new(0.0, 0.0); cplx_len],
    ];

    eprintln!("t,energy,enstrophy,epsilon");

    while time < t_final {
        // Resident CFL dt (device max-reduction, no field pull).
        let new_dt = solver.cfl_dt(cfl_safety, nu);
        if (new_dt - dt).abs() / dt.max(1e-30) > 0.01 {
            dt = new_dt;
        }
        solver.step(dt);
        time += dt;
        step += 1;

        // Diagnostic pull every 25 steps (and on the first step), to track the
        // enstrophy peak and confirm monotone energy decay without pulling the
        // field every step.
        if step.is_multiple_of(25) || step == 1 {
            solver.download_u_hat_all(&mut host);
            let e = energy_from_uhat(&host, &grid);
            let z = enstrophy_from_uhat(&host, &ops, &grid);
            let epsilon = nu * z; // dissipation rate nu*<|omega|^2> (z is <|omega|^2>)

            assert!(e.is_finite(), "NaN/Inf energy at step {step}, t={time}");
            assert!(
                e <= prev_energy + 1e-10 * prev_energy.abs().max(1e-30),
                "energy increased at step {step}: {prev_energy} -> {e} (t={time})"
            );

            if z > peak_enstrophy {
                peak_enstrophy = z;
                peak_time = time;
                peak_epsilon = epsilon;
            }

            if step.is_multiple_of(500) || step == 1 {
                eprintln!("{time:.4},{e:.8e},{z:.8e},{epsilon:.8e}");
            }
            prev_energy = e;
        }
    }

    solver.download_u_hat_all(&mut host);
    let final_energy = energy_from_uhat(&host, &grid);

    eprintln!("\n=== Resident GPU results (N={n}, Re=1600) ===");
    eprintln!("Peak enstrophy: {peak_enstrophy:.6e} at t={peak_time:.4}");
    eprintln!("Peak dissipation rate (epsilon): {peak_epsilon:.6e}");
    eprintln!("Final energy: {final_energy:.6e} (E0 = {e0:.6e})");
    eprintln!("Total steps: {step}");

    // Brachet et al. (1983): enstrophy peaks around t ~ 8-10. Same window as the
    // CPU reference_data gate.
    assert!(
        peak_time > 7.0 && peak_time < 11.0,
        "enstrophy peak at t={peak_time}, expected in [7, 11]"
    );

    // Peak dissipation rate gate, IDENTICAL to the CPU reference_data test:
    // epsilon = nu*<|omega|^2> in [0.005, 0.015]. With the correct convention the
    // N=128 peak is ~1.03e-2 (enstrophy ~16.5 at t~8.68), comfortably inside the
    // band. (Before the dissipation factor was fixed the diagnostic reported
    // 2*nu*z ~ 2.07e-2 and appeared to overshoot; the 256^3 run showed the gap
    // was this factor of two plus genuine under-resolution, epsilon rising 0.0103
    // at N=128 to 0.0113 at N=256 toward the Brachet/van Rees 0.0126. See
    // RESULTS.md.)
    assert!(
        peak_epsilon > 0.005 && peak_epsilon < 0.015,
        "peak epsilon={peak_epsilon:.6e}, expected in [0.005, 0.015]"
    );

    // Energy should have decayed significantly (E0 = 1/8 analytical).
    assert!(
        final_energy < 0.5 * e0,
        "insufficient energy decay: E0={e0}, E_final={final_energy}"
    );
}

/// Full 256^3 Taylor-Green Re=1600 dissipation-peak run on the resident GPU,
/// integrated to t=10 to capture the peak dissipation rate epsilon(t) and compare
/// it against the N=128 resident run and the Brachet 1983 / van Rees 2011
/// literature band (peak epsilon ~ 0.0126 near t~9).
///
/// This is the resolution study behind the honest accuracy caveat in RESULTS.md:
/// at N=128 BOTH vonkarman and hit3d over-predict the peak (epsilon ~ 0.020),
/// which the docs attribute to under-resolution. This run integrates the better-
/// resolved N=256 grid to completion and REPORTS the peak (it does not assert the
/// literature band; the point is to measure where N=256 lands). It prints the
/// epsilon(t) curve so the peak can be read off and plotted.
///
/// About half a GPU-hour at N=256 (~900 adaptive steps near 1.5 s each); run
/// explicitly and detached. Emits `CURVE,t,energy,enstrophy,epsilon` rows and a
/// `PEAK256` summary line.
#[test]
#[ignore] // GPU, ~0.5 hour at N=256 to t=10; run with --ignored --nocapture
fn taylor_green_re1600_resident_256_dissipation() {
    let backend = match CudaBackend::new(0) {
        Ok(be) => be,
        Err(e) => {
            eprintln!("skipping 256^3 dissipation run: CudaBackend::new(0) failed: {e}");
            return;
        }
    };

    let n = 256;
    let nu = 6.25e-4; // Re = 1600
    let t_final = 10.0;
    let cfl_safety = 0.5;
    let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
    let ops = SpectralOps::<f64>::new(&grid);
    let cplx_len = {
        let (a, b, c) = grid.spectral_shape();
        a * b * c
    };

    // TG IC via the CPU IC path (host only); the resident GPU carries the march.
    let cpu = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
    let dt0_cpu = cpu.dt();
    let u0 = cpu.u_hat();
    let u_hat_flat: [Vec<Complex<f64>>; 3] = [
        u0[0].iter().copied().collect(),
        u0[1].iter().copied().collect(),
        u0[2].iter().copied().collect(),
    ];
    let e0 = energy_from_uhat(&u_hat_flat, &grid);

    let mut solver = ResidentSolver::new(backend, grid, nu, &u_hat_flat);

    let dt0_gpu = solver.cfl_dt(cfl_safety, nu);
    let dt0_rel = (dt0_gpu - dt0_cpu).abs() / dt0_cpu;
    eprintln!("resident CFL dt at t=0: gpu {dt0_gpu:.6e} vs cpu {dt0_cpu:.6e} (rel {dt0_rel:e})");
    assert!(dt0_rel < 1e-9, "resident CFL dt diverged from CPU at t=0");

    let mut time = 0.0_f64;
    let mut dt = dt0_gpu;
    let mut step = 0u64;
    let mut peak_enstrophy = 0.0_f64;
    let mut peak_time = 0.0_f64;
    let mut peak_epsilon = 0.0_f64;
    let mut prev_energy = e0;

    let mut host: [Vec<Complex<f64>>; 3] = [
        vec![Complex::new(0.0, 0.0); cplx_len],
        vec![Complex::new(0.0, 0.0); cplx_len],
        vec![Complex::new(0.0, 0.0); cplx_len],
    ];

    eprintln!("CURVE,t,energy,enstrophy,epsilon");
    let t_run = std::time::Instant::now();

    while time < t_final {
        let new_dt = solver.cfl_dt(cfl_safety, nu);
        if (new_dt - dt).abs() / dt.max(1e-30) > 0.01 {
            dt = new_dt;
        }
        solver.step(dt);
        time += dt;
        step += 1;

        // Diagnostic pull every 20 steps (and on the first) to resolve the peak.
        if step.is_multiple_of(20) || step == 1 {
            solver.download_u_hat_all(&mut host);
            let e = energy_from_uhat(&host, &grid);
            let z = enstrophy_from_uhat(&host, &ops, &grid);
            let epsilon = nu * z; // dissipation rate nu*<|omega|^2> (z is <|omega|^2>)
            assert!(
                e.is_finite() && z.is_finite(),
                "NaN/Inf at step {step}, t={time}"
            );
            assert!(
                e <= prev_energy + 1e-9 * prev_energy.abs().max(1e-30),
                "energy increased at step {step}: {prev_energy} -> {e} (t={time})"
            );
            if z > peak_enstrophy {
                peak_enstrophy = z;
                peak_time = time;
                peak_epsilon = epsilon;
            }
            eprintln!("CURVE,{time:.4},{e:.8e},{z:.8e},{epsilon:.8e}");
            prev_energy = e;
        }
    }

    solver.download_u_hat_all(&mut host);
    let final_energy = energy_from_uhat(&host, &grid);
    let run_secs = t_run.elapsed().as_secs_f64();

    eprintln!("\n=== Resident GPU 256^3 dissipation (Re=1600, to t={t_final}) ===");
    eprintln!(
        "PEAK256,peak_enstrophy={peak_enstrophy:.6e},peak_epsilon={peak_epsilon:.6e},peak_time={peak_time:.4},final_energy={final_energy:.6e},E0={e0:.6e},steps={step},wall_s={run_secs:.1}"
    );
    // epsilon above is the standard dissipation rate nu*z with z = <|omega|^2>
    // (its t=0 value 0.75 matches the analytic Taylor-Green <|omega|^2> =
    // 1/8+1/8+1/2 = 3/4). Compare to the literature in this convention.
    eprintln!(
        "Compare: N=128 peak nu*z ~1.03e-2 @ t~8.68; N=256 here; Brachet/van Rees ~1.26e-2 @ t~9 (converging from below)."
    );

    // Sanity only (the literature band is REPORTED, not gated, for this study).
    assert!(
        peak_time > 7.0 && peak_time < 11.0,
        "enstrophy peak at t={peak_time}, expected in [7, 11]"
    );
    assert!(
        peak_enstrophy.is_finite() && peak_epsilon > 0.0,
        "no finite peak captured"
    );
    // Taylor-Green Re=1600 dissipates only ~41% of its energy by t=10 (the peak
    // dissipation is at t~9), so E(10) ~ 0.59 E0. A "< 0.5 E0" expectation is the
    // wrong physics; assert significant decay with a realistic bound (per-step
    // monotonicity is already enforced inside the loop above).
    assert!(
        final_energy < 0.7 * e0,
        "expected significant energy decay by t=10: E0={e0}, E_final={final_energy}"
    );
}

/// 256^3 Taylor-Green on the resident path: proves the allocation AND a
/// sustained sequence of resident steps fit inside the 8 GB RTX 5060, with no
/// NaN and monotone energy through the early enstrophy rise. This is the memory
/// fit + stability proof, NOT a full t=10 integration (that is the separate
/// final-gamut timing step). Instruments ACTUAL device usage via cudaMemGetInfo:
/// free before allocation, and the minimum free seen during stepping (peak use).
#[test]
#[ignore] // GPU, 8 GB fit + a stretch of steps at N=256; run with --ignored --nocapture
fn taylor_green_resident_256_fits_8gb() {
    let backend = match CudaBackend::new(0) {
        Ok(be) => be,
        Err(e) => {
            eprintln!("skipping 256^3 fit test: CudaBackend::new(0) failed: {e}");
            return;
        }
    };

    let n = 256;
    let nu = 6.25e-4; // Re = 1600
    let cfl_safety = 0.5;
    let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
    let ops = SpectralOps::<f64>::new(&grid);
    let gib = 1024.0 * 1024.0 * 1024.0;

    // Free/total BEFORE any of our allocations.
    let (free_before, total) = backend
        .mem_get_info()
        .expect("cudaMemGetInfo (before) failed");
    eprintln!(
        "device memory before alloc: free {:.3} GiB / total {:.3} GiB",
        free_before as f64 / gib,
        total as f64 / gib
    );

    // TG IC via the CPU IC path (host only); upload once into the resident solver.
    let cpu = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
    let u0 = cpu.u_hat();
    let u_hat_flat: [Vec<Complex<f64>>; 3] = [
        u0[0].iter().copied().collect(),
        u0[1].iter().copied().collect(),
        u0[2].iter().copied().collect(),
    ];
    let cplx_len = {
        let (a, b, c) = grid.spectral_shape();
        a * b * c
    };
    let e0 = energy_from_uhat(&u_hat_flat, &grid);
    let z0 = enstrophy_from_uhat(&u_hat_flat, &ops, &grid);

    let mut solver = ResidentSolver::new(backend, grid, nu, &u_hat_flat);

    // Analytic estimate (incl. cuFFT work areas, queried live).
    let est = solver.peak_memory_bytes();
    let pwork = solver.padded_workarea_bytes();
    eprintln!(
        "analytic peak estimate at N={n}: {:.3} GiB (shared padded cuFFT work area: {:.4} GiB)",
        est as f64 / gib,
        pwork as f64 / gib
    );

    // Free right after construction (all resident buffers + cuFFT plans live).
    let (free_after_alloc, _) = solver.mem_get_info();
    let used_alloc = free_before.saturating_sub(free_after_alloc);
    eprintln!(
        "device memory after construction: free {:.3} GiB (used by us so far: {:.3} GiB)",
        free_after_alloc as f64 / gib,
        used_alloc as f64 / gib
    );
    assert!(
        free_after_alloc as f64 / gib > 0.05,
        "construction left < 50 MiB free; N=256 does not fit"
    );

    // March a sustained sequence of resident steps through the early enstrophy
    // rise. Track the WORST-CASE (minimum) free memory across the run as the
    // peak usage, plus energy monotonicity and NaN-freedom.
    let mut host: [Vec<Complex<f64>>; 3] = [
        vec![Complex::new(0.0, 0.0); cplx_len],
        vec![Complex::new(0.0, 0.0); cplx_len],
        vec![Complex::new(0.0, 0.0); cplx_len],
    ];

    let target_time = 4.0_f64; // through the early rise toward the t~9 peak
    let max_steps = 240u64; // bound the wall time; report whatever we reach
    let mut time = 0.0_f64;
    let mut dt = solver.cfl_dt(cfl_safety, nu);
    let mut step = 0u64;
    let mut min_free = free_after_alloc;
    let mut prev_energy = e0;
    let mut peak_enstrophy = 0.0_f64;
    let mut peak_time = 0.0_f64;

    eprintln!("t,energy,enstrophy,free_GiB");
    while time < target_time && step < max_steps {
        let new_dt = solver.cfl_dt(cfl_safety, nu);
        if (new_dt - dt).abs() / dt.max(1e-30) > 0.01 {
            dt = new_dt;
        }
        solver.step(dt);
        time += dt;
        step += 1;

        // Peak device usage is the minimum free seen during stepping (the CFL
        // path allocates small reduction scratch transiently each step).
        let (free_now, _) = solver.mem_get_info();
        if free_now < min_free {
            min_free = free_now;
        }

        if step.is_multiple_of(10) || step == 1 {
            solver.download_u_hat_all(&mut host);
            let e = energy_from_uhat(&host, &grid);
            let z = enstrophy_from_uhat(&host, &ops, &grid);
            assert!(
                e.is_finite() && z.is_finite(),
                "NaN/Inf at step {step}, t={time}"
            );
            assert!(
                e <= prev_energy + 1e-9 * prev_energy.abs().max(1e-30),
                "energy increased at step {step}: {prev_energy} -> {e} (t={time})"
            );
            if z > peak_enstrophy {
                peak_enstrophy = z;
                peak_time = time;
            }
            eprintln!("{time:.4},{e:.8e},{z:.8e},{:.3}", free_now as f64 / gib);
            prev_energy = e;
        }
    }

    let peak_used = free_before.saturating_sub(min_free);
    eprintln!("\n=== Resident GPU 256^3 fit proof ===");
    eprintln!(
        "reached t={time:.4} in {step} resident steps (target t={target_time}, step cap {max_steps})"
    );
    eprintln!(
        "PEAK device usage: {:.3} GiB (min free {:.3} GiB of {:.3} GiB total)",
        peak_used as f64 / gib,
        min_free as f64 / gib,
        total as f64 / gib
    );
    eprintln!(
        "enstrophy rose to {peak_enstrophy:.6e} by t={peak_time:.4} (Z0 = {z0:.6e}, still rising)"
    );
    eprintln!("energy: E0 {e0:.6e} -> E {prev_energy:.6e} (monotone, NaN-free)");

    // The whole resident set + cuFFT work areas + transient CFL scratch fit the
    // 8 GB card with headroom (min free stayed positive throughout).
    assert!(
        min_free as f64 / gib > 0.05,
        "N=256 exhausted device memory: min free {:.3} GiB",
        min_free as f64 / gib
    );
    // Energy strictly decreased over the run (no spurious injection).
    assert!(
        prev_energy < e0,
        "energy did not decay over the 256^3 run: {e0} -> {prev_energy}"
    );
    // The early enstrophy rise is present (production of small scales).
    assert!(
        peak_enstrophy > 2.0 * z0,
        "no enstrophy rise observed at N=256 over t in [0, {time:.2}]"
    );
}
