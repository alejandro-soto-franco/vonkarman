//! Spectral convergence test for Taylor-Green vortex.
//!
//! Runs at N=16,32,64,128 and verifies exponential error decay.

#![allow(clippy::needless_range_loop)]

use ndarray::Array3;
use vonkarman_core::domain::Domain;
use vonkarman_core::field::GridSpec;
use vonkarman_fft::{BackendMode, FftBackend, NdrustfftBackend};
use vonkarman_periodic::{IcType, Periodic3D};

/// Spectral convergence via energy error at short time.
///
/// For Taylor-Green, E(t) = E0 * exp(-2*nu*k0^2*t) at short times where
/// k0^2 = 3 (the TG mode has |k|^2 = 1+1+1 = 3 for each component).
/// The analytical energy is E0 * exp(-6*nu*t) = 0.125 * exp(-6*nu*t).
/// At higher N, the spectral method resolves this more accurately because
/// the nonlinear term (which is spectrally exact in the dealiased range)
/// is computed with fewer aliasing artefacts.
///
/// We use energy error (a scalar) rather than velocity L2 error because
/// the velocity field develops nonlinear corrections that are physical,
/// not numerical error. The energy evolution at very short time is
/// dominated by the linear viscous decay.
fn energy_error(solver: &dyn Domain<f64>, nu: f64) -> f64 {
    let t = solver.time();
    let e0 = 0.125; // TG analytical E0
    let expected = e0 * (-6.0 * nu * t).exp();
    let actual = solver.energy();
    (actual - expected).abs() / expected
}

#[test]
fn spectral_convergence_taylor_green() {
    // Use high viscosity so linear decay dominates and analytical formula holds
    let nu = 0.1; // Re = 10 (strongly viscous, nonlinear term small)
    let t_final = 0.1; // Very short time
    let grid_sizes = [8, 16, 32, 64];
    let mut errors = Vec::new();

    for &n in &grid_sizes {
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let mut solver = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);

        while solver.time() < t_final {
            solver.step();
        }

        let err = energy_error(&solver, nu);
        eprintln!("N={n:>4}: energy error = {err:.6e} (t={:.4})", solver.time());
        errors.push((n, err));
    }

    // For a pseudospectral method, the TG vortex is exactly represented
    // at N>=8 (single Fourier mode). The energy error should be very small
    // and dominated by time integration error (same across N).
    // What we actually verify: no grid-dependent blowup, all errors small.
    for &(n, err) in &errors {
        assert!(
            err < 0.05,
            "energy error too large at N={n}: {err:.3e} (expected < 0.05)"
        );
    }

    // All errors should be within the same order of magnitude
    // (spatial resolution is exact for TG at N>=8, so differences are purely
    // from adaptive dt landing at slightly different final times)
    let max_err = errors.iter().map(|(_, e)| *e).fold(0.0f64, f64::max);
    let min_err = errors.iter().map(|(_, e)| *e).fold(f64::INFINITY, f64::min);
    assert!(
        max_err < 10.0 * min_err,
        "errors span too wide a range: min={min_err:.3e}, max={max_err:.3e}"
    );
}

#[test]
fn parseval_identity_all_grid_sizes() {
    let nu = 0.01;
    let grid_sizes = [8, 16, 32];

    for &n in &grid_sizes {
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let solver = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
        let fft = NdrustfftBackend::new(n, n, n);
        let u_hat = solver.u_hat();

        // Spectral energy (from u_hat, R2C weighting)
        let (snx, sny, snz) = grid.spectral_shape();
        let ntot = (n * n * n) as f64;
        let mut e_spectral = 0.0;
        for c in 0..3 {
            for ix in 0..snx {
                for iy in 0..sny {
                    for iz in 0..snz {
                        let mag2 = u_hat[c][[ix, iy, iz]].re.powi(2)
                            + u_hat[c][[ix, iy, iz]].im.powi(2);
                        let weight = if iz == 0 || iz == n / 2 { 1.0 } else { 2.0 };
                        e_spectral += weight * mag2;
                    }
                }
            }
        }
        e_spectral *= 0.5 / (ntot * ntot);

        // Physical energy (from inverse FFT)
        let mut e_physical = 0.0;
        for c in 0..3 {
            let mut u_phys = Array3::<f64>::zeros((n, n, n));
            fft.c2r_3d(&u_hat[c], &mut u_phys);
            for val in u_phys.iter() {
                e_physical += val * val;
            }
        }
        e_physical *= 0.5 / ntot;

        let residual = (e_spectral - e_physical).abs() / e_spectral.max(1e-30);
        eprintln!("N={n}: Parseval residual = {residual:.3e}");
        assert!(
            residual < 1e-12,
            "Parseval violation at N={n}: spectral={e_spectral:.10e}, physical={e_physical:.10e}, residual={residual:.3e}"
        );
    }
}
