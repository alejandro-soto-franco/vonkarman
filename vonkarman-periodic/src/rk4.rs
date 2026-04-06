use crate::nonlinear::compute_nonlinear;
use ndarray::Array3;
use num_complex::Complex;
use vonkarman_core::field::GridSpec;
use vonkarman_core::spectral_ops::SpectralOps;
use vonkarman_fft::FftBackend;

/// Classical RK4 time integrator for the Navier-Stokes equations.
///
/// Treats viscous + nonlinear terms together (no exponential integrating factor).
/// Less stable than ETD-RK4 for stiff diffusion at high wavenumbers, but useful
/// as a validation cross-check.
///
/// Advances: du_hat/dt = L*u_hat + N(u_hat)
/// where L = -nu*|k|^2 (diagonal viscous operator) and N is the nonlinear term.
pub fn rk4_step(
    u_hat: &mut [Array3<Complex<f64>>; 3],
    ops: &SpectralOps<f64>,
    fft: &dyn FftBackend<f64>,
    fft_padded: &dyn FftBackend<f64>,
    grid: &GridSpec,
    nu: f64,
    dt: f64,
) {
    let (snx, sny, snz) = grid.spectral_shape();
    let shape = (snx, sny, snz);
    let zero = Complex { re: 0.0, im: 0.0 };

    // Compute full RHS = L*u + N(u) for a given state
    let compute_rhs = |state: &[Array3<Complex<f64>>; 3], rhs: &mut [Array3<Complex<f64>>; 3]| {
        // Nonlinear term
        compute_nonlinear(ops, fft, fft_padded, grid, state, rhs);
        // Add viscous term: -nu * |k|^2 * u_hat
        for c in 0..3 {
            for ix in 0..snx {
                for iy in 0..sny {
                    for iz in 0..snz {
                        let visc = -nu * ops.k_mag_sq[[ix, iy, iz]];
                        rhs[c][[ix, iy, iz]].re += visc * state[c][[ix, iy, iz]].re;
                        rhs[c][[ix, iy, iz]].im += visc * state[c][[ix, iy, iz]].im;
                    }
                }
            }
        }
    };

    // k1 = dt * f(u_n)
    let mut k1 = [
        Array3::from_elem(shape, zero),
        Array3::from_elem(shape, zero),
        Array3::from_elem(shape, zero),
    ];
    compute_rhs(u_hat, &mut k1);

    // temp = u_n + 0.5 * dt * k1
    let mut temp = [
        Array3::from_elem(shape, zero),
        Array3::from_elem(shape, zero),
        Array3::from_elem(shape, zero),
    ];
    for c in 0..3 {
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    temp[c][[ix, iy, iz]] = Complex {
                        re: u_hat[c][[ix, iy, iz]].re + 0.5 * dt * k1[c][[ix, iy, iz]].re,
                        im: u_hat[c][[ix, iy, iz]].im + 0.5 * dt * k1[c][[ix, iy, iz]].im,
                    };
                }
            }
        }
    }

    // k2 = dt * f(u_n + 0.5*dt*k1)
    let mut k2 = [
        Array3::from_elem(shape, zero),
        Array3::from_elem(shape, zero),
        Array3::from_elem(shape, zero),
    ];
    compute_rhs(&temp, &mut k2);

    // temp = u_n + 0.5 * dt * k2
    for c in 0..3 {
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    temp[c][[ix, iy, iz]] = Complex {
                        re: u_hat[c][[ix, iy, iz]].re + 0.5 * dt * k2[c][[ix, iy, iz]].re,
                        im: u_hat[c][[ix, iy, iz]].im + 0.5 * dt * k2[c][[ix, iy, iz]].im,
                    };
                }
            }
        }
    }

    // k3 = dt * f(u_n + 0.5*dt*k2)
    let mut k3 = [
        Array3::from_elem(shape, zero),
        Array3::from_elem(shape, zero),
        Array3::from_elem(shape, zero),
    ];
    compute_rhs(&temp, &mut k3);

    // temp = u_n + dt * k3
    for c in 0..3 {
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    temp[c][[ix, iy, iz]] = Complex {
                        re: u_hat[c][[ix, iy, iz]].re + dt * k3[c][[ix, iy, iz]].re,
                        im: u_hat[c][[ix, iy, iz]].im + dt * k3[c][[ix, iy, iz]].im,
                    };
                }
            }
        }
    }

    // k4 = dt * f(u_n + dt*k3)
    let mut k4 = [
        Array3::from_elem(shape, zero),
        Array3::from_elem(shape, zero),
        Array3::from_elem(shape, zero),
    ];
    compute_rhs(&temp, &mut k4);

    // u_{n+1} = u_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    let sixth_dt = dt / 6.0;
    for c in 0..3 {
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    u_hat[c][[ix, iy, iz]].re += sixth_dt
                        * (k1[c][[ix, iy, iz]].re
                            + 2.0 * k2[c][[ix, iy, iz]].re
                            + 2.0 * k3[c][[ix, iy, iz]].re
                            + k4[c][[ix, iy, iz]].re);
                    u_hat[c][[ix, iy, iz]].im += sixth_dt
                        * (k1[c][[ix, iy, iz]].im
                            + 2.0 * k2[c][[ix, iy, iz]].im
                            + 2.0 * k3[c][[ix, iy, iz]].im
                            + k4[c][[ix, iy, iz]].im);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vonkarman_core::field::GridSpec;
    use vonkarman_core::spectral_ops::SpectralOps;
    use vonkarman_fft::{FftBackend, NdrustfftBackend};

    #[test]
    fn rk4_energy_decays() {
        // Run RK4 on Taylor-Green and verify energy decays.
        let n = 16;
        let nu = 0.01;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let ops = SpectralOps::<f64>::new(&grid);
        let fft = NdrustfftBackend::new(n, n, n);
        let pg = grid.padded_3half();
        let fft_padded = NdrustfftBackend::new(pg.nx, pg.ny, pg.nz);

        // Taylor-Green IC
        let v = crate::ic::taylor_green::<f64>(&grid);
        let (snx, sny, snz) = grid.spectral_shape();
        let shape = (snx, sny, snz);
        let mut u_hat: [Array3<Complex<f64>>; 3] = [
            Array3::zeros(shape),
            Array3::zeros(shape),
            Array3::zeros(shape),
        ];
        for c in 0..3 {
            fft.r2c_3d(&v.data[c], &mut u_hat[c]);
        }

        let energy_of = |u: &[Array3<Complex<f64>>; 3]| -> f64 {
            let ntot = (n * n * n) as f64;
            let mut e = 0.0_f64;
            for c in 0..3 {
                for ix in 0..snx {
                    for iy in 0..sny {
                        for iz in 0..snz {
                            let mag2 =
                                u[c][[ix, iy, iz]].re.powi(2) + u[c][[ix, iy, iz]].im.powi(2);
                            let weight = if iz == 0 || iz == n / 2 { 1.0 } else { 2.0 };
                            e += weight * mag2;
                        }
                    }
                }
            }
            0.5 * e / (ntot * ntot)
        };

        let e0 = energy_of(&u_hat);
        let dt = 0.001; // small dt for stability

        let mut prev_e = e0;
        for _ in 0..20 {
            rk4_step(&mut u_hat, &ops, &fft, &fft_padded, &grid, nu, dt);
            let e = energy_of(&u_hat);
            assert!(
                e <= prev_e + 1e-12 * prev_e.abs().max(1e-30),
                "RK4 energy increased: {prev_e} -> {e}"
            );
            prev_e = e;
        }
        assert!(
            prev_e < 0.999 * e0,
            "RK4 energy didn't decay: {e0} -> {prev_e}"
        );
    }
}
