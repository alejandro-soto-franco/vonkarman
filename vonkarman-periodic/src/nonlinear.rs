use ndarray::Array3;
use num_complex::Complex;
use vonkarman_core::field::GridSpec;
use vonkarman_core::spectral_ops::SpectralOps;
use vonkarman_fft::{FftBackend, dealiased_cross_product};

/// Compute the nonlinear term N_hat for the NS equations in rotation form.
///
/// N_hat = P * FFT(u x omega), where:
/// - omega = curl(u) in spectral space
/// - u x omega computed via dealiased (3/2 padded) cross product
/// - P is the Leray projector (removes pressure gradient)
///
/// The result is divergence-free and ready for ETD-RK4 / RK4 time integration.
pub fn compute_nonlinear(
    ops: &SpectralOps<f64>,
    backend: &dyn FftBackend<f64>,
    padded_backend: &dyn FftBackend<f64>,
    grid: &GridSpec,
    u_hat: &[Array3<Complex<f64>>; 3],
    n_hat: &mut [Array3<Complex<f64>>; 3],
) {
    let (snx, sny, snz) = grid.spectral_shape();
    let shape = (snx, sny, snz);

    // Step 1: spectral curl to get omega_hat
    let mut omega_hat: [Array3<Complex<f64>>; 3] = [
        Array3::zeros(shape),
        Array3::zeros(shape),
        Array3::zeros(shape),
    ];
    ops.curl(u_hat, &mut omega_hat);

    // Step 2: dealiased cross product u x omega
    dealiased_cross_product(backend, padded_backend, grid, u_hat, &omega_hat, n_hat);

    // Step 3: Leray projection to enforce divergence-free
    ops.leray_project(n_hat);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use num_complex::Complex;
    use vonkarman_core::field::GridSpec;
    use vonkarman_core::spectral_ops::SpectralOps;
    use vonkarman_fft::NdrustfftBackend;

    #[test]
    fn nonlinear_term_divergence_free() {
        let n = 16;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let ops = SpectralOps::<f64>::new(&grid);
        let backend = NdrustfftBackend::new(n, n, n);
        let pg = grid.padded_3half();
        let padded_backend = NdrustfftBackend::new(pg.nx, pg.ny, pg.nz);

        // Start from Taylor-Green IC, transform to spectral
        let v = crate::ic::taylor_green::<f64>(&grid);
        let (snx, sny, snz) = grid.spectral_shape();
        let shape = (snx, sny, snz);
        let mut u_hat: [Array3<Complex<f64>>; 3] = [
            Array3::zeros(shape), Array3::zeros(shape), Array3::zeros(shape),
        ];
        for c in 0..3 {
            backend.r2c_3d(&v.data[c], &mut u_hat[c]);
        }

        let mut n_hat: [Array3<Complex<f64>>; 3] = [
            Array3::zeros(shape), Array3::zeros(shape), Array3::zeros(shape),
        ];
        compute_nonlinear(&ops, &backend, &padded_backend, &grid, &u_hat, &mut n_hat);

        // Check divergence: k . N_hat = 0
        for ix in 0..snx {
            let kx = ops.kx[ix];
            for iy in 0..sny {
                let ky = ops.ky[iy];
                for iz in 0..snz {
                    let kz = ops.kz[iz];
                    let div_re = kx * n_hat[0][[ix, iy, iz]].re
                               + ky * n_hat[1][[ix, iy, iz]].re
                               + kz * n_hat[2][[ix, iy, iz]].re;
                    let div_im = kx * n_hat[0][[ix, iy, iz]].im
                               + ky * n_hat[1][[ix, iy, iz]].im
                               + kz * n_hat[2][[ix, iy, iz]].im;
                    let mag = (div_re * div_re + div_im * div_im).sqrt();
                    assert!(mag < 1e-6, "divergence at ({ix},{iy},{iz}) = {mag}");
                }
            }
        }
    }
}
