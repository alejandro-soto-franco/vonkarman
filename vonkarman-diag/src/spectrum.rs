use ndarray::Array3;
use num_complex::Complex;
use vonkarman_core::field::GridSpec;
use vonkarman_core::spectral_ops::SpectralOps;

/// Shell-averaged energy spectrum E(k).
///
/// For each integer shell k_n = n (n = 0, 1, ..., k_max), sums
/// (1/2) * |u_hat(k)|^2 / N^6 over all wavevectors with |k| in [n-0.5, n+0.5),
/// using R2C weighting (factor 2 for interior z-modes).
///
/// Returns (shells, spectrum) where shells[i] = i and spectrum[i] = E(i).
pub fn energy_spectrum(
    u_hat: &[Array3<Complex<f64>>; 3],
    ops: &SpectralOps<f64>,
    grid: &GridSpec,
) -> (Vec<f64>, Vec<f64>) {
    let (snx, sny, snz) = grid.spectral_shape();
    let ntot = (grid.nx * grid.ny * grid.nz) as f64;

    // Maximum wavenumber magnitude
    let k_max_x = (grid.nx / 2) as f64;
    let k_max_y = (grid.ny / 2) as f64;
    let k_max_z = (grid.nz / 2) as f64;
    let k_max = (k_max_x * k_max_x + k_max_y * k_max_y + k_max_z * k_max_z).sqrt();
    let n_shells = k_max.ceil() as usize + 1;

    let mut spectrum = vec![0.0_f64; n_shells];

    for ix in 0..snx {
        let kx = ops.kx[ix];
        for iy in 0..sny {
            let ky = ops.ky[iy];
            for iz in 0..snz {
                let kz = ops.kz[iz];
                let k_mag = (kx * kx + ky * ky + kz * kz).sqrt();
                let shell = k_mag.round() as usize;
                if shell >= n_shells {
                    continue;
                }

                let mut mag2 = 0.0_f64;
                for c in 0..3 {
                    mag2 += u_hat[c][[ix, iy, iz]].re.powi(2)
                          + u_hat[c][[ix, iy, iz]].im.powi(2);
                }

                // R2C weighting: interior z-modes counted twice
                let weight = if iz == 0 || iz == grid.nz / 2 { 1.0 } else { 2.0 };
                spectrum[shell] += 0.5 * weight * mag2 / (ntot * ntot);
            }
        }
    }

    let shells: Vec<f64> = (0..n_shells).map(|i| i as f64).collect();
    (shells, spectrum)
}

/// Compensated energy spectrum: k^{5/3} * E(k).
///
/// Used to identify the inertial range. In Kolmogorov turbulence,
/// k^{5/3} * E(k) ~ C_K * epsilon^{2/3} (constant plateau).
pub fn compensated_spectrum(shells: &[f64], spectrum: &[f64]) -> Vec<f64> {
    shells.iter().zip(spectrum.iter()).map(|(&k, &e)| {
        if k > 0.0 {
            k.powf(5.0 / 3.0) * e
        } else {
            0.0
        }
    }).collect()
}

/// Dissipation spectrum: 2 * nu * k^2 * E(k).
///
/// Integrating over k gives the total dissipation rate epsilon = 2*nu*Omega.
pub fn dissipation_spectrum(shells: &[f64], spectrum: &[f64], nu: f64) -> Vec<f64> {
    shells.iter().zip(spectrum.iter()).map(|(&k, &e)| {
        2.0 * nu * k * k * e
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vonkarman_core::field::GridSpec;
    use vonkarman_core::spectral_ops::SpectralOps;
    use vonkarman_fft::{FftBackend, NdrustfftBackend};

    #[test]
    fn spectrum_single_mode() {
        // Put energy in a single wavenumber mode k=(1,0,0)
        // and verify the spectrum peaks at shell 1.
        let n = 8;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let ops = SpectralOps::<f64>::new(&grid);
        let (snx, sny, snz) = grid.spectral_shape();
        let shape = (snx, sny, snz);
        let zero = Complex { re: 0.0, im: 0.0 };

        let mut u_hat: [Array3<Complex<f64>>; 3] = [
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
        ];
        // Single mode: u_x = sin(x) -> u_hat_x at k=(1,0,0) and k=(-1,0,0)
        // With N=8, L=2*pi: kx[1] = 1.0
        u_hat[0][[1, 0, 0]] = Complex { re: 0.0, im: -0.5 * (n * n * n) as f64 };
        u_hat[0][[n - 1, 0, 0]] = Complex { re: 0.0, im: 0.5 * (n * n * n) as f64 };

        let (shells, spec) = energy_spectrum(&u_hat, &ops, &grid);

        // Shell 1 should have the energy
        assert!(spec[1] > 0.0, "shell 1 should have energy, got {}", spec[1]);
        // Shell 0 and shells > 1 should be negligible
        assert!(spec[0].abs() < 1e-10, "shell 0 should be ~0, got {}", spec[0]);
        for i in 2..shells.len() {
            assert!(spec[i].abs() < 1e-10, "shell {i} should be ~0, got {}", spec[i]);
        }
    }

    #[test]
    fn spectrum_total_equals_energy() {
        // Total energy from spectrum should match Parseval sum.
        let n = 16;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let ops = SpectralOps::<f64>::new(&grid);
        let backend = NdrustfftBackend::new(n, n, n);

        // Use Taylor-Green IC
        let (snx, sny, snz) = grid.spectral_shape();
        let shape = (snx, sny, snz);
        let zero = Complex { re: 0.0, im: 0.0 };

        // Build TG in physical space and transform
        let mut phys_x = ndarray::Array3::<f64>::zeros((n, n, n));
        let mut phys_y = ndarray::Array3::<f64>::zeros((n, n, n));
        for i in 0..n {
            let x = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            for j in 0..n {
                let y = 2.0 * std::f64::consts::PI * (j as f64) / (n as f64);
                for k in 0..n {
                    let z = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
                    phys_x[[i, j, k]] = x.sin() * y.cos() * z.cos();
                    phys_y[[i, j, k]] = -(x.cos() * y.sin() * z.cos());
                }
            }
        }

        let mut u_hat: [Array3<Complex<f64>>; 3] = [
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
        ];
        backend.r2c_3d(&phys_x, &mut u_hat[0]);
        backend.r2c_3d(&phys_y, &mut u_hat[1]);
        // u_hat[2] stays zero (w=0 for TG)

        let (_shells, spec) = energy_spectrum(&u_hat, &ops, &grid);
        let spec_total: f64 = spec.iter().sum();

        // Direct Parseval energy
        let ntot = (n * n * n) as f64;
        let mut direct = 0.0_f64;
        for c in 0..3 {
            for ix in 0..snx {
                for iy in 0..sny {
                    for iz in 0..snz {
                        let mag2 = u_hat[c][[ix, iy, iz]].re.powi(2)
                                 + u_hat[c][[ix, iy, iz]].im.powi(2);
                        let weight = if iz == 0 || iz == n / 2 { 1.0 } else { 2.0 };
                        direct += weight * mag2;
                    }
                }
            }
        }
        direct *= 0.5 / (ntot * ntot);

        let rel_err = (spec_total - direct).abs() / direct.max(1e-30);
        assert!(
            rel_err < 1e-10,
            "spectrum total {spec_total} != Parseval energy {direct}, rel_err={rel_err}"
        );
    }
}
