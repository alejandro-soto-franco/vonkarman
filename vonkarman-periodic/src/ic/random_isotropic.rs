use ndarray::Array3;
use num_complex::Complex;
use vonkarman_core::field::{GridSpec, VectorField};
use vonkarman_core::spectral_ops::SpectralOps;

/// Random isotropic turbulence initial condition.
///
/// Generates a divergence-free velocity field with a prescribed energy
/// spectrum E(k) ~ k^p * exp(-k^2 / (2*k_peak^2)).
///
/// Parameters:
/// - `k_peak`: peak wavenumber of the initial spectrum
/// - `total_energy`: target total kinetic energy (1/2 * <|u|^2>)
/// - `seed`: RNG seed for reproducibility
/// - `fft_backend`: needed to transform back to physical space
///
/// The field is made divergence-free via Leray projection.
pub fn random_isotropic(
    grid: &GridSpec,
    k_peak: f64,
    total_energy: f64,
    seed: u64,
    fft_backend: &dyn vonkarman_fft::FftBackend<f64>,
) -> VectorField<f64> {
    let ops = SpectralOps::<f64>::new(grid);
    let (snx, sny, snz) = grid.spectral_shape();
    let shape = (snx, sny, snz);
    let zero = Complex { re: 0.0, im: 0.0 };

    // Simple xorshift64 RNG for reproducibility without external deps
    let mut rng_state = seed;
    let mut next_f64 = || -> f64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        // Map to [0, 1)
        (rng_state as f64) / (u64::MAX as f64)
    };

    let mut u_hat: [Array3<Complex<f64>>; 3] = [
        Array3::from_elem(shape, zero),
        Array3::from_elem(shape, zero),
        Array3::from_elem(shape, zero),
    ];

    // Assign random phases with prescribed amplitude
    for ix in 0..snx {
        for iy in 0..sny {
            for iz in 0..snz {
                let k_mag = ops.k_mag_sq[[ix, iy, iz]].sqrt();
                if k_mag < 1e-10 {
                    continue; // skip DC
                }

                // Energy spectrum: E(k) ~ k^4 * exp(-k^2 / (2*k_peak^2))
                // (von Karman spectrum with p=4 for smooth initial data)
                let amplitude = k_mag.powi(4) * (-k_mag * k_mag / (2.0 * k_peak * k_peak)).exp();
                let amplitude = amplitude.sqrt(); // |u_hat| ~ sqrt(E(k))

                for c in 0..3 {
                    let phase = 2.0 * std::f64::consts::PI * next_f64();
                    u_hat[c][[ix, iy, iz]] = Complex {
                        re: amplitude * phase.cos(),
                        im: amplitude * phase.sin(),
                    };
                }
            }
        }
    }

    // Leray projection to enforce divergence-free
    ops.leray_project(&mut u_hat);

    // Compute current energy and rescale to target
    let ntot = (grid.nx * grid.ny * grid.nz) as f64;
    let mut current_energy = 0.0_f64;
    for c in 0..3 {
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    let mag2 = u_hat[c][[ix, iy, iz]].re.powi(2)
                             + u_hat[c][[ix, iy, iz]].im.powi(2);
                    let weight = if iz == 0 || iz == grid.nz / 2 { 1.0 } else { 2.0 };
                    current_energy += weight * mag2;
                }
            }
        }
    }
    current_energy *= 0.5 / (ntot * ntot);

    if current_energy > 1e-30 {
        let scale = (total_energy / current_energy).sqrt();
        for c in 0..3 {
            u_hat[c].mapv_inplace(|v| Complex {
                re: v.re * scale,
                im: v.im * scale,
            });
        }
    }

    // Transform to physical space
    let mut v = VectorField::zeros(*grid);
    for c in 0..3 {
        fft_backend.c2r_3d(&u_hat[c], &mut v.data[c]);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use vonkarman_core::field::GridSpec;
    use vonkarman_fft::NdrustfftBackend;

    #[test]
    fn random_isotropic_energy() {
        let n = 16;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let backend = NdrustfftBackend::new(n, n, n);
        let target_energy = 0.5;

        let v = random_isotropic(&grid, 4.0, target_energy, 42, &backend);

        let dv = grid.dv();
        let vol = grid.lx * grid.ly * grid.lz;
        let mut energy = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let u2 = v.x()[[i, j, k]].powi(2)
                           + v.y()[[i, j, k]].powi(2)
                           + v.z()[[i, j, k]].powi(2);
                    energy += u2 * dv;
                }
            }
        }
        energy *= 0.5 / vol;

        let rel_err = (energy - target_energy).abs() / target_energy;
        assert!(
            rel_err < 0.1,
            "energy = {energy}, target = {target_energy}, rel_err = {rel_err}"
        );
    }

    #[test]
    fn random_isotropic_reproducible() {
        let n = 16;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let backend = NdrustfftBackend::new(n, n, n);

        let v1 = random_isotropic(&grid, 4.0, 0.5, 42, &backend);
        let v2 = random_isotropic(&grid, 4.0, 0.5, 42, &backend);

        for c in 0..3 {
            for (a, b) in v1.data[c].iter().zip(v2.data[c].iter()) {
                assert!((a - b).abs() < 1e-14, "not reproducible: {a} vs {b}");
            }
        }
    }
}
