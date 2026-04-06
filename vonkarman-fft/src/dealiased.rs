use crate::backend::FftBackend;
use ndarray::Array3;
use num_complex::Complex;
use vonkarman_core::field::GridSpec;

/// Compute the dealiased cross product u x omega in spectral space
/// using 3/2 zero-padding.
///
/// 1. Zero-pad u_hat and omega_hat from N^3 to (3N/2)^3 in spectral space.
/// 2. Inverse FFT to the padded physical grid.
/// 3. Pointwise cross product in physical space.
/// 4. Forward FFT back to spectral space.
/// 5. Truncate to N^3.
///
/// `backend` operates on the N^3 grid, `padded_backend` on the (3N/2)^3 grid.
pub fn dealiased_cross_product(
    _backend: &dyn FftBackend<f64>,
    padded_backend: &dyn FftBackend<f64>,
    grid: &GridSpec,
    u_hat: &[Array3<Complex<f64>>; 3],
    omega_hat: &[Array3<Complex<f64>>; 3],
    result_hat: &mut [Array3<Complex<f64>>; 3],
) {
    let pg = grid.padded_3half();
    let (pnx, pny, pnz_half) = pg.spectral_shape();

    // Step 1+2: zero-pad each component and inverse FFT to padded physical space
    let mut u_phys = [
        Array3::<f64>::zeros((pg.nx, pg.ny, pg.nz)),
        Array3::<f64>::zeros((pg.nx, pg.ny, pg.nz)),
        Array3::<f64>::zeros((pg.nx, pg.ny, pg.nz)),
    ];
    let mut omega_phys = [
        Array3::<f64>::zeros((pg.nx, pg.ny, pg.nz)),
        Array3::<f64>::zeros((pg.nx, pg.ny, pg.nz)),
        Array3::<f64>::zeros((pg.nx, pg.ny, pg.nz)),
    ];

    // The padded inverse FFT normalizes by 1/(3N/2)^3 (per axis).
    // We need the physical-space values to be correct, so we scale
    // the padded spectral coefficients by (3N/2)^3 / N^3 = (3/2)^3
    // to compensate for the larger grid.
    let scale = (pg.nx * pg.ny * pg.nz) as f64 / (grid.nx * grid.ny * grid.nz) as f64;

    for c in 0..3 {
        let mut padded_hat = Array3::<Complex<f64>>::zeros((pnx, pny, pnz_half));
        zero_pad_spectral(&u_hat[c], &mut padded_hat, grid.nx, grid.ny);
        padded_hat.mapv_inplace(|v| Complex {
            re: v.re * scale,
            im: v.im * scale,
        });
        padded_backend.c2r_3d(&padded_hat, &mut u_phys[c]);
    }
    for c in 0..3 {
        let mut padded_hat = Array3::<Complex<f64>>::zeros((pnx, pny, pnz_half));
        zero_pad_spectral(&omega_hat[c], &mut padded_hat, grid.nx, grid.ny);
        padded_hat.mapv_inplace(|v| Complex {
            re: v.re * scale,
            im: v.im * scale,
        });
        padded_backend.c2r_3d(&padded_hat, &mut omega_phys[c]);
    }

    // Step 3: pointwise cross product in physical space
    let mut cross_phys = [
        Array3::<f64>::zeros((pg.nx, pg.ny, pg.nz)),
        Array3::<f64>::zeros((pg.nx, pg.ny, pg.nz)),
        Array3::<f64>::zeros((pg.nx, pg.ny, pg.nz)),
    ];
    for i in 0..pg.nx {
        for j in 0..pg.ny {
            for k in 0..pg.nz {
                let ux = u_phys[0][[i, j, k]];
                let uy = u_phys[1][[i, j, k]];
                let uz = u_phys[2][[i, j, k]];
                let ox = omega_phys[0][[i, j, k]];
                let oy = omega_phys[1][[i, j, k]];
                let oz = omega_phys[2][[i, j, k]];
                cross_phys[0][[i, j, k]] = uy * oz - uz * oy;
                cross_phys[1][[i, j, k]] = uz * ox - ux * oz;
                cross_phys[2][[i, j, k]] = ux * oy - uy * ox;
            }
        }
    }

    // Step 4: forward FFT of cross product
    let mut cross_hat_padded = [
        Array3::<Complex<f64>>::zeros((pnx, pny, pnz_half)),
        Array3::<Complex<f64>>::zeros((pnx, pny, pnz_half)),
        Array3::<Complex<f64>>::zeros((pnx, pny, pnz_half)),
    ];
    for c in 0..3 {
        padded_backend.r2c_3d(&cross_phys[c], &mut cross_hat_padded[c]);
    }

    // Step 5: truncate back to original spectral size and normalize.
    // The forward FFT on the padded grid produces unnormalized coefficients
    // that are (3/2)^3 times larger than the original grid's convention.
    // Divide by (3N/2)^3 to normalize, then multiply by N^3 to get the
    // original grid's unnormalized convention.
    // Net factor: N^3 / (3N/2)^3 = 1/scale
    let inv_scale = 1.0 / scale;
    for c in 0..3 {
        truncate_spectral(&cross_hat_padded[c], &mut result_hat[c], grid.nx, grid.ny);
        result_hat[c].mapv_inplace(|v| Complex {
            re: v.re * inv_scale,
            im: v.im * inv_scale,
        });
    }
}

/// Zero-pad spectral coefficients from (nx, ny, nz/2+1) into a larger array.
/// Positive wavenumbers go to the same indices; negative wavenumbers go to the
/// end of the padded array.
fn zero_pad_spectral(
    src: &Array3<Complex<f64>>,
    dst: &mut Array3<Complex<f64>>,
    nx: usize,
    ny: usize,
) {
    let (snx, sny, snz) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let (dnx, dny, _dnz) = (dst.shape()[0], dst.shape()[1], dst.shape()[2]);

    // Zero out destination
    dst.fill(Complex { re: 0.0, im: 0.0 });

    // Copy wavenumbers, mapping negative frequencies to the end of the padded array
    for ix in 0..snx {
        let dix = if ix < nx / 2 { ix } else { dnx - (snx - ix) };
        for iy in 0..sny {
            let diy = if iy < ny / 2 { iy } else { dny - (sny - iy) };
            for iz in 0..snz {
                // z-direction is R2C (only positive freqs), copy directly
                dst[[dix, diy, iz]] = src[[ix, iy, iz]];
            }
        }
    }
}

/// Truncate spectral coefficients from a padded array back to the original size.
fn truncate_spectral(
    src: &Array3<Complex<f64>>,
    dst: &mut Array3<Complex<f64>>,
    nx: usize,
    ny: usize,
) {
    let (snx, _sny, _snz) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let (dnx, dny, dnz) = (dst.shape()[0], dst.shape()[1], dst.shape()[2]);

    for ix in 0..dnx {
        let six = if ix < nx / 2 { ix } else { snx - (dnx - ix) };
        for iy in 0..dny {
            let siy = if iy < ny / 2 {
                iy
            } else {
                src.shape()[1] - (dny - iy)
            };
            for iz in 0..dnz {
                dst[[ix, iy, iz]] = src[[six, siy, iz]];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndrustfft_backend::NdrustfftBackend;
    use ndarray::Array3;
    use num_complex::Complex;
    use vonkarman_core::field::GridSpec;

    #[test]
    fn dealiased_cross_product_zero_input() {
        let n = 8;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let backend = NdrustfftBackend::new(n, n, n);
        let padded_backend = NdrustfftBackend::new(3 * n / 2, 3 * n / 2, 3 * n / 2);
        let (snx, sny, snz) = grid.spectral_shape();
        let shape = (snx, sny, snz);
        let zero = Complex { re: 0.0, im: 0.0 };
        let u_hat = [
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
        ];
        let omega_hat = u_hat.clone();
        let mut result = u_hat.clone();
        dealiased_cross_product(
            &backend,
            &padded_backend,
            &grid,
            &u_hat,
            &omega_hat,
            &mut result,
        );
        for c in 0..3 {
            for val in result[c].iter() {
                assert!(val.re.abs() < 1e-14 && val.im.abs() < 1e-14);
            }
        }
    }

    #[test]
    fn cross_product_anti_symmetry() {
        // u x omega = -(omega x u)
        let n = 8;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let backend = NdrustfftBackend::new(n, n, n);
        let padded_backend = NdrustfftBackend::new(3 * n / 2, 3 * n / 2, 3 * n / 2);
        let (snx, sny, snz) = grid.spectral_shape();
        let shape = (snx, sny, snz);

        let mut u_hat: [Array3<Complex<f64>>; 3] = [
            Array3::zeros(shape),
            Array3::zeros(shape),
            Array3::zeros(shape),
        ];
        let mut omega_hat: [Array3<Complex<f64>>; 3] = [
            Array3::zeros(shape),
            Array3::zeros(shape),
            Array3::zeros(shape),
        ];
        // Put energy in a single mode
        u_hat[0][[1, 0, 0]] = Complex { re: 1.0, im: 0.5 };
        omega_hat[1][[0, 1, 0]] = Complex { re: 0.7, im: -0.3 };

        let mut r1 = [
            Array3::zeros(shape),
            Array3::zeros(shape),
            Array3::zeros(shape),
        ];
        let mut r2 = [
            Array3::zeros(shape),
            Array3::zeros(shape),
            Array3::zeros(shape),
        ];
        dealiased_cross_product(
            &backend,
            &padded_backend,
            &grid,
            &u_hat,
            &omega_hat,
            &mut r1,
        );
        dealiased_cross_product(
            &backend,
            &padded_backend,
            &grid,
            &omega_hat,
            &u_hat,
            &mut r2,
        );
        // r1 should equal -r2
        for c in 0..3 {
            for (a, b) in r1[c].iter().zip(r2[c].iter()) {
                assert!(
                    (a.re + b.re).abs() < 1e-10 && (a.im + b.im).abs() < 1e-10,
                    "anti-symmetry failed: {a:?} vs {b:?}"
                );
            }
        }
    }
}
