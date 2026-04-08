use crate::field::GridSpec;
use crate::float::Float;
use ndarray::{Array1, Array3};
use num_complex::Complex;

/// Spectral derivative operations in Fourier space.
///
/// All derivatives are wavenumber multiplications: d/dx -> i*kx, etc.
/// Operates on R2C spectral arrays of shape (nx, ny, nz/2+1).
pub struct SpectralOps<F: Float> {
    /// Wavenumbers in x-direction, length nx.
    /// Layout: [0, 1, ..., nx/2, -(nx/2-1), ..., -1] scaled by 2*pi/lx.
    pub kx: Array1<F>,
    /// Wavenumbers in y-direction, length ny.
    pub ky: Array1<F>,
    /// Wavenumbers in z-direction, length nz/2+1 (R2C).
    pub kz: Array1<F>,
    /// |k|^2 at each spectral point, shape (nx, ny, nz/2+1).
    pub k_mag_sq: Array3<F>,
    /// Grid metadata.
    pub grid: GridSpec,
}

impl<F: Float> SpectralOps<F> {
    /// Build wavenumber arrays and |k|^2 for a given grid.
    pub fn new(grid: &GridSpec) -> Self {
        let kx = Self::rfftfreq_full(grid.nx, grid.lx);
        let ky = Self::rfftfreq_full(grid.ny, grid.ly);
        let kz = Self::rfftfreq_half(grid.nz, grid.lz);

        let (snx, sny, snz) = grid.spectral_shape();
        let mut k_mag_sq = Array3::from_elem((snx, sny, snz), F::ZERO);
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    k_mag_sq[[ix, iy, iz]] = kx[ix] * kx[ix] + ky[iy] * ky[iy] + kz[iz] * kz[iz];
                }
            }
        }

        Self {
            kx,
            ky,
            kz,
            k_mag_sq,
            grid: *grid,
        }
    }

    /// Wavenumbers for a full (non-R2C) dimension: [0, 1, ..., n/2, -(n/2-1), ..., -1].
    fn rfftfreq_full(n: usize, l: f64) -> Array1<F> {
        let scale = F::TWO * F::PI / F::from_f64(l);
        let mut k = Array1::from_elem(n, F::ZERO);
        // Positive wavenumbers: 0, 1, ..., n/2-1
        for i in 0..n / 2 {
            k[i] = F::from_f64(i as f64) * scale;
        }
        // Negative wavenumbers: -n/2, -(n/2-1), ..., -1
        for i in (n / 2)..n {
            k[i] = F::from_f64(i as f64 - n as f64) * scale;
        }
        k
    }

    /// Wavenumbers for the R2C (half) dimension: [0, 1, ..., n/2].
    fn rfftfreq_half(n: usize, l: f64) -> Array1<F> {
        let scale = F::TWO * F::PI / F::from_f64(l);
        let len = n / 2 + 1;
        let mut k = Array1::from_elem(len, F::ZERO);
        for i in 0..len {
            k[i] = F::from_f64(i as f64) * scale;
        }
        k
    }

    /// Spectral curl: omega_hat = i*k x u_hat.
    ///
    /// omega_x = i*(ky*uz - kz*uy)
    /// omega_y = i*(kz*ux - kx*uz)
    /// omega_z = i*(kx*uy - ky*ux)
    pub fn curl(&self, u_hat: &[Array3<Complex<F>>; 3], omega_hat: &mut [Array3<Complex<F>>; 3]) {
        let (snx, sny, snz) = self.grid.spectral_shape();
        for ix in 0..snx {
            let kx = self.kx[ix];
            for iy in 0..sny {
                let ky = self.ky[iy];
                for iz in 0..snz {
                    let kz = self.kz[iz];
                    let ux = u_hat[0][[ix, iy, iz]];
                    let uy = u_hat[1][[ix, iy, iz]];
                    let uz = u_hat[2][[ix, iy, iz]];
                    // i*(ky*uz - kz*uy): re = -(ky*uz.im - kz*uy.im), im = ky*uz.re - kz*uy.re
                    omega_hat[0][[ix, iy, iz]] = Complex {
                        re: -(ky * uz.im - kz * uy.im),
                        im: ky * uz.re - kz * uy.re,
                    };
                    omega_hat[1][[ix, iy, iz]] = Complex {
                        re: -(kz * ux.im - kx * uz.im),
                        im: kz * ux.re - kx * uz.re,
                    };
                    omega_hat[2][[ix, iy, iz]] = Complex {
                        re: -(kx * uy.im - ky * ux.im),
                        im: kx * uy.re - ky * ux.re,
                    };
                }
            }
        }
    }

    /// Leray projection: P = I - k*k^T / |k|^2.
    ///
    /// Projects a spectral vector field onto the divergence-free subspace.
    /// The k=0 mode is left unchanged (mean flow).
    pub fn leray_project(&self, u_hat: &mut [Array3<Complex<F>>; 3]) {
        let (snx, sny, snz) = self.grid.spectral_shape();
        for ix in 0..snx {
            let kx = self.kx[ix];
            for iy in 0..sny {
                let ky = self.ky[iy];
                for iz in 0..snz {
                    let kz = self.kz[iz];
                    let k2 = self.k_mag_sq[[ix, iy, iz]];
                    if k2.to_f64() < 1e-30 {
                        continue; // skip k=0
                    }
                    let inv_k2 = F::ONE / k2;
                    let ux = u_hat[0][[ix, iy, iz]];
                    let uy = u_hat[1][[ix, iy, iz]];
                    let uz = u_hat[2][[ix, iy, iz]];
                    // k . u_hat (complex dot product with real k)
                    let k_dot_u = Complex {
                        re: kx * ux.re + ky * uy.re + kz * uz.re,
                        im: kx * ux.im + ky * uy.im + kz * uz.im,
                    };
                    // u_hat -= k * (k . u_hat) / |k|^2
                    u_hat[0][[ix, iy, iz]] = Complex {
                        re: ux.re - kx * k_dot_u.re * inv_k2,
                        im: ux.im - kx * k_dot_u.im * inv_k2,
                    };
                    u_hat[1][[ix, iy, iz]] = Complex {
                        re: uy.re - ky * k_dot_u.re * inv_k2,
                        im: uy.im - ky * k_dot_u.im * inv_k2,
                    };
                    u_hat[2][[ix, iy, iz]] = Complex {
                        re: uz.re - kz * k_dot_u.re * inv_k2,
                        im: uz.im - kz * k_dot_u.im * inv_k2,
                    };
                }
            }
        }
    }

    /// Apply -nu * |k|^2 to each component (viscous term in spectral space).
    pub fn apply_viscous(
        &self,
        u_hat: &[Array3<Complex<F>>; 3],
        nu: F,
        out: &mut [Array3<Complex<F>>; 3],
    ) {
        let (snx, sny, snz) = self.grid.spectral_shape();
        for c in 0..3 {
            for ix in 0..snx {
                for iy in 0..sny {
                    for iz in 0..snz {
                        let factor = -(nu * self.k_mag_sq[[ix, iy, iz]]);
                        let u = u_hat[c][[ix, iy, iz]];
                        out[c][[ix, iy, iz]] = Complex {
                            re: factor * u.re,
                            im: factor * u.im,
                        };
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn wavenumbers_n8() {
        let grid = GridSpec::cubic(8, 2.0 * std::f64::consts::PI);
        let ops = SpectralOps::<f64>::new(&grid);
        assert_eq!(ops.kx.len(), 8);
        assert!((ops.kx[0] - 0.0).abs() < 1e-14);
        assert!((ops.kx[1] - 1.0).abs() < 1e-14);
        assert!((ops.kx[4] - (-4.0)).abs() < 1e-14);
        assert!((ops.kx[7] - (-1.0)).abs() < 1e-14);
        assert_eq!(ops.kz.len(), 5); // N/2+1
        assert!((ops.kz[0] - 0.0).abs() < 1e-14);
        assert!((ops.kz[4] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn k_mag_sq_shape() {
        let grid = GridSpec::cubic(8, 2.0 * std::f64::consts::PI);
        let ops = SpectralOps::<f64>::new(&grid);
        assert_eq!(ops.k_mag_sq.shape(), &[8, 8, 5]);
    }

    #[test]
    fn leray_divergence_free() {
        let grid = GridSpec::cubic(8, 2.0 * std::f64::consts::PI);
        let ops = SpectralOps::<f64>::new(&grid);
        let (snx, sny, snz) = grid.spectral_shape();
        let shape = (snx, sny, snz);
        let mut u_hat: [ndarray::Array3<Complex<f64>>; 3] = [
            ndarray::Array3::from_elem(shape, Complex { re: 1.0, im: 1.0 }),
            ndarray::Array3::from_elem(shape, Complex { re: 2.0, im: 1.0 }),
            ndarray::Array3::from_elem(shape, Complex { re: 3.0, im: 1.0 }),
        ];
        ops.leray_project(&mut u_hat);
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    let kx = ops.kx[ix];
                    let ky = ops.ky[iy];
                    let kz = ops.kz[iz];
                    let div = Complex {
                        re: kx * u_hat[0][[ix, iy, iz]].re
                            + ky * u_hat[1][[ix, iy, iz]].re
                            + kz * u_hat[2][[ix, iy, iz]].re,
                        im: kx * u_hat[0][[ix, iy, iz]].im
                            + ky * u_hat[1][[ix, iy, iz]].im
                            + kz * u_hat[2][[ix, iy, iz]].im,
                    };
                    let mag = (div.re * div.re + div.im * div.im).sqrt();
                    assert!(mag < 1e-12, "divergence at ({ix},{iy},{iz}) = {mag}");
                }
            }
        }
    }

    #[test]
    fn curl_of_constant_is_zero() {
        let grid = GridSpec::cubic(8, 2.0 * std::f64::consts::PI);
        let ops = SpectralOps::<f64>::new(&grid);
        let (snx, sny, snz) = grid.spectral_shape();
        let shape = (snx, sny, snz);
        let mut u_hat: [ndarray::Array3<Complex<f64>>; 3] = [
            ndarray::Array3::from_elem(shape, Complex { re: 0.0, im: 0.0 }),
            ndarray::Array3::from_elem(shape, Complex { re: 0.0, im: 0.0 }),
            ndarray::Array3::from_elem(shape, Complex { re: 0.0, im: 0.0 }),
        ];
        u_hat[0][[0, 0, 0]] = Complex { re: 5.0, im: 0.0 };
        let mut omega_hat = [
            ndarray::Array3::from_elem(shape, Complex { re: 0.0, im: 0.0 }),
            ndarray::Array3::from_elem(shape, Complex { re: 0.0, im: 0.0 }),
            ndarray::Array3::from_elem(shape, Complex { re: 0.0, im: 0.0 }),
        ];
        ops.curl(&u_hat, &mut omega_hat);
        for component in &omega_hat {
            for val in component.iter() {
                assert!(val.re.abs() < 1e-14 && val.im.abs() < 1e-14);
            }
        }
    }
}
