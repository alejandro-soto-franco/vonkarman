use crate::backend::FftBackend;
use ndarray::Array3;
use ndrustfft::{FftHandler, R2cFftHandler, ndfft, ndfft_r2c, ndifft, ndifft_r2c};
use num_complex::Complex;

/// Pure-Rust CPU FFT backend using ndrustfft.
///
/// This is the automatic fallback when no GPU or FFTW is available.
/// Performance is adequate for development and small grids (up to ~128^3).
pub struct NdrustfftBackend {
    nx: usize,
    ny: usize,
    nz: usize,
    handler_x: FftHandler<f64>,
    handler_y: FftHandler<f64>,
    handler_z: R2cFftHandler<f64>,
}

impl NdrustfftBackend {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            nx,
            ny,
            nz,
            handler_x: FftHandler::new(nx),
            handler_y: FftHandler::new(ny),
            handler_z: R2cFftHandler::new(nz),
        }
    }
}

impl FftBackend<f64> for NdrustfftBackend {
    fn r2c_3d(&self, input: &Array3<f64>, output: &mut Array3<Complex<f64>>) {
        // Step 1: R2C along z-axis (axis 2)
        let mut temp_rz = Array3::<Complex<f64>>::zeros((self.nx, self.ny, self.nz / 2 + 1));
        ndfft_r2c(input, &mut temp_rz, &self.handler_z, 2);

        // Step 2: C2C along y-axis (axis 1)
        let mut temp_ry = temp_rz.clone();
        ndfft(&temp_rz, &mut temp_ry, &self.handler_y, 1);

        // Step 3: C2C along x-axis (axis 0)
        ndfft(&temp_ry, output, &self.handler_x, 0);
    }

    fn c2r_3d(&self, input: &Array3<Complex<f64>>, output: &mut Array3<f64>) {
        // Step 1: iC2C along x-axis (axis 0)
        let mut temp_ix = Array3::<Complex<f64>>::zeros(input.raw_dim());
        ndifft(input, &mut temp_ix, &self.handler_x, 0);

        // Step 2: iC2C along y-axis (axis 1)
        let mut temp_iy = temp_ix.clone();
        ndifft(&temp_ix, &mut temp_iy, &self.handler_y, 1);

        // Step 3: iR2C along z-axis (axis 2, complex-to-real)
        ndifft_r2c(&temp_iy, output, &self.handler_z, 2);
    }

    fn name(&self) -> &str {
        "ndrustfft (pure Rust CPU)"
    }

    fn precision_digits(&self) -> usize {
        15
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::FftBackend;
    use ndarray::Array3;
    use num_complex::Complex;

    #[test]
    fn roundtrip_identity() {
        let n = 8;
        let backend = NdrustfftBackend::new(n, n, n);
        let mut input = Array3::<f64>::zeros((n, n, n));
        for i in 0..n {
            let x = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            for j in 0..n {
                for k in 0..n {
                    input[[i, j, k]] = x.sin();
                }
            }
        }
        let mut spectral = Array3::<Complex<f64>>::zeros((n, n, n / 2 + 1));
        backend.r2c_3d(&input, &mut spectral);

        let mut output = Array3::<f64>::zeros((n, n, n));
        backend.c2r_3d(&spectral, &mut output);

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    assert!(
                        (output[[i, j, k]] - input[[i, j, k]]).abs() < 1e-12,
                        "mismatch at ({i},{j},{k}): got {}, expected {}",
                        output[[i, j, k]],
                        input[[i, j, k]]
                    );
                }
            }
        }
    }

    #[test]
    fn parseval_theorem() {
        let n = 16;
        let backend = NdrustfftBackend::new(n, n, n);
        let mut input = Array3::<f64>::zeros((n, n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let x = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                    let y = 2.0 * std::f64::consts::PI * (j as f64) / (n as f64);
                    input[[i, j, k]] = x.sin() + y.cos();
                }
            }
        }

        let phys_energy: f64 = input.iter().map(|v| v * v).sum::<f64>();

        let mut spectral = Array3::<Complex<f64>>::zeros((n, n, n / 2 + 1));
        backend.r2c_3d(&input, &mut spectral);

        // For R2C: sum |f_hat|^2 with factor 2 for non-DC/Nyquist modes
        let ntot = (n * n * n) as f64;
        let mut spec_energy = 0.0_f64;
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..(n / 2 + 1) {
                    let mag2 =
                        spectral[[ix, iy, iz]].re.powi(2) + spectral[[ix, iy, iz]].im.powi(2);
                    let weight = if iz == 0 || iz == n / 2 { 1.0 } else { 2.0 };
                    spec_energy += weight * mag2;
                }
            }
        }
        spec_energy /= ntot;

        assert!(
            (phys_energy - spec_energy).abs() / phys_energy < 1e-10,
            "Parseval violated: phys={phys_energy}, spec={spec_energy}"
        );
    }
}
