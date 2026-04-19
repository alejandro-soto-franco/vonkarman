//! FFT backend backed by the standalone `gpufft` crate (CUDA / cuFFT).
//!
//! Provides a thin adapter over [`gpufft::cuda`] that fits vonkarman's
//! [`FftBackend<f64>`](crate::FftBackend) contract: R2C / C2R 3D transforms
//! with 1/N^3 normalisation on the inverse (scipy convention). Device
//! buffers and cuFFT plans are created once at construction and reused on
//! every call, matching the cost model of the existing `CufftBackend`.
//!
//! Compared with the legacy `CufftBackend` in `cufft.rs`, this module
//! delegates all CUDA handle lifetime and plan management to `gpufft`,
//! eliminating the hand-rolled `libloading` layer in favour of a typed,
//! build-time-linked surface.

use std::cell::RefCell;

use gpufft::cuda::{
    CudaBackend, CudaBuffer, CudaC2rPlan, CudaDevice, CudaR2cPlan, DeviceOptions,
};
use gpufft::{BufferOps, C2rPlanOps, Device, PlanDesc, R2cPlanOps, Shape};
use ndarray::Array3;
use num_complex::Complex;

use crate::backend::FftBackend;

/// Errors returned by [`GpufftBackend::new`].
#[derive(Debug, thiserror::Error)]
pub enum GpufftBackendError {
    /// Failed to create a CUDA device.
    #[error("failed to initialise CUDA device: {0}")]
    Device(#[source] gpufft::cuda::CudaError),

    /// Failed to allocate a persistent device buffer.
    #[error("failed to allocate device buffer: {0}")]
    Alloc(#[source] gpufft::cuda::CudaError),

    /// Failed to build an FFT plan.
    #[error("failed to build FFT plan: {0}")]
    Plan(#[source] gpufft::cuda::CudaError),
}

/// State mutated on every call. Wrapped in a [`RefCell`] so the backend
/// exposes `&self` methods (matching the trait) while still letting us
/// borrow the buffers and plans mutably for each dispatch.
struct Inner {
    real_buf: CudaBuffer<f64>,
    complex_buf: CudaBuffer<Complex<f64>>,
    r2c_plan: CudaR2cPlan<f64>,
    c2r_plan: CudaC2rPlan<f64>,
}

/// GPU FFT backend delegating to `gpufft` (cuFFT via typed plans).
pub struct GpufftBackend {
    #[allow(dead_code)] // kept for context lifetime
    device: CudaDevice,
    nx: usize,
    ny: usize,
    nz: usize,
    inner: RefCell<Inner>,
}

impl GpufftBackend {
    /// Create a new backend for the given grid dimensions. Allocates
    /// persistent device buffers and cuFFT plans up front.
    pub fn new(nx: usize, ny: usize, nz: usize) -> Result<Self, GpufftBackendError> {
        let device =
            CudaBackend::new_device(DeviceOptions::default()).map_err(GpufftBackendError::Device)?;

        let real_total = nx * ny * nz;
        let complex_total = nx * ny * (nz / 2 + 1);

        let real_buf = device
            .alloc::<f64>(real_total)
            .map_err(GpufftBackendError::Alloc)?;
        let complex_buf = device
            .alloc::<Complex<f64>>(complex_total)
            .map_err(GpufftBackendError::Alloc)?;

        let desc = PlanDesc {
            shape: Shape::D3([nx as u32, ny as u32, nz as u32]),
            batch: 1,
            normalize: false, // we scale in-host after c2r to match scipy convention
        };

        let r2c_plan = device
            .plan_r2c::<f64>(&desc)
            .map_err(GpufftBackendError::Plan)?;
        let c2r_plan = device
            .plan_c2r::<f64>(&desc)
            .map_err(GpufftBackendError::Plan)?;

        Ok(Self {
            device,
            nx,
            ny,
            nz,
            inner: RefCell::new(Inner {
                real_buf,
                complex_buf,
                r2c_plan,
                c2r_plan,
            }),
        })
    }
}

impl FftBackend<f64> for GpufftBackend {
    fn r2c_3d(&self, input: &Array3<f64>, output: &mut Array3<Complex<f64>>) {
        let inner = &mut *self.inner.borrow_mut();
        let input_slice = input
            .as_slice()
            .expect("gpufft backend requires contiguous, standard-layout input");
        let output_slice = output
            .as_slice_mut()
            .expect("gpufft backend requires contiguous, standard-layout output");

        inner
            .real_buf
            .write(input_slice)
            .expect("gpufft: host-to-device upload failed");
        inner
            .r2c_plan
            .execute(&inner.real_buf, &mut inner.complex_buf)
            .expect("gpufft: R2C execution failed");
        inner
            .complex_buf
            .read(output_slice)
            .expect("gpufft: device-to-host download failed");
    }

    fn c2r_3d(&self, input: &Array3<Complex<f64>>, output: &mut Array3<f64>) {
        let inner = &mut *self.inner.borrow_mut();
        let input_slice = input
            .as_slice()
            .expect("gpufft backend requires contiguous, standard-layout input");
        let output_slice = output
            .as_slice_mut()
            .expect("gpufft backend requires contiguous, standard-layout output");

        inner
            .complex_buf
            .write(input_slice)
            .expect("gpufft: host-to-device upload failed");
        inner
            .c2r_plan
            .execute(&inner.complex_buf, &mut inner.real_buf)
            .expect("gpufft: C2R execution failed");
        inner
            .real_buf
            .read(output_slice)
            .expect("gpufft: device-to-host download failed");

        // cuFFT does not apply 1/N normalisation on inverse transforms;
        // scipy convention (which `FftBackend::c2r_3d` documents) does.
        let n_total = (self.nx * self.ny * self.nz) as f64;
        let inv = 1.0 / n_total;
        for x in output_slice.iter_mut() {
            *x *= inv;
        }
    }

    fn name(&self) -> &str {
        "gpufft (cuFFT via gpufft crate)"
    }

    fn precision_digits(&self) -> usize {
        15
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn device_available() -> bool {
        GpufftBackend::new(8, 8, 8).is_ok()
    }

    #[test]
    fn gpufft_roundtrip_identity() {
        if !device_available() {
            return;
        }
        let n = 8;
        let backend = GpufftBackend::new(n, n, n).unwrap();
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
                        (output[[i, j, k]] - input[[i, j, k]]).abs() < 1e-10,
                        "mismatch at ({i},{j},{k}): got {}, expected {}",
                        output[[i, j, k]],
                        input[[i, j, k]]
                    );
                }
            }
        }
    }

    #[test]
    fn gpufft_name_is_descriptive() {
        if !device_available() {
            return;
        }
        let backend = GpufftBackend::new(4, 4, 4).unwrap();
        assert!(backend.name().contains("gpufft"));
    }
}
