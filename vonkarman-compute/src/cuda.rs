//! GPU compute backend over `cuda-core` (Option B residency).
//!
//! [`CudaBackend`] owns the device PRIMARY context (`Arc<CudaContext>`) and
//! device buffers; kernels are loaded at runtime from offline-compiled PTX
//! (`src/ptx/`, see `KERNEL_BUILD.md`). The GPU path is **f64-only** for now:
//! the [`ComputeBackend`] trait is generic over `F: Float`, but every method
//! checks `F == f64` and works through the `f64` representation. A non-f64 `F`
//! triggers a documented panic rather than silent wrong results.
//!
//! # Complex buffer layout
//!
//! A complex buffer of `n` logical elements is stored as a `DeviceBuffer<f64>`
//! of length `2 * n`, **interleaved** `[re0, im0, re1, im1, ...]`. This matches
//! the memory layout of `num_complex::Complex<f64>` (`#[repr(C)]`, two `f64`
//! fields), so host upload/download are flat reinterpreting copies. cuFFT
//! (added later) reads this same interleaved layout directly off
//! [`CplxBuf::buf`]'s device pointer.

use crate::ptx;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig, launch_kernel_on_stream};
use num_complex::Complex;
use std::any::TypeId;
use std::sync::Arc;
use vonkarman_core::{ComputeBackend, float::Float};

/// A real device buffer of `f64` scalars.
///
/// Generic over `F` only to satisfy the trait associated type; the GPU path is
/// f64-only, so the stored element type is always `f64`.
pub struct RealBuf<F> {
    /// Device storage, length = logical element count.
    pub buf: DeviceBuffer<f64>,
    _marker: std::marker::PhantomData<F>,
}

/// A complex device buffer stored as interleaved `[re, im, ...]` `f64`.
///
/// `len` is the logical complex element count; `buf` has length `2 * len`.
pub struct CplxBuf<F> {
    /// Device storage, length = `2 * len` (interleaved re/im).
    pub buf: DeviceBuffer<f64>,
    /// Logical complex element count.
    pub len: usize,
    _marker: std::marker::PhantomData<F>,
}

// SAFETY: both buffers own a DeviceBuffer<f64> (itself Send) plus a
// zero-sized PhantomData<F> where F: Send (Float: Send). No thread affinity.
unsafe impl<F: Send> Send for RealBuf<F> {}
unsafe impl<F: Send> Send for CplxBuf<F> {}

/// GPU compute backend.
///
/// Holds the device primary context. Cheap to clone (an `Arc` bump). Bind the
/// context to the current thread with [`CudaContext::bind_to_thread`] (done in
/// [`CudaBackend::new`] and again inside every launch helper) before issuing
/// work from a worker thread.
#[derive(Debug, Clone)]
pub struct CudaBackend {
    ctx: Arc<CudaContext>,
}

/// Asserts at runtime that the generic `F` is `f64`, the only precision the
/// GPU path currently supports.
#[inline]
fn assert_f64<F: Float>() {
    assert!(
        TypeId::of::<F>() == TypeId::of::<f64>(),
        "CudaBackend supports only f64 in Phase 1; got a non-f64 Float"
    );
}

impl CudaBackend {
    /// Builds a GPU backend bound to CUDA device `ordinal`.
    ///
    /// Retains the device primary context and binds it to the calling thread.
    /// Returns the driver error if no such device exists or the driver cannot
    /// be initialised (so callers can fall back to the CPU backend).
    pub fn new(ordinal: usize) -> Result<Self, cuda_core::DriverError> {
        let ctx = CudaContext::new(ordinal)?;
        ctx.bind_to_thread()?;
        Ok(Self { ctx })
    }

    /// Borrows the owned CUDA context (for cuFFT plan creation later).
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Uploads a host real field into a fresh device [`RealBuf`].
    ///
    /// f64-only (Phase 1). Synchronises the copy before returning so the buffer
    /// is ready for a launch on the default stream.
    pub fn upload_real<F: Float>(&self, host: &[f64]) -> RealBuf<F> {
        assert_f64::<F>();
        let stream = self.ctx.default_stream();
        let buf = DeviceBuffer::<f64>::from_host(&stream, host)
            .expect("CudaBackend::upload_real device copy failed");
        stream
            .synchronize()
            .expect("CudaBackend::upload_real stream sync failed");
        RealBuf {
            buf,
            _marker: std::marker::PhantomData,
        }
    }

    /// Downloads a device [`RealBuf`] into a host `Vec<f64>` (f64-only).
    pub fn download_real<F: Float>(&self, src: &RealBuf<F>) -> Vec<f64> {
        assert_f64::<F>();
        let stream = self.ctx.default_stream();
        src.buf
            .to_host_vec(&stream)
            .expect("CudaBackend::download_real device copy failed")
    }

    /// Cross product `c = u x omega`, pointwise over resident device buffers.
    ///
    /// Loads the offline-compiled `cross_product` kernel from the checked-in
    /// PTX and launches it on the default stream, then synchronises. All nine
    /// buffers must hold the same number of grid points.
    ///
    /// Mirrors [`crate::ops::cross::cross_product_inplace`]; the GPU fuses each
    /// `a * b - c * d` into one `fma.rn.f64`, so results differ from the
    /// twice-rounded scalar CPU path by about one ULP.
    #[allow(clippy::too_many_arguments)]
    pub fn cross_product<F: Float>(
        &self,
        ux: &RealBuf<F>,
        uy: &RealBuf<F>,
        uz: &RealBuf<F>,
        ox: &RealBuf<F>,
        oy: &RealBuf<F>,
        oz: &RealBuf<F>,
        cx: &mut RealBuf<F>,
        cy: &mut RealBuf<F>,
        cz: &mut RealBuf<F>,
    ) -> Result<(), cuda_core::DriverError> {
        assert_f64::<F>();
        let n = ux.buf.len();
        debug_assert!(
            [
                uy.buf.len(),
                uz.buf.len(),
                ox.buf.len(),
                oy.buf.len(),
                oz.buf.len(),
                cx.buf.len(),
                cy.buf.len(),
                cz.buf.len(),
            ]
            .iter()
            .all(|&l| l == n),
            "cross_product: all buffers must have equal length"
        );

        let stream = self.ctx.default_stream();
        let module = self.ctx.load_module_from_ptx_src(ptx::KERNELS)?;
        let func = module.load_function("cross_product")?;

        // Each &[f64] / DisjointSlice<f64> kernel param lowers to a (ptr, len)
        // pair, in source order: ux, uy, uz, ox, oy, oz, cx, cy, cz.
        let mut ptrs = [
            ux.buf.cu_deviceptr(),
            uy.buf.cu_deviceptr(),
            uz.buf.cu_deviceptr(),
            ox.buf.cu_deviceptr(),
            oy.buf.cu_deviceptr(),
            oz.buf.cu_deviceptr(),
            cx.buf.cu_deviceptr(),
            cy.buf.cu_deviceptr(),
            cz.buf.cu_deviceptr(),
        ];
        let mut lens: [u64; 9] = [n as u64; 9];

        let mut params: [*mut std::ffi::c_void; 18] = [std::ptr::null_mut(); 18];
        for k in 0..9 {
            params[2 * k] = &mut ptrs[k] as *mut _ as *mut std::ffi::c_void;
            params[2 * k + 1] = &mut lens[k] as *mut _ as *mut std::ffi::c_void;
        }

        let cfg = LaunchConfig::for_num_elems(n as u32);
        // SAFETY: `func` came from a module loaded on `self.ctx`, the stream
        // belongs to the same context, the 18 params alias live device
        // pointers / lengths in the (ptr, len) order the PTX expects, and the
        // grid is sized for `n` elements. Params outlive the call (synchronise
        // below before they drop).
        unsafe {
            launch_kernel_on_stream(
                &func,
                cfg.grid_dim,
                cfg.block_dim,
                cfg.shared_mem_bytes,
                &stream,
                &mut params,
            )?;
        }
        stream.synchronize()
    }
}

impl ComputeBackend for CudaBackend {
    type RealBuf<F: Float> = RealBuf<F>;
    type CplxBuf<F: Float> = CplxBuf<F>;

    fn name(&self) -> &str {
        "cuda (cuda-oxide)"
    }

    fn alloc_real<F: Float>(&self, len: usize) -> Self::RealBuf<F> {
        assert_f64::<F>();
        let stream = self.ctx.default_stream();
        let buf = DeviceBuffer::<f64>::zeroed(&stream, len)
            .expect("CudaBackend::alloc_real device allocation failed");
        RealBuf {
            buf,
            _marker: std::marker::PhantomData,
        }
    }

    fn alloc_cplx<F: Float>(&self, len: usize) -> Self::CplxBuf<F> {
        assert_f64::<F>();
        let stream = self.ctx.default_stream();
        let buf = DeviceBuffer::<f64>::zeroed(&stream, 2 * len)
            .expect("CudaBackend::alloc_cplx device allocation failed");
        CplxBuf {
            buf,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    fn upload_cplx<F: Float>(&self, host: &[Complex<F>], dst: &mut Self::CplxBuf<F>) {
        assert_f64::<F>();
        debug_assert_eq!(host.len(), dst.len, "upload length mismatch");
        // Complex<f64> is #[repr(C)] of two f64, so the host slice is already
        // the interleaved [re, im, ...] f64 layout the device buffer expects.
        // SAFETY: F == f64 (checked), Complex<f64> is two contiguous f64 with
        // no padding, so reinterpreting as 2*len f64 is sound.
        let flat: &[f64] = unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const f64, host.len() * 2)
        };
        let stream = self.ctx.default_stream();
        let new = DeviceBuffer::<f64>::from_host(&stream, flat)
            .expect("CudaBackend::upload_cplx device copy failed");
        stream
            .synchronize()
            .expect("CudaBackend::upload_cplx stream sync failed");
        dst.buf = new;
    }

    fn download_cplx<F: Float>(&self, src: &Self::CplxBuf<F>, host: &mut [Complex<F>]) {
        assert_f64::<F>();
        debug_assert_eq!(host.len(), src.len, "download length mismatch");
        let stream = self.ctx.default_stream();
        let flat = src
            .buf
            .to_host_vec(&stream)
            .expect("CudaBackend::download_cplx device copy failed");
        // SAFETY: F == f64 (checked); reinterpret the host complex slice as
        // 2*len interleaved f64 and copy the flat device contents into it.
        let dst: &mut [f64] = unsafe {
            std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut f64, host.len() * 2)
        };
        dst.copy_from_slice(&flat);
    }

    fn sum_norm_sq<F: Float>(&self, buf: &Self::CplxBuf<F>) -> F {
        assert_f64::<F>();
        // TODO: on-device reduction. This downloads and reuses the CPU Kahan
        // reduction, which BREAKS residency (a device->host round trip every
        // call). A later task replaces it with an on-device reduction kernel.
        let stream = self.ctx.default_stream();
        let flat = buf
            .buf
            .to_host_vec(&stream)
            .expect("CudaBackend::sum_norm_sq device copy failed");
        let mut sum = 0.0_f64;
        let mut comp = 0.0_f64;
        for pair in flat.chunks_exact(2) {
            let term = pair[0] * pair[0] + pair[1] * pair[1];
            let y = term - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        F::from_f64(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a backend on device 0, or returns `None` (printing why) when no
    /// usable GPU is present so the test SKIPS rather than fails in CI.
    fn backend_or_skip() -> Option<CudaBackend> {
        match CudaBackend::new(0) {
            Ok(be) => Some(be),
            Err(e) => {
                eprintln!("skipping GPU test: CudaContext::new(0) failed: {e}");
                None
            }
        }
    }

    #[test]
    fn cplx_upload_download_roundtrip() {
        let Some(be) = backend_or_skip() else {
            return;
        };
        let host: Vec<Complex<f64>> = (0..128)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut dev = be.alloc_cplx::<f64>(host.len());
        be.upload_cplx(&host, &mut dev);
        let mut back = vec![Complex::new(0.0_f64, 0.0); host.len()];
        be.download_cplx(&dev, &mut back);
        assert_eq!(host, back, "complex buffer roundtrip must be bit-identical");
    }

    #[test]
    fn cplx_sum_norm_sq_matches_cpu() {
        let Some(be) = backend_or_skip() else {
            return;
        };
        // |3 + 4i|^2 = 25, four of them => 100.
        let host = vec![Complex::new(3.0_f64, 4.0); 4];
        let mut dev = be.alloc_cplx::<f64>(host.len());
        be.upload_cplx(&host, &mut dev);
        assert!((be.sum_norm_sq::<f64>(&dev) - 100.0).abs() < 1e-12);
    }
}
