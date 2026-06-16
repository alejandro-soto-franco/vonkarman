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

    /// Spectral curl `omega_hat = i k x u_hat` over resident device buffers.
    ///
    /// Loads the offline-compiled `curl` kernel from the checked-in PTX and
    /// launches it on the default stream, then synchronises. The velocity
    /// inputs and vorticity outputs are interleaved complex buffers of `n`
    /// spectral points (length `2 n`); the wavenumbers `kx`/`ky`/`kz` are real
    /// buffers of length `n`, one value per spectral grid point in the same
    /// flat order.
    ///
    /// Mirrors [`crate::ops::curl::curl_inplace`]; the GPU fuses each
    /// `a * b - c * d` into one `fma.rn.f64`, so results differ from the
    /// twice-rounded scalar CPU path by about one ULP per real output.
    #[allow(clippy::too_many_arguments)]
    pub fn curl<F: Float>(
        &self,
        ux: &CplxBuf<F>,
        uy: &CplxBuf<F>,
        uz: &CplxBuf<F>,
        kx: &RealBuf<F>,
        ky: &RealBuf<F>,
        kz: &RealBuf<F>,
        ox: &mut CplxBuf<F>,
        oy: &mut CplxBuf<F>,
        oz: &mut CplxBuf<F>,
    ) -> Result<(), cuda_core::DriverError> {
        assert_f64::<F>();
        let n = kx.buf.len();
        debug_assert!(
            ky.buf.len() == n
                && kz.buf.len() == n
                && ux.buf.len() == 2 * n
                && uy.buf.len() == 2 * n
                && uz.buf.len() == 2 * n
                && ox.buf.len() == 2 * n
                && oy.buf.len() == 2 * n
                && oz.buf.len() == 2 * n,
            "curl: complex buffers must be length 2n, wavenumber buffers length n"
        );

        let stream = self.ctx.default_stream();
        let module = self.ctx.load_module_from_ptx_src(ptx::KERNELS)?;
        let func = module.load_function("curl")?;

        // Param order matches the kernel signature: ux, uy, uz, kx, ky, kz,
        // ox, oy, oz. Each &[f64] / DisjointSlice<f64> lowers to a (ptr, len)
        // pair. The complex buffers report length 2n; the wavenumber buffers n.
        let mut ptrs = [
            ux.buf.cu_deviceptr(),
            uy.buf.cu_deviceptr(),
            uz.buf.cu_deviceptr(),
            kx.buf.cu_deviceptr(),
            ky.buf.cu_deviceptr(),
            kz.buf.cu_deviceptr(),
            ox.buf.cu_deviceptr(),
            oy.buf.cu_deviceptr(),
            oz.buf.cu_deviceptr(),
        ];
        let mut lens: [u64; 9] = [
            (2 * n) as u64,
            (2 * n) as u64,
            (2 * n) as u64,
            n as u64,
            n as u64,
            n as u64,
            (2 * n) as u64,
            (2 * n) as u64,
            (2 * n) as u64,
        ];

        let mut params: [*mut std::ffi::c_void; 18] = [std::ptr::null_mut(); 18];
        for k in 0..9 {
            params[2 * k] = &mut ptrs[k] as *mut _ as *mut std::ffi::c_void;
            params[2 * k + 1] = &mut lens[k] as *mut _ as *mut std::ffi::c_void;
        }

        // One thread per spectral grid point (n), not per f64 element.
        let cfg = LaunchConfig::for_num_elems(n as u32);
        // SAFETY: `func` came from a module loaded on `self.ctx`; the stream
        // belongs to the same context; the 18 params alias live device
        // pointers / lengths in the (ptr, len) order the PTX expects; the grid
        // is sized for `n` grid points and the kernel bounds-checks `i < n`.
        // Params outlive the call (synchronise below before they drop).
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

    /// Leray projection `u_hat -= k (k . u_hat) / |k|^2`, in place over resident
    /// device buffers.
    ///
    /// Loads the offline-compiled `leray` kernel from the checked-in PTX and
    /// launches it on the default stream, then synchronises. The velocity
    /// components `ux`/`uy`/`uz` are interleaved complex buffers of `n` spectral
    /// points (length `2 n`), modified in place; `kx`/`ky`/`kz` and `k2`
    /// (`= |k|^2`, the value the CPU stores in `k_mag_sq`) are real buffers of
    /// length `n` in the same flat order. The `k = 0` mode is left unchanged.
    ///
    /// Mirrors [`crate::ops::leray::leray_inplace`]; the GPU fuses the final
    /// `a - b * c` subtraction (and the dot-product accumulation) into
    /// `fma.rn.f64`, so results differ from the twice-rounded scalar CPU path by
    /// about one ULP per real output.
    #[allow(clippy::too_many_arguments)]
    pub fn leray_project<F: Float>(
        &self,
        kx: &RealBuf<F>,
        ky: &RealBuf<F>,
        kz: &RealBuf<F>,
        k2: &RealBuf<F>,
        ux: &mut CplxBuf<F>,
        uy: &mut CplxBuf<F>,
        uz: &mut CplxBuf<F>,
    ) -> Result<(), cuda_core::DriverError> {
        assert_f64::<F>();
        let n = kx.buf.len();
        debug_assert!(
            ky.buf.len() == n
                && kz.buf.len() == n
                && k2.buf.len() == n
                && ux.buf.len() == 2 * n
                && uy.buf.len() == 2 * n
                && uz.buf.len() == 2 * n,
            "leray_project: complex buffers must be length 2n, wavenumber buffers length n"
        );

        let stream = self.ctx.default_stream();
        let module = self.ctx.load_module_from_ptx_src(ptx::KERNELS)?;
        let func = module.load_function("leray")?;

        // Param order matches the kernel signature: kx, ky, kz, k2, ux, uy, uz.
        // Each &[f64] / DisjointSlice<f64> lowers to a (ptr, len) pair. The
        // wavenumber buffers report length n; the complex buffers 2n.
        let mut ptrs = [
            kx.buf.cu_deviceptr(),
            ky.buf.cu_deviceptr(),
            kz.buf.cu_deviceptr(),
            k2.buf.cu_deviceptr(),
            ux.buf.cu_deviceptr(),
            uy.buf.cu_deviceptr(),
            uz.buf.cu_deviceptr(),
        ];
        let mut lens: [u64; 7] = [
            n as u64,
            n as u64,
            n as u64,
            n as u64,
            (2 * n) as u64,
            (2 * n) as u64,
            (2 * n) as u64,
        ];

        let mut params: [*mut std::ffi::c_void; 14] = [std::ptr::null_mut(); 14];
        for k in 0..7 {
            params[2 * k] = &mut ptrs[k] as *mut _ as *mut std::ffi::c_void;
            params[2 * k + 1] = &mut lens[k] as *mut _ as *mut std::ffi::c_void;
        }

        // One thread per spectral grid point (n).
        let cfg = LaunchConfig::for_num_elems(n as u32);
        // SAFETY: `func` came from a module loaded on `self.ctx`; the stream
        // belongs to the same context; the 14 params alias live device
        // pointers / lengths in the (ptr, len) order the PTX expects; the grid
        // is sized for `n` grid points and the kernel bounds-checks `i < n`.
        // Params outlive the call (synchronise below before they drop).
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

    /// FMA-matched CPU oracle for the cross product.
    ///
    /// The GPU kernel computes each component as `a * b - c * d`, which the
    /// fork backend fuses into one `fma.rn.f64`: the second product `c * d` is
    /// rounded once, then `a * b - (c * d)` is computed with a single rounding.
    /// To compare against a few ULP rather than papering over a ~1 ULP gap, the
    /// oracle reproduces that rounding with `f64::mul_add`: `a.mul_add(b, -(c*d))`.
    #[allow(clippy::too_many_arguments)]
    fn cross_oracle_fma(
        ux: &[f64],
        uy: &[f64],
        uz: &[f64],
        ox: &[f64],
        oy: &[f64],
        oz: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = ux.len();
        let mut cx = vec![0.0; n];
        let mut cy = vec![0.0; n];
        let mut cz = vec![0.0; n];
        for i in 0..n {
            cx[i] = uy[i].mul_add(oz[i], -(uz[i] * oy[i]));
            cy[i] = uz[i].mul_add(ox[i], -(ux[i] * oz[i]));
            cz[i] = ux[i].mul_add(oy[i], -(uy[i] * ox[i]));
        }
        (cx, cy, cz)
    }

    #[test]
    fn cross_product_matches_cpu_oracle() {
        let Some(be) = backend_or_skip() else {
            return;
        };

        // Deterministic pseudo-random physical fields with non-trivial
        // mantissas so the fused-vs-unfused rounding actually differs.
        let n = 4096;
        let make = |seed: f64| -> Vec<f64> {
            (0..n)
                .map(|i| (seed + 0.013 * i as f64).sin() * 1.7 + 0.3)
                .collect()
        };
        let ux = make(0.1);
        let uy = make(1.3);
        let uz = make(2.7);
        let ox = make(3.9);
        let oy = make(4.2);
        let oz = make(5.6);

        // GPU path on resident buffers.
        let dux = be.upload_real::<f64>(&ux);
        let duy = be.upload_real::<f64>(&uy);
        let duz = be.upload_real::<f64>(&uz);
        let dox = be.upload_real::<f64>(&ox);
        let doy = be.upload_real::<f64>(&oy);
        let doz = be.upload_real::<f64>(&oz);
        let mut dcx = be.alloc_real::<f64>(n);
        let mut dcy = be.alloc_real::<f64>(n);
        let mut dcz = be.alloc_real::<f64>(n);
        be.cross_product::<f64>(
            &dux, &duy, &duz, &dox, &doy, &doz, &mut dcx, &mut dcy, &mut dcz,
        )
        .expect("GPU cross_product launch failed");
        let gcx = be.download_real::<f64>(&dcx);
        let gcy = be.download_real::<f64>(&dcy);
        let gcz = be.download_real::<f64>(&dcz);

        // FMA-matched CPU oracle. We use the mul_add oracle (not the plain
        // twice-rounded ops::cross body) so the tolerance can be tight: the GPU
        // fuses, so matching the fusion on the CPU isolates real divergence
        // from a benign ~1 ULP rounding difference.
        let (ocx, ocy, ocz) = cross_oracle_fma(&ux, &uy, &uz, &ox, &oy, &oz);

        // A few ULP, expressed as a relative tolerance. With the matched oracle
        // the agreement is essentially bit-exact; 1e-15 leaves head-room for
        // the handful of points where the driver JIT reorders within the FMA.
        let tol = 1e-15;
        for i in 0..n {
            for (g, o) in [(gcx[i], ocx[i]), (gcy[i], ocy[i]), (gcz[i], ocz[i])] {
                let rel = (g - o).abs() / o.abs().max(f64::MIN_POSITIVE);
                assert!(
                    rel <= tol,
                    "mismatch at {i}: gpu {g}, oracle {o}, rel {rel:e} > {tol:e}"
                );
            }
        }

        // Cross-check against the plain twice-rounded CPU body too, so the
        // kernel maths (not just the fusion) is correct. Here the GPU fuses and
        // the CPU does not, so the gap is one fused-vs-unfused product. Its
        // magnitude is bounded ABSOLUTELY by ~eps * |product|, not relatively
        // by the result, which can be tiny under catastrophic cancellation
        // (e.g. cx ~ 1e-4 differencing two O(1) products). With operands in
        // roughly [-1.4, 2.0] the products are O(1), so a small multiple of
        // f64::EPSILON is the right absolute bound; a relative bound on the
        // cancelled result would spuriously blow up.
        let mut ccx = vec![0.0; n];
        let mut ccy = vec![0.0; n];
        let mut ccz = vec![0.0; n];
        crate::ops::cross::cross_product_inplace::<f64>(
            &ux, &uy, &uz, &ox, &oy, &oz, &mut ccx, &mut ccy, &mut ccz,
        );
        let abs_tol = 8.0 * f64::EPSILON;
        for i in 0..n {
            for (g, c) in [(gcx[i], ccx[i]), (gcy[i], ccy[i]), (gcz[i], ccz[i])] {
                let abs = (g - c).abs();
                assert!(
                    abs <= abs_tol,
                    "twice-rounded mismatch at {i}: gpu {g}, cpu {c}, abs {abs:e} > {abs_tol:e}"
                );
            }
        }
    }

    /// FMA-matched CPU oracle for the spectral curl.
    ///
    /// The kernel writes each real output as `a * b - c * d` and the fork fuses
    /// it into one `fma.rn.f64` (the second product is rounded once, then
    /// `a * b - (c * d)` is one rounding). The oracle reproduces that with
    /// `f64::mul_add` so the comparison isolates real divergence from a benign
    /// ~1 ULP fused-vs-unfused gap.
    #[allow(clippy::too_many_arguments)]
    fn curl_oracle_fma(
        ux: &[f64],
        uy: &[f64],
        uz: &[f64],
        kx: &[f64],
        ky: &[f64],
        kz: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = kx.len();
        let mut ox = vec![0.0; 2 * n];
        let mut oy = vec![0.0; 2 * n];
        let mut oz = vec![0.0; 2 * n];
        for i in 0..n {
            let (re, im) = (2 * i, 2 * i + 1);
            let (kxi, kyi, kzi) = (kx[i], ky[i], kz[i]);
            let (uxr, uxi) = (ux[re], ux[im]);
            let (uyr, uyi) = (uy[re], uy[im]);
            let (uzr, uzi) = (uz[re], uz[im]);
            // -(ky uz - kz uy) = (kz uy) - (ky uz); fuse as kz.mul_add(uy, -(ky*uz))
            ox[re] = kzi.mul_add(uyi, -(kyi * uzi));
            ox[im] = kyi.mul_add(uzr, -(kzi * uyr));
            oy[re] = kxi.mul_add(uzi, -(kzi * uxi));
            oy[im] = kzi.mul_add(uxr, -(kxi * uzr));
            oz[re] = kyi.mul_add(uxi, -(kxi * uyi));
            oz[im] = kxi.mul_add(uyr, -(kyi * uxr));
        }
        (ox, oy, oz)
    }

    #[test]
    fn curl_matches_cpu_oracle() {
        use vonkarman_core::field::GridSpec;
        use vonkarman_core::spectral_ops::SpectralOps;

        let Some(be) = backend_or_skip() else {
            return;
        };

        // Real wavenumbers from the same source SpectralOps::curl uses, so the
        // GPU and CPU share identical k values.
        let grid = GridSpec::cubic(16, 2.0 * std::f64::consts::PI);
        let ops = SpectralOps::<f64>::new(&grid);
        let (snx, sny, snz) = grid.spectral_shape();
        let n = snx * sny * snz;
        let (mut fkx, mut fky, mut fkz) =
            (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    fkx.push(ops.kx[ix]);
                    fky.push(ops.ky[iy]);
                    fkz.push(ops.kz[iz]);
                }
            }
        }

        // Deterministic pseudo-random interleaved complex fields.
        let make = |seed: f64| -> Vec<f64> {
            (0..2 * n)
                .map(|j| (seed + 0.017 * j as f64).sin() * 1.6 + 0.25)
                .collect()
        };
        let ux = make(0.1);
        let uy = make(1.3);
        let uz = make(2.7);

        // GPU path on resident buffers. Upload k as real buffers; the velocity
        // and vorticity buffers are complex (interleaved, length 2n).
        let dkx = be.upload_real::<f64>(&fkx);
        let dky = be.upload_real::<f64>(&fky);
        let dkz = be.upload_real::<f64>(&fkz);
        let dux = be.upload_real::<f64>(&ux);
        let duy = be.upload_real::<f64>(&uy);
        let duz = be.upload_real::<f64>(&uz);
        // Reinterpret the real (2n) buffers as complex buffers of n points; the
        // device storage and flat layout are identical.
        let cux = CplxBuf::<f64> { buf: dux.buf, len: n, _marker: std::marker::PhantomData };
        let cuy = CplxBuf::<f64> { buf: duy.buf, len: n, _marker: std::marker::PhantomData };
        let cuz = CplxBuf::<f64> { buf: duz.buf, len: n, _marker: std::marker::PhantomData };
        let mut cox = be.alloc_cplx::<f64>(n);
        let mut coy = be.alloc_cplx::<f64>(n);
        let mut coz = be.alloc_cplx::<f64>(n);
        be.curl::<f64>(
            &cux, &cuy, &cuz, &dkx, &dky, &dkz, &mut cox, &mut coy, &mut coz,
        )
        .expect("GPU curl launch failed");

        let mut hox = vec![Complex::new(0.0_f64, 0.0); n];
        let mut hoy = vec![Complex::new(0.0_f64, 0.0); n];
        let mut hoz = vec![Complex::new(0.0_f64, 0.0); n];
        be.download_cplx(&cox, &mut hox);
        be.download_cplx(&coy, &mut hoy);
        be.download_cplx(&coz, &mut hoz);
        let to_flat = |h: &[Complex<f64>]| -> Vec<f64> {
            let mut v = Vec::with_capacity(2 * h.len());
            for c in h {
                v.push(c.re);
                v.push(c.im);
            }
            v
        };
        let (gox, goy, goz) = (to_flat(&hox), to_flat(&hoy), to_flat(&hoz));

        // FMA-matched oracle: agreement is essentially bit-exact at well
        // conditioned points. Each real output is a DIFFERENCE of two products
        // (`ky uz - kz uy`, and cyclic), so under catastrophic cancellation the
        // result can be ~1e-2 while the products are O(15); a pure relative
        // bound then spuriously amplifies a sub-ULP gap. So we accept a point
        // if it passes EITHER a tight relative bound OR an absolute bound sized
        // to the product magnitude (k up to ~8, u up to ~1.85 => product ~15),
        // exactly the cancellation argument the cross test documents.
        let (oox, ooy, ooz) = curl_oracle_fma(&ux, &uy, &uz, &fkx, &fky, &fkz);
        let rel_tol = 1e-14;
        let abs_floor = 16.0 * f64::EPSILON;
        for (g, o) in [(&gox, &oox), (&goy, &ooy), (&goz, &ooz)] {
            for j in 0..2 * n {
                let d = (g[j] - o[j]).abs();
                let rel = d / o[j].abs().max(f64::MIN_POSITIVE);
                assert!(
                    rel <= rel_tol || d <= abs_floor,
                    "curl oracle mismatch idx {j}: gpu {}, oracle {}, rel {rel:e} > {rel_tol:e}, abs {d:e} > {abs_floor:e}",
                    g[j],
                    o[j]
                );
            }
        }

        // Cross-check against the plain twice-rounded CPU body too, so the
        // kernel maths (not just the fusion) is correct. Here the GPU fuses and
        // the CPU body does not, so the gap is one fused-vs-unfused product;
        // its magnitude is bounded ABSOLUTELY by ~eps * |product|. The products
        // k * u are O(8 * 2) = O(16), so a small multiple of EPSILON scaled by
        // 16 is the right absolute bound (a relative bound would blow up where
        // the curl difference cancels to near zero).
        let (mut cox2, mut coy2, mut coz2) = (vec![0.0; 2 * n], vec![0.0; 2 * n], vec![0.0; 2 * n]);
        crate::ops::curl::curl_inplace::<f64>(
            &ux, &uy, &uz, &fkx, &fky, &fkz, &mut cox2, &mut coy2, &mut coz2,
        );
        let abs_tol = 16.0 * f64::EPSILON;
        for (g, c) in [(&gox, &cox2), (&goy, &coy2), (&goz, &coz2)] {
            for j in 0..2 * n {
                let abs = (g[j] - c[j]).abs();
                assert!(
                    abs <= abs_tol,
                    "curl twice-rounded mismatch idx {j}: gpu {}, cpu {}, abs {abs:e} > {abs_tol:e}",
                    g[j],
                    c[j]
                );
            }
        }
    }

    /// FMA-matched CPU oracle for the Leray projection.
    ///
    /// Reproduces the kernel's exact arithmetic grouping: pre-scale `k` by
    /// `1 / k2` (`sx = kx / k2`), then write `u - sx * kdu`, which the fork
    /// fuses into one `fma.rn.f64`. The dot product `kx u_x + ky u_y + kz u_z`
    /// also accumulates via `mul_add` to match the kernel's contracted form.
    fn leray_oracle_fma(
        kx: &[f64],
        ky: &[f64],
        kz: &[f64],
        k2: &[f64],
        ux: &[f64],
        uy: &[f64],
        uz: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = kx.len();
        let (mut ox, mut oy, mut oz) = (ux.to_vec(), uy.to_vec(), uz.to_vec());
        for i in 0..n {
            let k2i = k2[i];
            if k2i < 1e-30 {
                continue;
            }
            let (re, im) = (2 * i, 2 * i + 1);
            let (kxi, kyi, kzi) = (kx[i], ky[i], kz[i]);
            let inv_k2 = 1.0 / k2i;
            let (uxr, uxi) = (ux[re], ux[im]);
            let (uyr, uyi) = (uy[re], uy[im]);
            let (uzr, uzi) = (uz[re], uz[im]);
            // Dot product, accumulated with mul_add (matches FMA contraction).
            let kdu_re = kzi.mul_add(uzr, kyi.mul_add(uyr, kxi * uxr));
            let kdu_im = kzi.mul_add(uzi, kyi.mul_add(uyi, kxi * uxi));
            let sx = kxi * inv_k2;
            let sy = kyi * inv_k2;
            let sz = kzi * inv_k2;
            // u - s * kdu fuses to fma(-s, kdu, u).
            ox[re] = sx.mul_add(-kdu_re, uxr);
            ox[im] = sx.mul_add(-kdu_im, uxi);
            oy[re] = sy.mul_add(-kdu_re, uyr);
            oy[im] = sy.mul_add(-kdu_im, uyi);
            oz[re] = sz.mul_add(-kdu_re, uzr);
            oz[im] = sz.mul_add(-kdu_im, uzi);
        }
        (ox, oy, oz)
    }

    #[test]
    fn leray_matches_cpu_oracle_and_is_divergence_free() {
        use vonkarman_core::field::GridSpec;
        use vonkarman_core::spectral_ops::SpectralOps;

        let Some(be) = backend_or_skip() else {
            return;
        };

        // Real wavenumbers and |k|^2 from the same SpectralOps the reference
        // uses, so GPU and CPU share identical k and k2 values.
        let grid = GridSpec::cubic(16, 2.0 * std::f64::consts::PI);
        let ops = SpectralOps::<f64>::new(&grid);
        let (snx, sny, snz) = grid.spectral_shape();
        let n = snx * sny * snz;
        let (mut fkx, mut fky, mut fkz, mut fk2) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    fkx.push(ops.kx[ix]);
                    fky.push(ops.ky[iy]);
                    fkz.push(ops.kz[iz]);
                    fk2.push(ops.k_mag_sq[[ix, iy, iz]]);
                }
            }
        }

        let make = |seed: f64| -> Vec<f64> {
            (0..2 * n)
                .map(|j| (seed + 0.013 * j as f64).sin() * 1.4 + 0.35)
                .collect()
        };
        let ux = make(0.2);
        let uy = make(1.1);
        let uz = make(3.3);

        // GPU path on resident buffers; the velocity buffers are projected in
        // place. Reinterpret the uploaded real (2n) buffers as complex (n).
        let dkx = be.upload_real::<f64>(&fkx);
        let dky = be.upload_real::<f64>(&fky);
        let dkz = be.upload_real::<f64>(&fkz);
        let dk2 = be.upload_real::<f64>(&fk2);
        let dux = be.upload_real::<f64>(&ux);
        let duy = be.upload_real::<f64>(&uy);
        let duz = be.upload_real::<f64>(&uz);
        let mut cux = CplxBuf::<f64> { buf: dux.buf, len: n, _marker: std::marker::PhantomData };
        let mut cuy = CplxBuf::<f64> { buf: duy.buf, len: n, _marker: std::marker::PhantomData };
        let mut cuz = CplxBuf::<f64> { buf: duz.buf, len: n, _marker: std::marker::PhantomData };
        be.leray_project::<f64>(&dkx, &dky, &dkz, &dk2, &mut cux, &mut cuy, &mut cuz)
            .expect("GPU leray launch failed");

        let mut hux = vec![Complex::new(0.0_f64, 0.0); n];
        let mut huy = vec![Complex::new(0.0_f64, 0.0); n];
        let mut huz = vec![Complex::new(0.0_f64, 0.0); n];
        be.download_cplx(&cux, &mut hux);
        be.download_cplx(&cuy, &mut huy);
        be.download_cplx(&cuz, &mut huz);
        let to_flat = |h: &[Complex<f64>]| -> Vec<f64> {
            let mut v = Vec::with_capacity(2 * h.len());
            for c in h {
                v.push(c.re);
                v.push(c.im);
            }
            v
        };
        let (gux, guy, guz) = (to_flat(&hux), to_flat(&huy), to_flat(&huz));

        // FMA-matched oracle. The projected components are differences of an
        // O(1) field and an O(1) correction, so under cancellation a single
        // value can be small while the operands are O(1); accept EITHER a tight
        // relative bound OR an absolute bound (one fused-vs-unfused product,
        // ~eps * |product|, products O(few)), the cross/curl cancellation rule.
        let (oox, ooy, ooz) = leray_oracle_fma(&fkx, &fky, &fkz, &fk2, &ux, &uy, &uz);
        let rel_tol = 1e-14;
        let abs_floor = 8.0 * f64::EPSILON;
        for (g, o) in [(&gux, &oox), (&guy, &ooy), (&guz, &ooz)] {
            for j in 0..2 * n {
                let d = (g[j] - o[j]).abs();
                let rel = d / o[j].abs().max(f64::MIN_POSITIVE);
                assert!(
                    rel <= rel_tol || d <= abs_floor,
                    "leray oracle mismatch idx {j}: gpu {}, oracle {}, rel {rel:e}, abs {d:e}",
                    g[j],
                    o[j]
                );
            }
        }

        // The projected GPU field must be divergence-free: k . u_hat ~ 0.
        for i in 0..n {
            let (re, im) = (2 * i, 2 * i + 1);
            let div_re = fkx[i] * gux[re] + fky[i] * guy[re] + fkz[i] * guz[re];
            let div_im = fkx[i] * gux[im] + fky[i] * guy[im] + fkz[i] * guz[im];
            let mag = (div_re * div_re + div_im * div_im).sqrt();
            assert!(mag < 1e-12, "GPU divergence at flat idx {i} = {mag:e}");
        }
    }
}
