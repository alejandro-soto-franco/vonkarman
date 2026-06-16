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
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
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
    /// Count of host<->device transfers issued through the backend's upload and
    /// download methods (`upload_real`/`download_real`/`upload_cplx`/
    /// `download_cplx`), the ONLY points that move data across the bus. Shared
    /// (`Arc`) so clones of the backend observe one counter; used by the
    /// residency proof to assert a resident FFT round trip copies nothing.
    transfers: Arc<AtomicU64>,
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
        Ok(Self {
            ctx,
            transfers: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Borrows the owned CUDA context (for cuFFT plan creation later).
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Raw device pointer of a resident complex buffer, as `*mut f64`.
    ///
    /// The buffer is interleaved `[re, im, ...]` of length `2 * buf.len` `f64`s,
    /// which is the layout cuFFT's complex transforms read and write directly.
    /// Hand this to [`vonkarman_fft::CufftBackend::r2c_3d_dev`] /
    /// [`vonkarman_fft::CufftBackend::c2r_3d_dev`] to run an FFT in place on the
    /// resident buffer with no host transfer.
    ///
    /// The pointer is owned by `buf` and stays valid only while `buf` lives in
    /// this backend's primary context; do not free or move `buf` while a
    /// transform using the pointer is in flight.
    pub fn cplx_device_ptr(&self, buf: &CplxBuf<f64>) -> *mut f64 {
        buf.buf.cu_deviceptr() as *mut f64
    }

    /// Raw device pointer of a resident real buffer, as `*mut f64`.
    ///
    /// Length `buf.buf.len()` `f64`s. The companion of [`Self::cplx_device_ptr`]
    /// for the real side of a cuFFT transform; same validity rules.
    pub fn real_device_ptr(&self, buf: &RealBuf<f64>) -> *mut f64 {
        buf.buf.cu_deviceptr() as *mut f64
    }

    /// Raw `CUstream` handle of this backend's stream, as `*mut c_void`, for
    /// [`vonkarman_fft::CufftBackend::set_stream`].
    ///
    /// The backend issues all kernels on the context default stream, whose raw
    /// handle is null (the driver interprets null as the default stream). Binding
    /// the cuFFT plans to this same handle keeps the FFT and the pointwise
    /// operators on one ordered stream in the shared primary context, so no
    /// cross-stream synchronisation is needed between them.
    pub fn stream_raw(&self) -> *mut c_void {
        self.ctx.default_stream().cu_stream() as *mut c_void
    }

    /// Number of host<->device transfers issued so far (the residency proof).
    ///
    /// Incremented once per upload/download (`upload_real`, `download_real`,
    /// `upload_cplx`, `download_cplx`), the only methods that move data across
    /// the PCIe bus. A resident FFT round trip issues no such call, so a test can
    /// bracket the round trip and assert the count rose by exactly the setup
    /// upload plus the final download.
    pub fn transfer_count(&self) -> u64 {
        self.transfers.load(Ordering::Relaxed)
    }

    /// Uploads a host real field into a fresh device [`RealBuf`].
    ///
    /// f64-only (Phase 1). Synchronises the copy before returning so the buffer
    /// is ready for a launch on the default stream.
    pub fn upload_real<F: Float>(&self, host: &[f64]) -> RealBuf<F> {
        assert_f64::<F>();
        self.transfers.fetch_add(1, Ordering::Relaxed);
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
        self.transfers.fetch_add(1, Ordering::Relaxed);
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

    /// ETD-RK4 stage 2 / stage 3 multiply: `out = exp_half * u + dt * a * n`,
    /// out of place, over resident device buffers (one velocity component).
    ///
    /// Loads the offline-compiled `etd_stage_axpy` kernel and launches it on the
    /// default stream, then synchronises. `exp_half`/`a` are real per-mode
    /// buffers (length `n`); `u`/`n`/`out` are interleaved complex buffers
    /// (length `2 n`). With `a = a21` this is stage 2; with `a = a31`, stage 3.
    ///
    /// Mirrors [`crate::ops::etd::etd_stage_axpy_inplace`]; the GPU contracts
    /// `exp_half * u + (dt a) * n` into one `fma.rn.f64` per real slot, so
    /// results differ from the twice-rounded scalar CPU path by about one ULP.
    #[allow(clippy::too_many_arguments)]
    pub fn etd_stage_axpy<F: Float>(
        &self,
        exp_half: &RealBuf<F>,
        a: &RealBuf<F>,
        dt: F,
        u: &CplxBuf<F>,
        n: &CplxBuf<F>,
        out: &mut CplxBuf<F>,
    ) -> Result<(), cuda_core::DriverError> {
        assert_f64::<F>();
        let m = exp_half.buf.len();
        debug_assert!(
            a.buf.len() == m
                && u.buf.len() == 2 * m
                && n.buf.len() == 2 * m
                && out.buf.len() == 2 * m,
            "etd_stage_axpy: complex buffers must be length 2n, coeff buffers length n"
        );

        let stream = self.ctx.default_stream();
        let module = self.ctx.load_module_from_ptx_src(ptx::KERNELS)?;
        let func = module.load_function("etd_stage_axpy")?;

        // Param order matches the kernel: exp_half, a, dt, u, n, out. The slice
        // params each lower to a (ptr, len) pair; the scalar `dt` is a single
        // f64 by value, inserted between the `a` and `u` slice pairs.
        let mut p_exp_half = exp_half.buf.cu_deviceptr();
        let mut l_exp_half = m as u64;
        let mut p_a = a.buf.cu_deviceptr();
        let mut l_a = m as u64;
        let mut v_dt = self.as_f64(dt);
        let mut p_u = u.buf.cu_deviceptr();
        let mut l_u = (2 * m) as u64;
        let mut p_n = n.buf.cu_deviceptr();
        let mut l_n = (2 * m) as u64;
        let mut p_out = out.buf.cu_deviceptr();
        let mut l_out = (2 * m) as u64;

        let mut params: [*mut std::ffi::c_void; 11] = [
            &mut p_exp_half as *mut _ as *mut std::ffi::c_void,
            &mut l_exp_half as *mut _ as *mut std::ffi::c_void,
            &mut p_a as *mut _ as *mut std::ffi::c_void,
            &mut l_a as *mut _ as *mut std::ffi::c_void,
            &mut v_dt as *mut _ as *mut std::ffi::c_void,
            &mut p_u as *mut _ as *mut std::ffi::c_void,
            &mut l_u as *mut _ as *mut std::ffi::c_void,
            &mut p_n as *mut _ as *mut std::ffi::c_void,
            &mut l_n as *mut _ as *mut std::ffi::c_void,
            &mut p_out as *mut _ as *mut std::ffi::c_void,
            &mut l_out as *mut _ as *mut std::ffi::c_void,
        ];

        // One thread per spectral mode (n).
        let cfg = LaunchConfig::for_num_elems(m as u32);
        // SAFETY: `func` came from a module loaded on `self.ctx`; the stream
        // belongs to the same context; the 11 params alias live device pointers
        // / lengths (and the scalar dt) in the order the PTX expects; the grid
        // is sized for `n` modes and the kernel bounds-checks `i < n`. Params
        // outlive the call (synchronise below before they drop).
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

    /// ETD-RK4 stage 4 multiply: `out = exp_full * u + dt * a41 * (2 n3 - n1)`,
    /// out of place, over resident device buffers (one velocity component).
    ///
    /// Loads the offline-compiled `etd_stage4` kernel and launches it on the
    /// default stream, then synchronises. `exp_full`/`a41` are real per-mode
    /// buffers (length `n`); `u`/`n1`/`n3`/`out` are interleaved complex buffers
    /// (length `2 n`).
    ///
    /// Mirrors [`crate::ops::etd::etd_stage4_inplace`]; the GPU contracts the
    /// difference `2 n3 - n1` and the outer combination into `fma.rn.f64`, so
    /// results differ from the twice-rounded scalar CPU path by about one ULP.
    #[allow(clippy::too_many_arguments)]
    pub fn etd_stage4<F: Float>(
        &self,
        exp_full: &RealBuf<F>,
        a41: &RealBuf<F>,
        dt: F,
        u: &CplxBuf<F>,
        n1: &CplxBuf<F>,
        n3: &CplxBuf<F>,
        out: &mut CplxBuf<F>,
    ) -> Result<(), cuda_core::DriverError> {
        assert_f64::<F>();
        let m = exp_full.buf.len();
        debug_assert!(
            a41.buf.len() == m
                && u.buf.len() == 2 * m
                && n1.buf.len() == 2 * m
                && n3.buf.len() == 2 * m
                && out.buf.len() == 2 * m,
            "etd_stage4: complex buffers must be length 2n, coeff buffers length n"
        );

        let stream = self.ctx.default_stream();
        let module = self.ctx.load_module_from_ptx_src(ptx::KERNELS)?;
        let func = module.load_function("etd_stage4")?;

        // Param order matches the kernel: exp_full, a41, dt, u, n1, n3, out.
        let mut p_exp_full = exp_full.buf.cu_deviceptr();
        let mut l_exp_full = m as u64;
        let mut p_a41 = a41.buf.cu_deviceptr();
        let mut l_a41 = m as u64;
        let mut v_dt = self.as_f64(dt);
        let mut p_u = u.buf.cu_deviceptr();
        let mut l_u = (2 * m) as u64;
        let mut p_n1 = n1.buf.cu_deviceptr();
        let mut l_n1 = (2 * m) as u64;
        let mut p_n3 = n3.buf.cu_deviceptr();
        let mut l_n3 = (2 * m) as u64;
        let mut p_out = out.buf.cu_deviceptr();
        let mut l_out = (2 * m) as u64;

        let mut params: [*mut std::ffi::c_void; 13] = [
            &mut p_exp_full as *mut _ as *mut std::ffi::c_void,
            &mut l_exp_full as *mut _ as *mut std::ffi::c_void,
            &mut p_a41 as *mut _ as *mut std::ffi::c_void,
            &mut l_a41 as *mut _ as *mut std::ffi::c_void,
            &mut v_dt as *mut _ as *mut std::ffi::c_void,
            &mut p_u as *mut _ as *mut std::ffi::c_void,
            &mut l_u as *mut _ as *mut std::ffi::c_void,
            &mut p_n1 as *mut _ as *mut std::ffi::c_void,
            &mut l_n1 as *mut _ as *mut std::ffi::c_void,
            &mut p_n3 as *mut _ as *mut std::ffi::c_void,
            &mut l_n3 as *mut _ as *mut std::ffi::c_void,
            &mut p_out as *mut _ as *mut std::ffi::c_void,
            &mut l_out as *mut _ as *mut std::ffi::c_void,
        ];

        let cfg = LaunchConfig::for_num_elems(m as u32);
        // SAFETY: as for etd_stage_axpy; 13 params in (ptr, len) / scalar order
        // the PTX expects, grid sized for `n` modes, kernel bounds-checks `i < n`.
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

    /// ETD-RK4 final update, in place on `u`:
    /// `u = exp_full * u + dt * (b1 n1 + b23 (n2 + n3) + b4 n4)`, over resident
    /// device buffers (one velocity component).
    ///
    /// Loads the offline-compiled `etd_final` kernel and launches it on the
    /// default stream, then synchronises. `exp_full`/`b1`/`b23`/`b4` are real
    /// per-mode buffers (length `n`); `n1`/`n2`/`n3`/`n4` are interleaved complex
    /// buffers (length `2 n`); `u` (length `2 n`) is read then written in place.
    ///
    /// Mirrors [`crate::ops::etd::etd_final_inplace`]; the GPU contracts the
    /// right-hand side and the outer combination into a short `fma.rn.f64`
    /// chain, so results differ from the twice-rounded scalar CPU path by about
    /// one ULP per real slot.
    #[allow(clippy::too_many_arguments)]
    pub fn etd_final<F: Float>(
        &self,
        exp_full: &RealBuf<F>,
        b1: &RealBuf<F>,
        b23: &RealBuf<F>,
        b4: &RealBuf<F>,
        dt: F,
        n1: &CplxBuf<F>,
        n2: &CplxBuf<F>,
        n3: &CplxBuf<F>,
        n4: &CplxBuf<F>,
        u: &mut CplxBuf<F>,
    ) -> Result<(), cuda_core::DriverError> {
        assert_f64::<F>();
        let m = exp_full.buf.len();
        debug_assert!(
            b1.buf.len() == m
                && b23.buf.len() == m
                && b4.buf.len() == m
                && n1.buf.len() == 2 * m
                && n2.buf.len() == 2 * m
                && n3.buf.len() == 2 * m
                && n4.buf.len() == 2 * m
                && u.buf.len() == 2 * m,
            "etd_final: complex buffers must be length 2n, coeff buffers length n"
        );

        let stream = self.ctx.default_stream();
        let module = self.ctx.load_module_from_ptx_src(ptx::KERNELS)?;
        let func = module.load_function("etd_final")?;

        // Param order matches the kernel:
        // exp_full, b1, b23, b4, dt, n1, n2, n3, n4, u.
        let mut p_exp_full = exp_full.buf.cu_deviceptr();
        let mut l_exp_full = m as u64;
        let mut p_b1 = b1.buf.cu_deviceptr();
        let mut l_b1 = m as u64;
        let mut p_b23 = b23.buf.cu_deviceptr();
        let mut l_b23 = m as u64;
        let mut p_b4 = b4.buf.cu_deviceptr();
        let mut l_b4 = m as u64;
        let mut v_dt = self.as_f64(dt);
        let mut p_n1 = n1.buf.cu_deviceptr();
        let mut l_n1 = (2 * m) as u64;
        let mut p_n2 = n2.buf.cu_deviceptr();
        let mut l_n2 = (2 * m) as u64;
        let mut p_n3 = n3.buf.cu_deviceptr();
        let mut l_n3 = (2 * m) as u64;
        let mut p_n4 = n4.buf.cu_deviceptr();
        let mut l_n4 = (2 * m) as u64;
        let mut p_u = u.buf.cu_deviceptr();
        let mut l_u = (2 * m) as u64;

        let mut params: [*mut std::ffi::c_void; 19] = [
            &mut p_exp_full as *mut _ as *mut std::ffi::c_void,
            &mut l_exp_full as *mut _ as *mut std::ffi::c_void,
            &mut p_b1 as *mut _ as *mut std::ffi::c_void,
            &mut l_b1 as *mut _ as *mut std::ffi::c_void,
            &mut p_b23 as *mut _ as *mut std::ffi::c_void,
            &mut l_b23 as *mut _ as *mut std::ffi::c_void,
            &mut p_b4 as *mut _ as *mut std::ffi::c_void,
            &mut l_b4 as *mut _ as *mut std::ffi::c_void,
            &mut v_dt as *mut _ as *mut std::ffi::c_void,
            &mut p_n1 as *mut _ as *mut std::ffi::c_void,
            &mut l_n1 as *mut _ as *mut std::ffi::c_void,
            &mut p_n2 as *mut _ as *mut std::ffi::c_void,
            &mut l_n2 as *mut _ as *mut std::ffi::c_void,
            &mut p_n3 as *mut _ as *mut std::ffi::c_void,
            &mut l_n3 as *mut _ as *mut std::ffi::c_void,
            &mut p_n4 as *mut _ as *mut std::ffi::c_void,
            &mut l_n4 as *mut _ as *mut std::ffi::c_void,
            &mut p_u as *mut _ as *mut std::ffi::c_void,
            &mut l_u as *mut _ as *mut std::ffi::c_void,
        ];

        let cfg = LaunchConfig::for_num_elems(m as u32);
        // SAFETY: as for etd_stage_axpy; 19 params in (ptr, len) / scalar order
        // the PTX expects, grid sized for `n` modes, kernel bounds-checks `i < n`.
        // `u` is read then written in place by the kernel (DisjointSlice).
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

    /// On-device compensated reduction of `|z|^2 = re*re + im*im` over an
    /// interleaved complex buffer, pulling back only a small scalar tail.
    ///
    /// This is the residency-preserving replacement for the old download-then-
    /// Kahan path. It runs entirely on the device except for a final transfer of
    /// at most `TAIL` partials:
    ///
    /// 1. `norm_sq_map` writes `out[i] = re*re + im*im` (length `n`) from the
    ///    interleaved input (length `2 n`).
    /// 2. `pairwise_reduce` halves the length each pass (`out[i] = in[2 i] +
    ///    in[2 i + 1]`, carrying an odd final element) until at most `TAIL`
    ///    values remain.
    /// 3. The `<= TAIL` partials are downloaded and finished with a host Kahan
    ///    sum (a negligible transfer).
    ///
    /// Pairwise summation has error `~ eps log2(n)`, comfortably under `1e-14`
    /// relative versus the CPU sequential Kahan oracle for `n` up to `256^3`.
    /// Returns the scalar sum as `f64` (the caller wraps it back into `F`).
    fn sum_norm_sq_device(&self, buf: &CplxBuf<f64>) -> f64 {
        // Stop the device tree once the working length is at or below this, then
        // finish the handful of partials on the host with a compensated sum. A
        // small tail keeps the device->host transfer negligible.
        const TAIL: usize = 256;

        let n = buf.len;
        if n == 0 {
            return 0.0;
        }

        let stream = self.ctx.default_stream();
        let module = self
            .ctx
            .load_module_from_ptx_src(ptx::KERNELS)
            .expect("sum_norm_sq_device: module load failed");

        // Step 1: map |z|^2 into a real device buffer of length n.
        let mut cur = DeviceBuffer::<f64>::zeroed(&stream, n)
            .expect("sum_norm_sq_device: map buffer alloc failed");
        {
            let func = module
                .load_function("norm_sq_map")
                .expect("sum_norm_sq_device: norm_sq_map not found");
            let mut p_in = buf.buf.cu_deviceptr();
            let mut l_in = (2 * n) as u64;
            let mut p_out = cur.cu_deviceptr();
            let mut l_out = n as u64;
            let mut params: [*mut std::ffi::c_void; 4] = [
                &mut p_in as *mut _ as *mut std::ffi::c_void,
                &mut l_in as *mut _ as *mut std::ffi::c_void,
                &mut p_out as *mut _ as *mut std::ffi::c_void,
                &mut l_out as *mut _ as *mut std::ffi::c_void,
            ];
            let cfg = LaunchConfig::for_num_elems(n as u32);
            // SAFETY: `func` comes from a module on `self.ctx`; the stream
            // belongs to the same context; the 4 params alias the live (ptr,
            // len) pairs the PTX expects; the grid is sized for `n` complex
            // elements and the kernel bounds-checks `i < n`. Params outlive the
            // launch (synchronise below before they drop).
            unsafe {
                launch_kernel_on_stream(
                    &func,
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    &stream,
                    &mut params,
                )
                .expect("sum_norm_sq_device: norm_sq_map launch failed");
            }
            stream
                .synchronize()
                .expect("sum_norm_sq_device: map sync failed");
        }

        // Step 2: pairwise tree reduction, halving the length each pass.
        let reduce = module
            .load_function("pairwise_reduce")
            .expect("sum_norm_sq_device: pairwise_reduce not found");
        let mut cur_len = n;
        while cur_len > TAIL {
            let out_len = cur_len.div_ceil(2);
            let next = DeviceBuffer::<f64>::zeroed(&stream, out_len)
                .expect("sum_norm_sq_device: reduce buffer alloc failed");
            let mut p_in = cur.cu_deviceptr();
            let mut l_in = cur_len as u64;
            let mut v_in_len = cur_len as u64;
            let mut p_out = next.cu_deviceptr();
            let mut l_out = out_len as u64;
            let mut params: [*mut std::ffi::c_void; 5] = [
                &mut p_in as *mut _ as *mut std::ffi::c_void,
                &mut l_in as *mut _ as *mut std::ffi::c_void,
                &mut v_in_len as *mut _ as *mut std::ffi::c_void,
                &mut p_out as *mut _ as *mut std::ffi::c_void,
                &mut l_out as *mut _ as *mut std::ffi::c_void,
            ];
            let cfg = LaunchConfig::for_num_elems(out_len as u32);
            // SAFETY: as above; 5 params (the `in_len` scalar sits between the
            // two (ptr, len) pairs), grid sized for `out_len` outputs, kernel
            // bounds-checks `i < out_len`. `cur` and `next` are distinct
            // buffers, so the read and write do not alias.
            unsafe {
                launch_kernel_on_stream(
                    &reduce,
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    &stream,
                    &mut params,
                )
                .expect("sum_norm_sq_device: pairwise_reduce launch failed");
            }
            stream
                .synchronize()
                .expect("sum_norm_sq_device: reduce sync failed");
            cur = next;
            cur_len = out_len;
        }

        // Step 3: download the small partial tail and finish with host Kahan.
        let tail = cur
            .to_host_vec(&stream)
            .expect("sum_norm_sq_device: tail download failed");
        let mut sum = 0.0_f64;
        let mut comp = 0.0_f64;
        for &term in tail.iter().take(cur_len) {
            let y = term - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        sum
    }

    /// Spectral zero-pad scatter for 3/2 dealiasing, over resident device
    /// buffers (pure index remap, no scaling).
    ///
    /// Loads the offline-compiled `spectral_pad` kernel and launches it on the
    /// default stream, then synchronises. `src` is an interleaved complex buffer
    /// of shape `(snx, sny, snz)` (length `2 snx sny snz`); `dst` is an
    /// interleaved complex buffer of shape `(dnx, dny, dnz)` (length
    /// `2 dnx dny dnz`). `nx`/`ny` are the original `N`-grid extents.
    ///
    /// `dst` must be PRE-ZEROED (the kernel scatters from source points only and
    /// never touches the padding region); [`Self::alloc_cplx`] zeroes on alloc.
    /// Mirrors [`crate::ops::pad::zero_pad_inplace`]; the result is bit-identical
    /// to the CPU body (no arithmetic on the copied value).
    #[allow(clippy::too_many_arguments)]
    pub fn spectral_pad<F: Float>(
        &self,
        src: &CplxBuf<F>,
        dst: &mut CplxBuf<F>,
        snx: usize,
        sny: usize,
        snz: usize,
        dnx: usize,
        dny: usize,
        dnz: usize,
        nx: usize,
        ny: usize,
    ) -> Result<(), cuda_core::DriverError> {
        assert_f64::<F>();
        debug_assert!(
            src.buf.len() == 2 * snx * sny * snz && dst.buf.len() == 2 * dnx * dny * dnz,
            "spectral_pad: src length 2*snx*sny*snz, dst length 2*dnx*dny*dnz"
        );

        let stream = self.ctx.default_stream();
        let module = self.ctx.load_module_from_ptx_src(ptx::KERNELS)?;
        let func = module.load_function("spectral_pad")?;

        // Param order matches the kernel: src (ptr, len), then the scalar dims
        // snx, sny, snz, dny, dnz, nx, ny, dnx, then dst (ptr, len).
        let mut p_src = src.buf.cu_deviceptr();
        let mut l_src = src.buf.len() as u64;
        let mut v_snx = snx as u64;
        let mut v_sny = sny as u64;
        let mut v_snz = snz as u64;
        let mut v_dny = dny as u64;
        let mut v_dnz = dnz as u64;
        let mut v_nx = nx as u64;
        let mut v_ny = ny as u64;
        let mut v_dnx = dnx as u64;
        let mut p_dst = dst.buf.cu_deviceptr();
        let mut l_dst = dst.buf.len() as u64;

        let mut params: [*mut std::ffi::c_void; 12] = [
            &mut p_src as *mut _ as *mut std::ffi::c_void,
            &mut l_src as *mut _ as *mut std::ffi::c_void,
            &mut v_snx as *mut _ as *mut std::ffi::c_void,
            &mut v_sny as *mut _ as *mut std::ffi::c_void,
            &mut v_snz as *mut _ as *mut std::ffi::c_void,
            &mut v_dny as *mut _ as *mut std::ffi::c_void,
            &mut v_dnz as *mut _ as *mut std::ffi::c_void,
            &mut v_nx as *mut _ as *mut std::ffi::c_void,
            &mut v_ny as *mut _ as *mut std::ffi::c_void,
            &mut v_dnx as *mut _ as *mut std::ffi::c_void,
            &mut p_dst as *mut _ as *mut std::ffi::c_void,
            &mut l_dst as *mut _ as *mut std::ffi::c_void,
        ];

        // One thread per SOURCE complex element.
        let cfg = LaunchConfig::for_num_elems((snx * sny * snz) as u32);
        // SAFETY: `func` comes from a module on `self.ctx`; the stream belongs
        // to the same context; the 12 params alias the live src (ptr, len), the
        // scalar dims, and the dst (ptr, len) in the order the PTX expects; the
        // grid is sized for the source point count and the kernel bounds-checks
        // `i < snx*sny*snz`. `dst` is pre-zeroed by the caller. Params outlive
        // the launch (synchronise below before they drop).
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

    /// Spectral truncate gather for 3/2 dealiasing, over resident device
    /// buffers (pure index remap, no scaling).
    ///
    /// Loads the offline-compiled `spectral_truncate` kernel and launches it on
    /// the default stream, then synchronises. `src` is an interleaved complex
    /// buffer of shape `(snx, sny, snz)`; `dst` is an interleaved complex buffer
    /// of shape `(dnx, dny, dnz)`. `nx`/`ny` are the original `N`-grid extents.
    ///
    /// Every `dst` element is written (gather), so no pre-zero is needed.
    /// Mirrors [`crate::ops::pad::truncate_inplace`]; the result is bit-identical
    /// to the CPU body.
    #[allow(clippy::too_many_arguments)]
    pub fn spectral_truncate<F: Float>(
        &self,
        src: &CplxBuf<F>,
        dst: &mut CplxBuf<F>,
        snx: usize,
        sny: usize,
        snz: usize,
        dnx: usize,
        dny: usize,
        dnz: usize,
        nx: usize,
        ny: usize,
    ) -> Result<(), cuda_core::DriverError> {
        assert_f64::<F>();
        debug_assert!(
            src.buf.len() == 2 * snx * sny * snz && dst.buf.len() == 2 * dnx * dny * dnz,
            "spectral_truncate: src length 2*snx*sny*snz, dst length 2*dnx*dny*dnz"
        );

        let stream = self.ctx.default_stream();
        let module = self.ctx.load_module_from_ptx_src(ptx::KERNELS)?;
        let func = module.load_function("spectral_truncate")?;

        // Param order matches the kernel: src (ptr, len), then the scalar dims
        // snx, sny, snz, dny, dnz, nx, ny, dnx, then dst (ptr, len).
        let mut p_src = src.buf.cu_deviceptr();
        let mut l_src = src.buf.len() as u64;
        let mut v_snx = snx as u64;
        let mut v_sny = sny as u64;
        let mut v_snz = snz as u64;
        let mut v_dny = dny as u64;
        let mut v_dnz = dnz as u64;
        let mut v_nx = nx as u64;
        let mut v_ny = ny as u64;
        let mut v_dnx = dnx as u64;
        let mut p_dst = dst.buf.cu_deviceptr();
        let mut l_dst = dst.buf.len() as u64;

        let mut params: [*mut std::ffi::c_void; 12] = [
            &mut p_src as *mut _ as *mut std::ffi::c_void,
            &mut l_src as *mut _ as *mut std::ffi::c_void,
            &mut v_snx as *mut _ as *mut std::ffi::c_void,
            &mut v_sny as *mut _ as *mut std::ffi::c_void,
            &mut v_snz as *mut _ as *mut std::ffi::c_void,
            &mut v_dny as *mut _ as *mut std::ffi::c_void,
            &mut v_dnz as *mut _ as *mut std::ffi::c_void,
            &mut v_nx as *mut _ as *mut std::ffi::c_void,
            &mut v_ny as *mut _ as *mut std::ffi::c_void,
            &mut v_dnx as *mut _ as *mut std::ffi::c_void,
            &mut p_dst as *mut _ as *mut std::ffi::c_void,
            &mut l_dst as *mut _ as *mut std::ffi::c_void,
        ];

        // One thread per DESTINATION complex element.
        let cfg = LaunchConfig::for_num_elems((dnx * dny * dnz) as u32);
        // SAFETY: as for spectral_pad; 12 params in the (ptr, len) / scalar
        // order the PTX expects, grid sized for the destination point count,
        // kernel bounds-checks `i < dnx*dny*dnz`. Every dst point is written.
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

    /// Reinterprets a generic `F` value as `f64` for the f64-only GPU path.
    ///
    /// `assert_f64::<F>()` guarantees `F == f64` at every call site, so the
    /// value already IS an `f64`; this just reads it back through `to_f64`.
    #[inline]
    fn as_f64<F: Float>(&self, x: F) -> f64 {
        x.to_f64()
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
        self.transfers.fetch_add(1, Ordering::Relaxed);
        debug_assert_eq!(host.len(), dst.len, "upload length mismatch");
        // Complex<f64> is #[repr(C)] of two f64, so the host slice is already
        // the interleaved [re, im, ...] f64 layout the device buffer expects.
        // SAFETY: F == f64 (checked), Complex<f64> is two contiguous f64 with
        // no padding, so reinterpreting as 2*len f64 is sound.
        let flat: &[f64] =
            unsafe { std::slice::from_raw_parts(host.as_ptr() as *const f64, host.len() * 2) };
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
        self.transfers.fetch_add(1, Ordering::Relaxed);
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
        // On-device compensated reduction: only a small scalar tail (<= 256
        // partials) is pulled back to the host, preserving residency. See
        // `sum_norm_sq_device` for the map + pairwise-tree strategy and its
        // error bound versus the CPU sequential Kahan oracle.
        //
        // SAFETY: `assert_f64::<F>()` above guarantees `F == f64`, and
        // `CplxBuf<F>` is `CplxBuf<f64>` for that `F` (the only `F`-typed field
        // is a zero-sized PhantomData), so the reference reinterpret is sound.
        let buf64: &CplxBuf<f64> = unsafe { &*(buf as *const CplxBuf<F> as *const CplxBuf<f64>) };
        F::from_f64(self.sum_norm_sq_device(buf64))
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
        let (mut fkx, mut fky, mut fkz) = (
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
        let cux = CplxBuf::<f64> {
            buf: dux.buf,
            len: n,
            _marker: std::marker::PhantomData,
        };
        let cuy = CplxBuf::<f64> {
            buf: duy.buf,
            len: n,
            _marker: std::marker::PhantomData,
        };
        let cuz = CplxBuf::<f64> {
            buf: duz.buf,
            len: n,
            _marker: std::marker::PhantomData,
        };
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
        let mut cux = CplxBuf::<f64> {
            buf: dux.buf,
            len: n,
            _marker: std::marker::PhantomData,
        };
        let mut cuy = CplxBuf::<f64> {
            buf: duy.buf,
            len: n,
            _marker: std::marker::PhantomData,
        };
        let mut cuz = CplxBuf::<f64> {
            buf: duz.buf,
            len: n,
            _marker: std::marker::PhantomData,
        };
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

    /// Deterministic pseudo-random real buffer of `len` values for the ETD
    /// tests (non-trivial mantissas so the fused-vs-unfused rounding differs).
    fn etd_make(seed: f64, len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| (seed + 0.013 * i as f64).sin() * 1.7 + 0.3)
            .collect()
    }

    /// FMA-matched CPU oracle for the ETD stage 2 / stage 3 multiply.
    ///
    /// The kernel writes `exp_half * u + (dt a) * n` with plain arithmetic, and
    /// the fork contracts the trailing product into the add as one
    /// `fma.rn.f64`: `(dt a) * n` is rounded once, then `exp_half * u + that` is
    /// one rounding. The oracle reproduces that with `f64::mul_add` so the
    /// comparison isolates real divergence from a benign ~1 ULP gap.
    fn etd_stage_axpy_oracle_fma(
        exp_half: &[f64],
        a: &[f64],
        dt: f64,
        u: &[f64],
        n: &[f64],
    ) -> Vec<f64> {
        let m = exp_half.len();
        let mut out = vec![0.0; 2 * m];
        for i in 0..m {
            let (re, im) = (2 * i, 2 * i + 1);
            let eh = exp_half[i];
            let dta = dt * a[i];
            out[re] = eh.mul_add(u[re], dta * n[re]);
            out[im] = eh.mul_add(u[im], dta * n[im]);
        }
        out
    }

    #[test]
    fn etd_stage_axpy_matches_cpu_oracle() {
        let Some(be) = backend_or_skip() else {
            return;
        };

        let m = 4096;
        let exp_half = etd_make(0.1, m);
        let a = etd_make(0.7, m);
        let dt = 0.013_f64;
        let u = etd_make(1.3, 2 * m);
        let n = etd_make(2.7, 2 * m);

        // GPU path on resident buffers. The real (2m) host buffers are uploaded
        // and reinterpreted as complex (m); device storage / layout are identical.
        let dexp = be.upload_real::<f64>(&exp_half);
        let da = be.upload_real::<f64>(&a);
        let du = be.upload_real::<f64>(&u);
        let dn = be.upload_real::<f64>(&n);
        let cu = CplxBuf::<f64> {
            buf: du.buf,
            len: m,
            _marker: std::marker::PhantomData,
        };
        let cn = CplxBuf::<f64> {
            buf: dn.buf,
            len: m,
            _marker: std::marker::PhantomData,
        };
        let mut cout = be.alloc_cplx::<f64>(m);
        be.etd_stage_axpy::<f64>(&dexp, &da, dt, &cu, &cn, &mut cout)
            .expect("GPU etd_stage_axpy launch failed");

        let mut hout = vec![Complex::new(0.0_f64, 0.0); m];
        be.download_cplx(&cout, &mut hout);
        let mut gout = Vec::with_capacity(2 * m);
        for c in &hout {
            gout.push(c.re);
            gout.push(c.im);
        }

        // FMA-matched oracle. Each slot is a single multiply-accumulate, so with
        // the matched oracle the agreement is essentially bit-exact; a tight
        // relative bound suffices (no cancellation in a pure axpy).
        let oracle = etd_stage_axpy_oracle_fma(&exp_half, &a, dt, &u, &n);
        let tol = 1e-15;
        for j in 0..2 * m {
            let rel = (gout[j] - oracle[j]).abs() / oracle[j].abs().max(f64::MIN_POSITIVE);
            assert!(
                rel <= tol,
                "etd_stage_axpy mismatch idx {j}: gpu {}, oracle {}, rel {rel:e} > {tol:e}",
                gout[j],
                oracle[j]
            );
        }

        // Cross-check against the plain twice-rounded CPU body: the GPU fuses,
        // the CPU body does not, so the gap is one fused-vs-unfused product,
        // bounded ABSOLUTELY by ~eps * |product|. With operands O(few) the
        // products are O(few), so a small multiple of EPSILON is the right
        // absolute bound (a pure axpy does not cancel, but we keep the same
        // absolute-floor reasoning the other ETD tests use).
        let mut cpu = vec![0.0; 2 * m];
        crate::ops::etd::etd_stage_axpy_inplace::<f64>(&exp_half, &a, dt, &u, &n, &mut cpu);
        let abs_tol = 8.0 * f64::EPSILON;
        for j in 0..2 * m {
            let abs = (gout[j] - cpu[j]).abs();
            assert!(
                abs <= abs_tol,
                "etd_stage_axpy twice-rounded mismatch idx {j}: gpu {}, cpu {}, abs {abs:e} > {abs_tol:e}",
                gout[j],
                cpu[j]
            );
        }
    }

    /// FMA-matched CPU oracle for the ETD stage 4 multiply.
    ///
    /// The kernel writes `dn = 2 n3 - n1` (which contracts to `fma(2, n3, -n1)`)
    /// then `exp_full * u + (dt a41) * dn` (which contracts to
    /// `fma(exp_full, u, dta * dn)`). The oracle reproduces both contractions
    /// with `f64::mul_add`.
    fn etd_stage4_oracle_fma(
        exp_full: &[f64],
        a41: &[f64],
        dt: f64,
        u: &[f64],
        n1: &[f64],
        n3: &[f64],
    ) -> Vec<f64> {
        let m = exp_full.len();
        let mut out = vec![0.0; 2 * m];
        for i in 0..m {
            let (re, im) = (2 * i, 2 * i + 1);
            let ef = exp_full[i];
            let dta = dt * a41[i];
            let dn_re = 2.0_f64.mul_add(n3[re], -n1[re]);
            let dn_im = 2.0_f64.mul_add(n3[im], -n1[im]);
            out[re] = ef.mul_add(u[re], dta * dn_re);
            out[im] = ef.mul_add(u[im], dta * dn_im);
        }
        out
    }

    #[test]
    fn etd_stage4_matches_cpu_oracle() {
        let Some(be) = backend_or_skip() else {
            return;
        };

        let m = 4096;
        let exp_full = etd_make(0.2, m);
        let a41 = etd_make(0.9, m);
        let dt = 0.021_f64;
        let u = etd_make(1.1, 2 * m);
        let n1 = etd_make(2.2, 2 * m);
        let n3 = etd_make(3.3, 2 * m);

        let dexp = be.upload_real::<f64>(&exp_full);
        let da = be.upload_real::<f64>(&a41);
        let du = be.upload_real::<f64>(&u);
        let dn1 = be.upload_real::<f64>(&n1);
        let dn3 = be.upload_real::<f64>(&n3);
        let cu = CplxBuf::<f64> {
            buf: du.buf,
            len: m,
            _marker: std::marker::PhantomData,
        };
        let cn1 = CplxBuf::<f64> {
            buf: dn1.buf,
            len: m,
            _marker: std::marker::PhantomData,
        };
        let cn3 = CplxBuf::<f64> {
            buf: dn3.buf,
            len: m,
            _marker: std::marker::PhantomData,
        };
        let mut cout = be.alloc_cplx::<f64>(m);
        be.etd_stage4::<f64>(&dexp, &da, dt, &cu, &cn1, &cn3, &mut cout)
            .expect("GPU etd_stage4 launch failed");

        let mut hout = vec![Complex::new(0.0_f64, 0.0); m];
        be.download_cplx(&cout, &mut hout);
        let mut gout = Vec::with_capacity(2 * m);
        for c in &hout {
            gout.push(c.re);
            gout.push(c.im);
        }

        // FMA-matched oracle. The difference `2 n3 - n1` can cancel (n1 ~ 2 n3),
        // so a single slot can be small while the operands are O(1); accept a
        // point if it passes EITHER a tight relative bound OR an absolute bound
        // sized to the O(few) operands, the cross/curl/leray cancellation rule.
        let oracle = etd_stage4_oracle_fma(&exp_full, &a41, dt, &u, &n1, &n3);
        let rel_tol = 1e-14;
        let abs_floor = 8.0 * f64::EPSILON;
        for j in 0..2 * m {
            let d = (gout[j] - oracle[j]).abs();
            let rel = d / oracle[j].abs().max(f64::MIN_POSITIVE);
            assert!(
                rel <= rel_tol || d <= abs_floor,
                "etd_stage4 mismatch idx {j}: gpu {}, oracle {}, rel {rel:e} > {rel_tol:e}, abs {d:e} > {abs_floor:e}",
                gout[j],
                oracle[j]
            );
        }

        // Cross-check against the plain twice-rounded CPU body. The GPU fuses
        // (dn and the outer combination), the CPU does not; the gap is bounded
        // ABSOLUTELY by ~eps * |product| (products O(few)).
        let mut cpu = vec![0.0; 2 * m];
        crate::ops::etd::etd_stage4_inplace::<f64>(&exp_full, &a41, dt, &u, &n1, &n3, &mut cpu);
        let abs_tol = 8.0 * f64::EPSILON;
        for j in 0..2 * m {
            let abs = (gout[j] - cpu[j]).abs();
            assert!(
                abs <= abs_tol,
                "etd_stage4 twice-rounded mismatch idx {j}: gpu {}, cpu {}, abs {abs:e} > {abs_tol:e}",
                gout[j],
                cpu[j]
            );
        }
    }

    /// FMA-matched CPU oracle for the ETD final update.
    ///
    /// The kernel writes the right-hand side as
    /// `b1 n1 + b23 (n2 + n3) + b4 n4`, parsed `((b1 n1 + b23 s23) + b4 n4)`;
    /// the contraction folds the trailing product of each add, so the inner sum
    /// is `fma(b23, s23, b1 * n1)` and the outer is `fma(b4, n4, inner)`. The
    /// final combination `exp_full * u + dt * rhs` contracts to
    /// `fma(exp_full, u, dt * rhs)`. The oracle reproduces that grouping.
    #[allow(clippy::too_many_arguments)]
    fn etd_final_oracle_fma(
        exp_full: &[f64],
        b1: &[f64],
        b23: &[f64],
        b4: &[f64],
        dt: f64,
        n1: &[f64],
        n2: &[f64],
        n3: &[f64],
        n4: &[f64],
        u: &[f64],
    ) -> Vec<f64> {
        let m = exp_full.len();
        let mut out = u.to_vec();
        for i in 0..m {
            let (re, im) = (2 * i, 2 * i + 1);
            let ef = exp_full[i];
            let (b1i, b23i, b4i) = (b1[i], b23[i], b4[i]);
            let s23_re = n2[re] + n3[re];
            let s23_im = n2[im] + n3[im];
            // fma(b4, n4, fma(b23, s23, b1 * n1)).
            let rhs_re = b4i.mul_add(n4[re], b23i.mul_add(s23_re, b1i * n1[re]));
            let rhs_im = b4i.mul_add(n4[im], b23i.mul_add(s23_im, b1i * n1[im]));
            out[re] = ef.mul_add(u[re], dt * rhs_re);
            out[im] = ef.mul_add(u[im], dt * rhs_im);
        }
        out
    }

    #[test]
    fn etd_final_matches_cpu_oracle() {
        let Some(be) = backend_or_skip() else {
            return;
        };

        let m = 4096;
        let exp_full = etd_make(0.3, m);
        let b1 = etd_make(0.5, m);
        let b23 = etd_make(0.6, m);
        let b4 = etd_make(0.8, m);
        let dt = 0.017_f64;
        let n1 = etd_make(1.0, 2 * m);
        let n2 = etd_make(1.5, 2 * m);
        let n3 = etd_make(2.0, 2 * m);
        let n4 = etd_make(2.5, 2 * m);
        let u0 = etd_make(3.0, 2 * m);

        let dexp = be.upload_real::<f64>(&exp_full);
        let db1 = be.upload_real::<f64>(&b1);
        let db23 = be.upload_real::<f64>(&b23);
        let db4 = be.upload_real::<f64>(&b4);
        let dn1 = be.upload_real::<f64>(&n1);
        let dn2 = be.upload_real::<f64>(&n2);
        let dn3 = be.upload_real::<f64>(&n3);
        let dn4 = be.upload_real::<f64>(&n4);
        let du = be.upload_real::<f64>(&u0);
        let cn1 = CplxBuf::<f64> {
            buf: dn1.buf,
            len: m,
            _marker: std::marker::PhantomData,
        };
        let cn2 = CplxBuf::<f64> {
            buf: dn2.buf,
            len: m,
            _marker: std::marker::PhantomData,
        };
        let cn3 = CplxBuf::<f64> {
            buf: dn3.buf,
            len: m,
            _marker: std::marker::PhantomData,
        };
        let cn4 = CplxBuf::<f64> {
            buf: dn4.buf,
            len: m,
            _marker: std::marker::PhantomData,
        };
        let mut cu = CplxBuf::<f64> {
            buf: du.buf,
            len: m,
            _marker: std::marker::PhantomData,
        };
        be.etd_final::<f64>(
            &dexp, &db1, &db23, &db4, dt, &cn1, &cn2, &cn3, &cn4, &mut cu,
        )
        .expect("GPU etd_final launch failed");

        let mut hu = vec![Complex::new(0.0_f64, 0.0); m];
        be.download_cplx(&cu, &mut hu);
        let mut gu = Vec::with_capacity(2 * m);
        for c in &hu {
            gu.push(c.re);
            gu.push(c.im);
        }

        // FMA-matched oracle. The right-hand side is a sum of signed products,
        // which can cancel, so a slot can be small while the products are
        // O(few); accept EITHER a tight relative bound OR an absolute bound, the
        // cross/curl/leray cancellation rule.
        let oracle = etd_final_oracle_fma(&exp_full, &b1, &b23, &b4, dt, &n1, &n2, &n3, &n4, &u0);
        let rel_tol = 1e-14;
        let abs_floor = 8.0 * f64::EPSILON;
        for j in 0..2 * m {
            let d = (gu[j] - oracle[j]).abs();
            let rel = d / oracle[j].abs().max(f64::MIN_POSITIVE);
            assert!(
                rel <= rel_tol || d <= abs_floor,
                "etd_final mismatch idx {j}: gpu {}, oracle {}, rel {rel:e} > {rel_tol:e}, abs {d:e} > {abs_floor:e}",
                gu[j],
                oracle[j]
            );
        }

        // Cross-check against the plain twice-rounded CPU body. The GPU
        // contracts the rhs chain and the outer combination; the CPU body does
        // not, so the gap accumulates a few fused-vs-unfused products, each
        // bounded by ~eps * |product| (products O(few)). A slightly larger
        // absolute multiple of EPSILON covers the short chain.
        let mut cpu = u0.clone();
        crate::ops::etd::etd_final_inplace::<f64>(
            &exp_full, &b1, &b23, &b4, dt, &n1, &n2, &n3, &n4, &mut cpu,
        );
        let abs_tol = 16.0 * f64::EPSILON;
        for j in 0..2 * m {
            let abs = (gu[j] - cpu[j]).abs();
            assert!(
                abs <= abs_tol,
                "etd_final twice-rounded mismatch idx {j}: gpu {}, cpu {}, abs {abs:e} > {abs_tol:e}",
                gu[j],
                cpu[j]
            );
        }
    }

    #[test]
    fn sum_norm_sq_device_matches_cpu_kahan() {
        use crate::cpu::CpuBackend;
        use vonkarman_core::ComputeBackend;

        let Some(be) = backend_or_skip() else {
            return;
        };
        let cpu = CpuBackend::new();

        // Several sizes including non-powers-of-2 (so the pairwise tree's odd-
        // length carry is exercised) and sizes above the host-tail cutoff so the
        // device tree actually runs multiple passes (TAIL = 256).
        for &n in &[1usize, 2, 3, 255, 257, 1000, 4095, 32768, 65537] {
            // Deterministic pseudo-random complex field with non-trivial
            // mantissas so the reduction order genuinely matters.
            let host: Vec<Complex<f64>> = (0..n)
                .map(|i| {
                    let t = 0.013 * i as f64;
                    Complex::new(t.sin() * 1.7 + 0.3, (t * 1.3).cos() * 1.1 - 0.2)
                })
                .collect();

            // CPU sequential Kahan oracle.
            let mut cbuf = cpu.alloc_cplx::<f64>(n);
            cpu.upload_cplx(&host, &mut cbuf);
            let want = cpu.sum_norm_sq::<f64>(&cbuf);

            // GPU on-device reduction (pulls only a small scalar tail).
            let mut dev = be.alloc_cplx::<f64>(n);
            be.upload_cplx(&host, &mut dev);
            let got = be.sum_norm_sq::<f64>(&dev);

            // Pairwise summation error is ~ eps log2(n), so 1e-14 relative holds
            // with comfortable head-room at every size here. `want` is strictly
            // positive (a sum of squares with a non-zero field), so a relative
            // bound is well defined.
            let rel = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE);
            assert!(
                rel <= 1e-14,
                "sum_norm_sq device vs CPU Kahan mismatch at n={n}: gpu {got}, cpu {want}, rel {rel:e}"
            );
        }
    }

    /// Build the flat interleaved host source and CPU-oracle padded/truncated
    /// destination for a pad/truncate differential test, plus the dst dims.
    fn pad_inputs(n: usize) -> (GridSpecDims, Vec<f64>) {
        let grid = vonkarman_core::field::GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let pg = grid.padded_3half();
        let (snx, sny, snz) = grid.spectral_shape();
        let (pnx, pny, pnz) = pg.spectral_shape();
        // Deterministic pseudo-random interleaved complex N-grid field.
        let src: Vec<f64> = (0..2 * snx * sny * snz)
            .map(|j| (0.017 * j as f64).sin() * 1.6 + 0.25)
            .collect();
        (
            GridSpecDims {
                snx,
                sny,
                snz,
                pnx,
                pny,
                pnz,
                nx: grid.nx,
                ny: grid.ny,
            },
            src,
        )
    }

    /// Grid dimensions for the pad/truncate differential tests.
    struct GridSpecDims {
        snx: usize,
        sny: usize,
        snz: usize,
        pnx: usize,
        pny: usize,
        pnz: usize,
        nx: usize,
        ny: usize,
    }

    #[test]
    fn spectral_pad_matches_cpu_body_bit_exact() {
        let Some(be) = backend_or_skip() else {
            return;
        };
        let (d, src) = pad_inputs(8);

        // CPU body oracle.
        let mut cpu_dst = vec![0.0; 2 * d.pnx * d.pny * d.pnz];
        crate::ops::pad::zero_pad_inplace::<f64>(
            &src,
            &mut cpu_dst,
            d.snx,
            d.sny,
            d.snz,
            d.pnx,
            d.pny,
            d.pnz,
            d.nx,
            d.ny,
        );

        // GPU path on resident buffers. Upload the N-grid source as a complex
        // buffer; the padded dst is zeroed by alloc_cplx (the scatter relies on
        // that for the padding region).
        let src_real = be.upload_real::<f64>(&src);
        let csrc = CplxBuf::<f64> {
            buf: src_real.buf,
            len: d.snx * d.sny * d.snz,
            _marker: std::marker::PhantomData,
        };
        let mut cdst = be.alloc_cplx::<f64>(d.pnx * d.pny * d.pnz);
        be.spectral_pad::<f64>(
            &csrc, &mut cdst, d.snx, d.sny, d.snz, d.pnx, d.pny, d.pnz, d.nx, d.ny,
        )
        .expect("GPU spectral_pad launch failed");

        let mut host = vec![Complex::new(0.0_f64, 0.0); d.pnx * d.pny * d.pnz];
        be.download_cplx(&cdst, &mut host);
        let mut gpu = Vec::with_capacity(2 * host.len());
        for c in &host {
            gpu.push(c.re);
            gpu.push(c.im);
        }

        // Pure data movement: the GPU result must be BIT-IDENTICAL to the CPU.
        assert_eq!(gpu.len(), cpu_dst.len());
        for (j, (g, c)) in gpu.iter().zip(cpu_dst.iter()).enumerate() {
            assert_eq!(
                g, c,
                "spectral_pad mismatch at flat idx {j}: gpu {g}, cpu {c}"
            );
        }
    }

    #[test]
    fn spectral_truncate_matches_cpu_body_bit_exact() {
        let Some(be) = backend_or_skip() else {
            return;
        };
        let (d, _) = pad_inputs(8);
        // Padded-grid source for truncation: shape (pnx, pny, pnz).
        let src: Vec<f64> = (0..2 * d.pnx * d.pny * d.pnz)
            .map(|j| (0.011 * j as f64).cos() * 1.4 - 0.15)
            .collect();

        // CPU body oracle (truncate padded -> N grid).
        let mut cpu_dst = vec![0.0; 2 * d.snx * d.sny * d.snz];
        crate::ops::pad::truncate_inplace::<f64>(
            &src,
            &mut cpu_dst,
            d.pnx,
            d.pny,
            d.pnz,
            d.snx,
            d.sny,
            d.snz,
            d.nx,
            d.ny,
        );

        // GPU path on resident buffers.
        let src_real = be.upload_real::<f64>(&src);
        let csrc = CplxBuf::<f64> {
            buf: src_real.buf,
            len: d.pnx * d.pny * d.pnz,
            _marker: std::marker::PhantomData,
        };
        let mut cdst = be.alloc_cplx::<f64>(d.snx * d.sny * d.snz);
        be.spectral_truncate::<f64>(
            &csrc, &mut cdst, d.pnx, d.pny, d.pnz, d.snx, d.sny, d.snz, d.nx, d.ny,
        )
        .expect("GPU spectral_truncate launch failed");

        let mut host = vec![Complex::new(0.0_f64, 0.0); d.snx * d.sny * d.snz];
        be.download_cplx(&cdst, &mut host);
        let mut gpu = Vec::with_capacity(2 * host.len());
        for c in &host {
            gpu.push(c.re);
            gpu.push(c.im);
        }

        assert_eq!(gpu.len(), cpu_dst.len());
        for (j, (g, c)) in gpu.iter().zip(cpu_dst.iter()).enumerate() {
            assert_eq!(
                g, c,
                "spectral_truncate mismatch at flat idx {j}: gpu {g}, cpu {c}"
            );
        }
    }
}
