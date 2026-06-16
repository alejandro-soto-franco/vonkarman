//! GATE spike: GPU-resident in-place cuFFT round-trip on cuda-core buffers.
//!
//! Proves that a `DeviceBuffer<f64>` allocated through the cuda-core crate (the
//! cuda-oxide fork) can be transformed by cuFFT (D2Z forward, Z2D inverse) with
//! NO host copy between the transforms, and that the result round-trips to
//! double-precision tolerance.
//!
//! The single de-risk: cuFFT (driven via libloading against the CUDA runtime
//! API) executes in the device PRIMARY context. cuda-core's
//! `CudaContext::new(0)` retains that SAME primary context via
//! `cuDevicePrimaryCtxRetain`, and `ctx.bind_to_thread()` makes it current on
//! the calling thread via `cuCtxSetCurrent`. So we bind cuda-core's context to
//! this thread BEFORE every cuFFT call and they agree.

use anyhow::{Context, Result, bail};
use cuda_core::{CudaContext, DeviceBuffer};
use libloading::Library;

/// cuFFT transform-type constants.
const CUFFT_D2Z: i32 = 0x6a;
const CUFFT_Z2D: i32 = 0x6c;

/// cuFFT plan handle (opaque integer).
type CufftHandle = i32;

/// Resolved cuFFT symbols loaded from `libcufft.so` at runtime.
struct Cufft {
    _lib: Library,
    plan_3d: unsafe extern "C" fn(*mut CufftHandle, i32, i32, i32, i32) -> i32,
    exec_d2z: unsafe extern "C" fn(CufftHandle, *mut f64, *mut [f64; 2]) -> i32,
    exec_z2d: unsafe extern "C" fn(CufftHandle, *mut [f64; 2], *mut f64) -> i32,
    destroy: unsafe extern "C" fn(CufftHandle) -> i32,
}

impl Cufft {
    fn load() -> Result<Self> {
        unsafe {
            let lib = Library::new("libcufft.so").context("loading libcufft.so")?;
            let plan_3d = *lib
                .get::<unsafe extern "C" fn(*mut CufftHandle, i32, i32, i32, i32) -> i32>(
                    b"cufftPlan3d\0",
                )
                .context("cufftPlan3d")?;
            let exec_d2z = *lib
                .get::<unsafe extern "C" fn(CufftHandle, *mut f64, *mut [f64; 2]) -> i32>(
                    b"cufftExecD2Z\0",
                )
                .context("cufftExecD2Z")?;
            let exec_z2d = *lib
                .get::<unsafe extern "C" fn(CufftHandle, *mut [f64; 2], *mut f64) -> i32>(
                    b"cufftExecZ2D\0",
                )
                .context("cufftExecZ2D")?;
            let destroy = *lib
                .get::<unsafe extern "C" fn(CufftHandle) -> i32>(b"cufftDestroy\0")
                .context("cufftDestroy")?;
            Ok(Self {
                _lib: lib,
                plan_3d,
                exec_d2z,
                exec_z2d,
                destroy,
            })
        }
    }
}

fn main() -> Result<()> {
    let (nx, ny, nz) = (32usize, 32usize, 32usize);
    let n_total = nx * ny * nz;
    let n_cplx = nx * ny * (nz / 2 + 1); // hermitian half-spectrum elements

    println!("== cuFFT GPU-resident round-trip spike ==");
    println!("grid {nx}x{ny}x{nz} (n_total={n_total}, n_cplx={n_cplx})");

    // --- 1. cuda-core context (retains device PRIMARY context). -------------
    let ctx = CudaContext::new(0).context("CudaContext::new(0)")?;
    println!("device: {}", ctx.device_name()?);
    let raw_ctx = ctx.cu_ctx();
    println!("cuda-core primary CUcontext = {raw_ctx:?}");

    // cuda-core's stream-based ops bind the context internally, but the cuFFT
    // libloading calls do NOT go through cuda-core, so we must make the same
    // context current on THIS thread before any cuFFT call.
    ctx.bind_to_thread().context("bind_to_thread")?;
    let stream = ctx.default_stream();

    // --- 2. Known real field on the host (a few sine/cosine modes). ---------
    let mut field = vec![0.0f64; n_total];
    let idx = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
    let tau = std::f64::consts::TAU;
    for i in 0..nx {
        let x = tau * i as f64 / nx as f64;
        for j in 0..ny {
            let y = tau * j as f64 / ny as f64;
            for k in 0..nz {
                let z = tau * k as f64 / nz as f64;
                field[idx(i, j, k)] =
                    (x).sin() + 0.5 * (2.0 * y).cos() + 0.25 * (3.0 * z).sin() * (x).cos();
            }
        }
    }

    // --- 3. Resident device buffers (owned by cuda-core). -------------------
    // d_real holds the input real field; we deliberately keep the result of the
    // inverse transform in a SEPARATE buffer (d_real2) so we can confirm the
    // forward transform consumed the resident input. There is NO host copy
    // between the D2Z and Z2D execs.
    let d_real = DeviceBuffer::from_host(&stream, &field).context("from_host d_real")?;
    // Complex buffer represented as 2*n_cplx f64s (interleaved re,im).
    let d_cplx = DeviceBuffer::<f64>::zeroed(&stream, 2 * n_cplx).context("zeroed d_cplx")?;
    let d_real2 = DeviceBuffer::<f64>::zeroed(&stream, n_total).context("zeroed d_real2")?;
    // Ensure the async alloc + H2D copy on the stream have completed before the
    // cuFFT default-stream execs read the pointers.
    stream.synchronize().context("sync after uploads")?;

    let p_real = d_real.cu_deviceptr() as *mut f64;
    let p_cplx = d_cplx.cu_deviceptr() as *mut [f64; 2];
    let p_real2 = d_real2.cu_deviceptr() as *mut f64;
    println!("d_real  ptr = {:#x}", d_real.cu_deviceptr());
    println!("d_cplx  ptr = {:#x}", d_cplx.cu_deviceptr());
    println!("d_real2 ptr = {:#x}", d_real2.cu_deviceptr());

    // --- 4. cuFFT plans + execs, all in the bound primary context. ----------
    let cufft = Cufft::load()?;

    // Re-affirm context currency right before talking to cuFFT (defensive: in
    // the real backend other code may have run between).
    ctx.bind_to_thread().context("bind_to_thread pre-cufft")?;

    let mut plan_d2z: CufftHandle = 0;
    let mut plan_z2d: CufftHandle = 0;
    unsafe {
        let rc = (cufft.plan_3d)(&mut plan_d2z, nx as i32, ny as i32, nz as i32, CUFFT_D2Z);
        if rc != 0 {
            bail!("cufftPlan3d(D2Z) failed: code {rc}");
        }
        let rc = (cufft.plan_3d)(&mut plan_z2d, nx as i32, ny as i32, nz as i32, CUFFT_Z2D);
        if rc != 0 {
            bail!("cufftPlan3d(Z2D) failed: code {rc}");
        }

        // Forward: resident d_real -> resident d_cplx.
        let rc = (cufft.exec_d2z)(plan_d2z, p_real, p_cplx);
        if rc != 0 {
            bail!("cufftExecD2Z failed: code {rc}");
        }
        // Inverse: resident d_cplx -> resident d_real2. NO host copy in between.
        let rc = (cufft.exec_z2d)(plan_z2d, p_cplx, p_real2);
        if rc != 0 {
            bail!("cufftExecZ2D failed: code {rc}");
        }
    }

    // cuFFT execs are enqueued on the default stream of the current context;
    // synchronize the context to ensure they finished before we read back.
    ctx.synchronize().context("ctx.synchronize after execs")?;

    // --- 5. Read back, normalize, compare. ----------------------------------
    let out = d_real2.to_host_vec(&stream).context("to_host_vec d_real2")?;
    let inv_n = 1.0 / n_total as f64;

    let mut max_abs_err = 0.0f64;
    let mut max_rel_err = 0.0f64;
    let mut max_abs_field = 0.0f64;
    for (o, f) in out.iter().zip(field.iter()) {
        let recovered = o * inv_n;
        let abs_err = (recovered - f).abs();
        max_abs_err = max_abs_err.max(abs_err);
        max_abs_field = max_abs_field.max(f.abs());
        let denom = f.abs().max(1e-300);
        max_rel_err = max_rel_err.max(abs_err / denom);
    }
    // Normalized error relative to the field's peak magnitude (robust where the
    // true value is near zero).
    let norm_err = max_abs_err / max_abs_field;

    unsafe {
        (cufft.destroy)(plan_d2z);
        (cufft.destroy)(plan_z2d);
    }

    println!("max abs error          = {max_abs_err:.3e}");
    println!("max field magnitude    = {max_abs_field:.3e}");
    println!("normalized error       = {norm_err:.3e}  (abs_err / peak field)");
    println!("max pointwise rel err  = {max_rel_err:.3e}");

    let tol = 1e-13;
    if norm_err < tol {
        println!("RESULT: PASS  (normalized round-trip error {norm_err:.3e} < {tol:.0e})");
        Ok(())
    } else {
        println!("RESULT: FAIL  (normalized round-trip error {norm_err:.3e} >= {tol:.0e})");
        bail!("round-trip error {norm_err:.3e} exceeds tolerance {tol:.0e}");
    }
}
