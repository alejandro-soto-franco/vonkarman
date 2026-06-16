//! cuFFT backend via runtime-loaded shared libraries.
//!
//! Loads `libcufft.so` and `libcudart.so` at runtime using `libloading`.
//! No compile-time CUDA SDK dependency. Falls back gracefully when the
//! libraries are not present.

use std::ffi::c_void;
use std::ptr;

use libloading::Library;
use ndarray::Array3;
use num_complex::Complex;

use crate::backend::FftBackend;

/// cuFFT plan handle (opaque integer).
type CufftHandle = i32;

/// cuFFT transform types.
const CUFFT_D2Z: i32 = 0x6a;
const CUFFT_Z2D: i32 = 0x6c;

/// cudaMemcpy directions.
const CUDA_MEMCPY_H2D: i32 = 1;
const CUDA_MEMCPY_D2H: i32 = 2;

/// Errors from cuFFT/CUDA runtime operations.
#[derive(Debug, thiserror::Error)]
pub enum CufftError {
    #[error("failed to load {lib}: {source}")]
    LibraryLoad {
        lib: String,
        #[source]
        source: libloading::Error,
    },

    #[error("failed to resolve symbol {sym}: {source}")]
    SymbolLoad {
        sym: String,
        #[source]
        source: libloading::Error,
    },

    #[error("cudaMalloc failed (code {code})")]
    Alloc { code: i32 },

    #[error("cudaMemcpy failed (code {code})")]
    Memcpy { code: i32 },

    #[error("cufftPlan3d failed (code {code})")]
    PlanCreate { code: i32 },

    #[error("cufftExec failed (code {code})")]
    Exec { code: i32 },

    #[error("cufftSetStream failed (code {code})")]
    SetStream { code: i32 },
}

/// Loaded CUDA runtime and cuFFT function pointers.
///
/// Holds the loaded shared libraries and resolved symbols. The libraries
/// are kept alive for the lifetime of this struct via the `Library` handles.
struct CudaLibs {
    _cudart: Library,
    _cufft: Library,
    cuda_malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> i32,
    cuda_free: unsafe extern "C" fn(*mut c_void) -> i32,
    cuda_memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32,
    cufft_plan_3d: unsafe extern "C" fn(*mut CufftHandle, i32, i32, i32, i32) -> i32,
    cufft_create: unsafe extern "C" fn(*mut CufftHandle) -> i32,
    cufft_set_auto_allocation: unsafe extern "C" fn(CufftHandle, i32) -> i32,
    cufft_make_plan_3d: unsafe extern "C" fn(CufftHandle, i32, i32, i32, i32, *mut usize) -> i32,
    cufft_set_work_area: unsafe extern "C" fn(CufftHandle, *mut c_void) -> i32,
    cufft_exec_d2z: unsafe extern "C" fn(CufftHandle, *mut f64, *mut [f64; 2]) -> i32,
    cufft_exec_z2d: unsafe extern "C" fn(CufftHandle, *mut [f64; 2], *mut f64) -> i32,
    cufft_set_stream: unsafe extern "C" fn(CufftHandle, *mut c_void) -> i32,
    cufft_get_size: unsafe extern "C" fn(CufftHandle, *mut usize) -> i32,
    cufft_destroy: unsafe extern "C" fn(CufftHandle) -> i32,
}

impl CudaLibs {
    /// Try to load CUDA runtime and cuFFT shared libraries.
    fn load() -> Result<Self, CufftError> {
        unsafe {
            let cudart = Library::new("libcudart.so").map_err(|e| CufftError::LibraryLoad {
                lib: "libcudart.so".into(),
                source: e,
            })?;
            let cufft_lib = Library::new("libcufft.so").map_err(|e| CufftError::LibraryLoad {
                lib: "libcufft.so".into(),
                source: e,
            })?;

            let cuda_malloc = *cudart
                .get::<unsafe extern "C" fn(*mut *mut c_void, usize) -> i32>(b"cudaMalloc\0")
                .map_err(|e| CufftError::SymbolLoad {
                    sym: "cudaMalloc".into(),
                    source: e,
                })?;
            let cuda_free = *cudart
                .get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"cudaFree\0")
                .map_err(|e| CufftError::SymbolLoad {
                    sym: "cudaFree".into(),
                    source: e,
                })?;
            let cuda_memcpy = *cudart
                .get::<unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32>(
                    b"cudaMemcpy\0",
                )
                .map_err(|e| CufftError::SymbolLoad {
                    sym: "cudaMemcpy".into(),
                    source: e,
                })?;

            let cufft_plan_3d = *cufft_lib
                .get::<unsafe extern "C" fn(*mut CufftHandle, i32, i32, i32, i32) -> i32>(
                    b"cufftPlan3d\0",
                )
                .map_err(|e| CufftError::SymbolLoad {
                    sym: "cufftPlan3d".into(),
                    source: e,
                })?;
            let cufft_exec_d2z = *cufft_lib
                .get::<unsafe extern "C" fn(CufftHandle, *mut f64, *mut [f64; 2]) -> i32>(
                    b"cufftExecD2Z\0",
                )
                .map_err(|e| CufftError::SymbolLoad {
                    sym: "cufftExecD2Z".into(),
                    source: e,
                })?;
            let cufft_exec_z2d = *cufft_lib
                .get::<unsafe extern "C" fn(CufftHandle, *mut [f64; 2], *mut f64) -> i32>(
                    b"cufftExecZ2D\0",
                )
                .map_err(|e| CufftError::SymbolLoad {
                    sym: "cufftExecZ2D".into(),
                    source: e,
                })?;
            let cufft_set_stream = *cufft_lib
                .get::<unsafe extern "C" fn(CufftHandle, *mut c_void) -> i32>(b"cufftSetStream\0")
                .map_err(|e| CufftError::SymbolLoad {
                    sym: "cufftSetStream".into(),
                    source: e,
                })?;
            let cufft_get_size = *cufft_lib
                .get::<unsafe extern "C" fn(CufftHandle, *mut usize) -> i32>(b"cufftGetSize\0")
                .map_err(|e| CufftError::SymbolLoad {
                    sym: "cufftGetSize".into(),
                    source: e,
                })?;
            let cufft_create = *cufft_lib
                .get::<unsafe extern "C" fn(*mut CufftHandle) -> i32>(b"cufftCreate\0")
                .map_err(|e| CufftError::SymbolLoad {
                    sym: "cufftCreate".into(),
                    source: e,
                })?;
            let cufft_set_auto_allocation = *cufft_lib
                .get::<unsafe extern "C" fn(CufftHandle, i32) -> i32>(b"cufftSetAutoAllocation\0")
                .map_err(|e| CufftError::SymbolLoad {
                    sym: "cufftSetAutoAllocation".into(),
                    source: e,
                })?;
            let cufft_make_plan_3d = *cufft_lib
                .get::<unsafe extern "C" fn(CufftHandle, i32, i32, i32, i32, *mut usize) -> i32>(
                    b"cufftMakePlan3d\0",
                )
                .map_err(|e| CufftError::SymbolLoad {
                    sym: "cufftMakePlan3d".into(),
                    source: e,
                })?;
            let cufft_set_work_area = *cufft_lib
                .get::<unsafe extern "C" fn(CufftHandle, *mut c_void) -> i32>(b"cufftSetWorkArea\0")
                .map_err(|e| CufftError::SymbolLoad {
                    sym: "cufftSetWorkArea".into(),
                    source: e,
                })?;
            let cufft_destroy = *cufft_lib
                .get::<unsafe extern "C" fn(CufftHandle) -> i32>(b"cufftDestroy\0")
                .map_err(|e| CufftError::SymbolLoad {
                    sym: "cufftDestroy".into(),
                    source: e,
                })?;

            Ok(Self {
                _cudart: cudart,
                _cufft: cufft_lib,
                cuda_malloc,
                cuda_free,
                cuda_memcpy,
                cufft_plan_3d,
                cufft_create,
                cufft_set_auto_allocation,
                cufft_make_plan_3d,
                cufft_set_work_area,
                cufft_exec_d2z,
                cufft_exec_z2d,
                cufft_set_stream,
                cufft_get_size,
                cufft_destroy,
            })
        }
    }
}

/// GPU FFT backend using NVIDIA cuFFT.
///
/// Device buffers and cuFFT plans are allocated once at construction and
/// reused across all calls. The CUDA runtime and cuFFT libraries are loaded
/// at runtime via `libloading`, so no CUDA SDK is needed at compile time.
pub struct CufftBackend {
    libs: CudaLibs,
    nx: usize,
    ny: usize,
    nz: usize,
    /// Device pointer to real buffer (nx * ny * nz f64s).
    d_real: *mut c_void,
    /// Device pointer to complex buffer (nx * ny * (nz/2+1) Complex<f64>s).
    d_complex: *mut c_void,
    /// cuFFT plan for forward D2Z transform.
    plan_d2z: CufftHandle,
    /// cuFFT plan for inverse Z2D transform.
    plan_z2d: CufftHandle,
    /// Shared cuFFT work area for the device-only constructor (one buffer used by
    /// BOTH plans, valid because the resident solver runs them sequentially on a
    /// single stream). Null for the host-path constructor (cuFFT auto-allocates a
    /// separate work area per plan there). Freed on Drop when non-null.
    shared_work_area: *mut c_void,
    /// The shared work area's size in bytes, recorded from `cufftMakePlan3d`
    /// (`max` of the two plans' required sizes), for the memory budget. Zero for
    /// the host-path constructor.
    shared_work_bytes: usize,
}

// CufftBackend is Send because the device pointers and plans are only
// accessed through &self methods that perform synchronous CUDA calls.
unsafe impl Send for CufftBackend {}

impl CufftBackend {
    /// Create a new cuFFT backend for the given grid dimensions.
    ///
    /// Loads CUDA libraries, allocates device memory, and creates FFT plans.
    pub fn new(nx: usize, ny: usize, nz: usize) -> Result<Self, CufftError> {
        let libs = CudaLibs::load()?;

        let real_bytes = nx * ny * nz * std::mem::size_of::<f64>();
        let complex_elems = nx * ny * (nz / 2 + 1);
        let complex_bytes = complex_elems * std::mem::size_of::<Complex<f64>>();

        let mut d_real: *mut c_void = ptr::null_mut();
        let mut d_complex: *mut c_void = ptr::null_mut();

        unsafe {
            let rc = (libs.cuda_malloc)(&mut d_real, real_bytes);
            if rc != 0 {
                return Err(CufftError::Alloc { code: rc });
            }
            let rc = (libs.cuda_malloc)(&mut d_complex, complex_bytes);
            if rc != 0 {
                (libs.cuda_free)(d_real);
                return Err(CufftError::Alloc { code: rc });
            }
        }

        let mut plan_d2z: CufftHandle = 0;
        let mut plan_z2d: CufftHandle = 0;

        unsafe {
            let rc =
                (libs.cufft_plan_3d)(&mut plan_d2z, nx as i32, ny as i32, nz as i32, CUFFT_D2Z);
            if rc != 0 {
                (libs.cuda_free)(d_real);
                (libs.cuda_free)(d_complex);
                return Err(CufftError::PlanCreate { code: rc });
            }
            let rc =
                (libs.cufft_plan_3d)(&mut plan_z2d, nx as i32, ny as i32, nz as i32, CUFFT_Z2D);
            if rc != 0 {
                (libs.cufft_destroy)(plan_d2z);
                (libs.cuda_free)(d_real);
                (libs.cuda_free)(d_complex);
                return Err(CufftError::PlanCreate { code: rc });
            }
        }

        Ok(Self {
            libs,
            nx,
            ny,
            nz,
            d_real,
            d_complex,
            plan_d2z,
            plan_z2d,
            shared_work_area: ptr::null_mut(),
            shared_work_bytes: 0,
        })
    }

    /// Creates a DEVICE-ONLY cuFFT backend: the two plans share ONE work area and
    /// the host-path `d_real`/`d_complex` scratch is NOT allocated.
    ///
    /// The host [`FftBackend`] methods (`r2c_3d`/`c2r_3d`) require the per-plan
    /// `d_real`/`d_complex` scratch and PANIC on a device-only backend; only the
    /// resident `*_dev` entry points are valid here. This exists to shrink the
    /// resident solver's footprint: at the 3/2-padded 384^3 grid the host-path
    /// scratch alone is ~0.85 GiB and cuFFT's two auto-allocated work areas are
    /// another large chunk; dropping the scratch and sharing ONE work area between
    /// the two plans (safe because the resident solver runs them sequentially on a
    /// single stream) reclaims over 1 GiB, the difference between fitting 256^3 in
    /// 8 GB and overflowing it.
    ///
    /// Mechanics: `cufftCreate` both plans, `cufftSetAutoAllocation(plan, 0)`,
    /// `cufftMakePlan3d` (which returns each plan's work size), allocate ONE
    /// device buffer of the MAX of the two sizes, then `cufftSetWorkArea` both
    /// plans to it.
    ///
    /// # Errors
    ///
    /// Returns a [`CufftError`] if any cuFFT call or the work-area allocation
    /// fails.
    pub fn new_device_only(nx: usize, ny: usize, nz: usize) -> Result<Self, CufftError> {
        let libs = CudaLibs::load()?;

        let mut plan_d2z: CufftHandle = 0;
        let mut plan_z2d: CufftHandle = 0;
        let mut sz_d2z: usize = 0;
        let mut sz_z2d: usize = 0;
        let mut work: *mut c_void = ptr::null_mut();

        unsafe {
            // Create both plans with auto-allocation OFF so cuFFT does not
            // allocate a separate work area per plan.
            let rc = (libs.cufft_create)(&mut plan_d2z);
            if rc != 0 {
                return Err(CufftError::PlanCreate { code: rc });
            }
            let rc = (libs.cufft_create)(&mut plan_z2d);
            if rc != 0 {
                (libs.cufft_destroy)(plan_d2z);
                return Err(CufftError::PlanCreate { code: rc });
            }
            let rc = (libs.cufft_set_auto_allocation)(plan_d2z, 0);
            if rc != 0 {
                (libs.cufft_destroy)(plan_d2z);
                (libs.cufft_destroy)(plan_z2d);
                return Err(CufftError::PlanCreate { code: rc });
            }
            let rc = (libs.cufft_set_auto_allocation)(plan_z2d, 0);
            if rc != 0 {
                (libs.cufft_destroy)(plan_d2z);
                (libs.cufft_destroy)(plan_z2d);
                return Err(CufftError::PlanCreate { code: rc });
            }
            // Build the plans; each returns its required work-area size.
            let rc = (libs.cufft_make_plan_3d)(
                plan_d2z,
                nx as i32,
                ny as i32,
                nz as i32,
                CUFFT_D2Z,
                &mut sz_d2z,
            );
            if rc != 0 {
                (libs.cufft_destroy)(plan_d2z);
                (libs.cufft_destroy)(plan_z2d);
                return Err(CufftError::PlanCreate { code: rc });
            }
            let rc = (libs.cufft_make_plan_3d)(
                plan_z2d,
                nx as i32,
                ny as i32,
                nz as i32,
                CUFFT_Z2D,
                &mut sz_z2d,
            );
            if rc != 0 {
                (libs.cufft_destroy)(plan_d2z);
                (libs.cufft_destroy)(plan_z2d);
                return Err(CufftError::PlanCreate { code: rc });
            }
            // ONE work area of the max size, shared by both plans (they run
            // sequentially on the resident solver's single stream).
            let work_bytes = sz_d2z.max(sz_z2d);
            let rc = (libs.cuda_malloc)(&mut work, work_bytes);
            if rc != 0 {
                (libs.cufft_destroy)(plan_d2z);
                (libs.cufft_destroy)(plan_z2d);
                return Err(CufftError::Alloc { code: rc });
            }
            let rc = (libs.cufft_set_work_area)(plan_d2z, work);
            if rc != 0 {
                (libs.cuda_free)(work);
                (libs.cufft_destroy)(plan_d2z);
                (libs.cufft_destroy)(plan_z2d);
                return Err(CufftError::PlanCreate { code: rc });
            }
            let rc = (libs.cufft_set_work_area)(plan_z2d, work);
            if rc != 0 {
                (libs.cuda_free)(work);
                (libs.cufft_destroy)(plan_d2z);
                (libs.cufft_destroy)(plan_z2d);
                return Err(CufftError::PlanCreate { code: rc });
            }
        }

        Ok(Self {
            libs,
            nx,
            ny,
            nz,
            d_real: ptr::null_mut(),
            d_complex: ptr::null_mut(),
            plan_d2z,
            plan_z2d,
            shared_work_area: work,
            shared_work_bytes: sz_d2z.max(sz_z2d),
        })
    }

    /// Binds both cuFFT plans (forward D2Z and inverse Z2D) to `stream`.
    ///
    /// After this call every `r2c_3d_dev`/`c2r_3d_dev` execution is enqueued on
    /// `stream` rather than the implicit default stream, so cuFFT runs on the
    /// same stream as the resident compute backend's kernels (no cross-stream
    /// synchronisation needed between an FFT and the pointwise operators).
    ///
    /// The stream pointer must belong to the SAME CUDA primary context that
    /// owns the device buffers passed to the `*_dev` methods (in practice the
    /// `cuda-core` backend stream obtained from `CudaBackend::stream_raw`). A
    /// null pointer selects the default stream, which is valid and harmless.
    ///
    /// # Errors
    ///
    /// Returns [`CufftError::SetStream`] if cuFFT rejects the stream on either
    /// plan (for example a stream from a different context).
    // cuFFT only stores the handle here; it does not dereference the stream
    // until a later exec, and the stream-validity contract is documented above.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn set_stream(&self, stream: *mut c_void) -> Result<(), CufftError> {
        unsafe {
            let rc = (self.libs.cufft_set_stream)(self.plan_d2z, stream);
            if rc != 0 {
                return Err(CufftError::SetStream { code: rc });
            }
            let rc = (self.libs.cufft_set_stream)(self.plan_z2d, stream);
            if rc != 0 {
                return Err(CufftError::SetStream { code: rc });
            }
        }
        Ok(())
    }

    /// Grid dimensions this backend was planned for: `(nx, ny, nz)`.
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// cuFFT internal work-area size in bytes, summed over the forward (D2Z) and
    /// inverse (Z2D) plans this backend owns.
    ///
    /// `cufftPlan3d` auto-allocates a separate work area per plan IN ADDITION to
    /// the `d_real`/`d_complex` host-path scratch and any caller-owned resident
    /// buffers. `cufftGetSize` reports that per-plan size, so the resident
    /// memory budget can account for it instead of leaving it uncounted. The two
    /// plans are queried independently and the sizes added (both are live for the
    /// lifetime of the backend). Returns `(d2z_bytes, z2d_bytes)`.
    ///
    /// # Errors
    ///
    /// Returns [`CufftError::Exec`] if `cufftGetSize` fails on either plan.
    pub fn workarea_bytes(&self) -> Result<(usize, usize), CufftError> {
        let mut d2z: usize = 0;
        let mut z2d: usize = 0;
        unsafe {
            let rc = (self.libs.cufft_get_size)(self.plan_d2z, &mut d2z as *mut usize);
            if rc != 0 {
                return Err(CufftError::Exec { code: rc });
            }
            let rc = (self.libs.cufft_get_size)(self.plan_z2d, &mut z2d as *mut usize);
            if rc != 0 {
                return Err(CufftError::Exec { code: rc });
            }
        }
        Ok((d2z, z2d))
    }

    /// Whether this backend was built device-only (one shared work area, no
    /// host-path scratch), as opposed to the host-path [`Self::new`].
    pub fn is_device_only(&self) -> bool {
        !self.shared_work_area.is_null()
    }

    /// ACTUAL cuFFT work-area bytes this backend has allocated on the device.
    ///
    /// For a device-only backend ([`Self::new_device_only`]) this is the SINGLE
    /// shared buffer (`max` of the two plans' required sizes), the real cost. For
    /// the host-path backend, cuFFT auto-allocated one work area per plan, so the
    /// real cost is the SUM of the two `cufftGetSize` queries. Returns the actual
    /// allocated work-area bytes either way, for the resident memory budget.
    ///
    /// # Errors
    ///
    /// Returns [`CufftError::Exec`] if `cufftGetSize` fails on either plan.
    pub fn allocated_workarea_bytes(&self) -> Result<usize, CufftError> {
        if self.is_device_only() {
            // The real size we allocated, recorded from cufftMakePlan3d (a
            // post-hoc cufftGetSize can read 0 once the work area is set).
            Ok(self.shared_work_bytes)
        } else {
            let (d2z, z2d) = self.workarea_bytes()?;
            Ok(d2z + z2d)
        }
    }

    /// Forward real-to-complex FFT executed DIRECTLY on caller-owned device
    /// buffers, with no host transfer.
    ///
    /// Runs `cufftExecD2Z(plan_d2z, d_real, d_complex)` against the resident
    /// pointers; `self.d_real`/`self.d_complex` (the host-path scratch) are not
    /// touched. The result lands in `d_complex` in cuFFT's native interleaved
    /// `[re, im, ...]` layout, length `nx * ny * (nz/2 + 1)` complex elements.
    ///
    /// # Safety
    ///
    /// - `d_real` must point to a valid device allocation of at least
    ///   `nx * ny * nz` `f64`s, and `d_complex` to at least
    ///   `nx * ny * (nz/2 + 1)` interleaved complex `f64` pairs
    ///   (`2 * nx * ny * (nz/2 + 1)` `f64`s).
    /// - Both allocations must live in the active CUDA primary context (the one
    ///   the plan was created in and `set_stream`'s stream belongs to).
    /// - The buffers must not be freed or moved before the transform completes
    ///   (synchronise the stream before reuse).
    pub unsafe fn r2c_3d_dev(&self, d_real: *mut f64, d_complex: *mut f64) {
        unsafe {
            let rc = (self.libs.cufft_exec_d2z)(self.plan_d2z, d_real, d_complex as *mut [f64; 2]);
            assert_eq!(rc, 0, "cufftExecD2Z (device) failed: {rc}");
        }
    }

    /// Inverse complex-to-real FFT executed DIRECTLY on caller-owned device
    /// buffers, with no host transfer and NO normalisation.
    ///
    /// Runs `cufftExecZ2D(plan_z2d, d_complex, d_real)` against the resident
    /// pointers; `self.d_real`/`self.d_complex` are not touched. The result
    /// lands in `d_real`, length `nx * ny * nz` reals.
    ///
    /// CRITICAL: unlike the host [`FftBackend::c2r_3d`] path, this leaves the
    /// output UNNORMALISED. cuFFT's inverse transform is unscaled, so the caller
    /// MUST divide the result by `N = nx * ny * nz` itself. Responsibility for
    /// the `1/N` is handed back deliberately so the resident step can fuse it
    /// with other per-mode scaling (a later task supplies a device scale kernel;
    /// until then the round-trip test normalises after the single download).
    ///
    /// # Safety
    ///
    /// - `d_complex` must point to a valid device allocation of at least
    ///   `nx * ny * (nz/2 + 1)` interleaved complex `f64` pairs, and `d_real` to
    ///   at least `nx * ny * nz` `f64`s.
    /// - Both allocations must live in the active CUDA primary context.
    /// - The buffers must not be freed or moved before the transform completes
    ///   (synchronise the stream before reuse).
    pub unsafe fn c2r_3d_dev(&self, d_complex: *mut f64, d_real: *mut f64) {
        unsafe {
            let rc = (self.libs.cufft_exec_z2d)(self.plan_z2d, d_complex as *mut [f64; 2], d_real);
            assert_eq!(rc, 0, "cufftExecZ2D (device) failed: {rc}");
        }
    }
}

impl Drop for CufftBackend {
    fn drop(&mut self) {
        unsafe {
            (self.libs.cufft_destroy)(self.plan_d2z);
            (self.libs.cufft_destroy)(self.plan_z2d);
            // Host-path scratch (null on a device-only backend; cudaFree(null) is
            // a documented no-op, but guard anyway for clarity).
            if !self.d_real.is_null() {
                (self.libs.cuda_free)(self.d_real);
            }
            if !self.d_complex.is_null() {
                (self.libs.cuda_free)(self.d_complex);
            }
            // The single shared work area of a device-only backend.
            if !self.shared_work_area.is_null() {
                (self.libs.cuda_free)(self.shared_work_area);
            }
        }
    }
}

impl FftBackend<f64> for CufftBackend {
    fn r2c_3d(&self, input: &Array3<f64>, output: &mut Array3<Complex<f64>>) {
        let real_bytes = self.nx * self.ny * self.nz * std::mem::size_of::<f64>();
        let complex_elems = self.nx * self.ny * (self.nz / 2 + 1);
        let complex_bytes = complex_elems * std::mem::size_of::<Complex<f64>>();

        unsafe {
            // Host -> Device (real data)
            let rc = (self.libs.cuda_memcpy)(
                self.d_real,
                input.as_slice().unwrap().as_ptr() as *const c_void,
                real_bytes,
                CUDA_MEMCPY_H2D,
            );
            assert_eq!(rc, 0, "cudaMemcpy H2D failed: {rc}");

            // Execute forward D2Z
            let rc = (self.libs.cufft_exec_d2z)(
                self.plan_d2z,
                self.d_real as *mut f64,
                self.d_complex as *mut [f64; 2],
            );
            assert_eq!(rc, 0, "cufftExecD2Z failed: {rc}");

            // Device -> Host (complex data)
            let rc = (self.libs.cuda_memcpy)(
                output.as_slice_mut().unwrap().as_mut_ptr() as *mut c_void,
                self.d_complex as *const c_void,
                complex_bytes,
                CUDA_MEMCPY_D2H,
            );
            assert_eq!(rc, 0, "cudaMemcpy D2H failed: {rc}");
        }
    }

    fn c2r_3d(&self, input: &Array3<Complex<f64>>, output: &mut Array3<f64>) {
        let real_bytes = self.nx * self.ny * self.nz * std::mem::size_of::<f64>();
        let complex_elems = self.nx * self.ny * (self.nz / 2 + 1);
        let complex_bytes = complex_elems * std::mem::size_of::<Complex<f64>>();
        let n_total = (self.nx * self.ny * self.nz) as f64;

        unsafe {
            // Host -> Device (complex data)
            let rc = (self.libs.cuda_memcpy)(
                self.d_complex,
                input.as_slice().unwrap().as_ptr() as *const c_void,
                complex_bytes,
                CUDA_MEMCPY_H2D,
            );
            assert_eq!(rc, 0, "cudaMemcpy H2D failed: {rc}");

            // Execute inverse Z2D
            let rc = (self.libs.cufft_exec_z2d)(
                self.plan_z2d,
                self.d_complex as *mut [f64; 2],
                self.d_real as *mut f64,
            );
            assert_eq!(rc, 0, "cufftExecZ2D failed: {rc}");

            // Device -> Host (real data)
            let rc = (self.libs.cuda_memcpy)(
                output.as_slice_mut().unwrap().as_mut_ptr() as *mut c_void,
                self.d_real as *const c_void,
                real_bytes,
                CUDA_MEMCPY_D2H,
            );
            assert_eq!(rc, 0, "cudaMemcpy D2H failed: {rc}");
        }

        // cuFFT inverse is unnormalized; apply 1/N normalization
        output.mapv_inplace(|v| v / n_total);
    }

    fn name(&self) -> &str {
        "cuFFT (GPU)"
    }

    fn precision_digits(&self) -> usize {
        15
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // requires CUDA; run locally with: cargo test -- --ignored
    fn cufft_backend_constructs() {
        let backend = CufftBackend::new(8, 8, 8).expect("cuFFT should load on this machine");
        assert_eq!(backend.name(), "cuFFT (GPU)");
        assert_eq!(backend.precision_digits(), 15);
    }

    #[test]
    #[ignore]
    fn cufft_roundtrip() {
        let n = 16;
        let backend = CufftBackend::new(n, n, n).expect("cuFFT load failed");
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
                        "roundtrip mismatch at ({i},{j},{k}): got {}, expected {}",
                        output[[i, j, k]],
                        input[[i, j, k]]
                    );
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn cufft_matches_cpu() {
        use crate::NdrustfftBackend;

        let n = 16;
        let gpu = CufftBackend::new(n, n, n).expect("cuFFT load failed");
        let cpu = NdrustfftBackend::new(n, n, n);

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

        let mut spec_gpu = Array3::<Complex<f64>>::zeros((n, n, n / 2 + 1));
        let mut spec_cpu = Array3::<Complex<f64>>::zeros((n, n, n / 2 + 1));
        gpu.r2c_3d(&input, &mut spec_gpu);
        cpu.r2c_3d(&input, &mut spec_cpu);

        for ((g, c), idx) in spec_gpu.iter().zip(spec_cpu.iter()).zip(0..) {
            assert!(
                (g.re - c.re).abs() < 1e-10 && (g.im - c.im).abs() < 1e-10,
                "GPU/CPU mismatch at flat index {idx}: gpu={g:?}, cpu={c:?}"
            );
        }
    }
}
