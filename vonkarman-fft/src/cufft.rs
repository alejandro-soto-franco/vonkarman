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
    cufft_exec_d2z: unsafe extern "C" fn(CufftHandle, *mut f64, *mut [f64; 2]) -> i32,
    cufft_exec_z2d: unsafe extern "C" fn(CufftHandle, *mut [f64; 2], *mut f64) -> i32,
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
                cufft_exec_d2z,
                cufft_exec_z2d,
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
        })
    }
}

impl Drop for CufftBackend {
    fn drop(&mut self) {
        unsafe {
            (self.libs.cufft_destroy)(self.plan_d2z);
            (self.libs.cufft_destroy)(self.plan_z2d);
            (self.libs.cuda_free)(self.d_real);
            (self.libs.cuda_free)(self.d_complex);
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
