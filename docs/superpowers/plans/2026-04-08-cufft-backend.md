# cuFFT GPU Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runtime-loaded cuFFT backend to vonkarman-fft that implements `FftBackend<f64>` with automatic fallback to the CPU backend when CUDA is unavailable.

**Architecture:** The `CufftBackend` loads `libcufft.so` and `libcudart.so` at runtime via `libloading`. Device buffers persist across calls. The solver uses `Box<dyn FftBackend<f64>>` instead of concrete `NdrustfftBackend`. A `create_backend(mode)` factory handles selection and fallback. CI tests the CPU path; GPU correctness is verified locally.

**Tech Stack:** Rust, libloading 0.8, CUDA Runtime API (cudaMalloc/cudaFree/cudaMemcpy), cuFFT (cufftPlan3d/cufftExecD2Z/cufftExecZ2D)

---

### Task 1: Add libloading dependency and CufftError type

**Files:**
- Modify: `vonkarman-fft/Cargo.toml`
- Create: `vonkarman-fft/src/cufft.rs`
- Modify: `vonkarman-fft/src/lib.rs`

- [ ] **Step 1: Add libloading to vonkarman-fft/Cargo.toml**

```toml
[dependencies]
vonkarman-core = { workspace = true }
ndarray = { workspace = true }
num-complex = { workspace = true }
rayon = { workspace = true }
ndrustfft = "0.5"
libloading = "0.8"
thiserror = "2"
```

- [ ] **Step 2: Create cufft.rs with error type and FFI constants**

Create `vonkarman-fft/src/cufft.rs`:

```rust
//! cuFFT backend via runtime-loaded shared libraries.
//!
//! Loads `libcufft.so` and `libcudart.so` at runtime using `libloading`.
//! No compile-time CUDA SDK dependency. Falls back gracefully when the
//! libraries are not present.

use std::ffi::c_void;
use std::ptr;

use libloading::{Library, Symbol};
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
```

- [ ] **Step 3: Add the module to lib.rs**

Edit `vonkarman-fft/src/lib.rs`:

```rust
pub mod backend;
pub mod cufft;
pub mod dealiased;
pub mod ndrustfft_backend;

pub use backend::FftBackend;
pub use cufft::CufftError;
pub use dealiased::dealiased_cross_product;
pub use ndrustfft_backend::NdrustfftBackend;
```

- [ ] **Step 4: Verify it compiles**

Run: `cd ~/vonkarman && cargo check -p vonkarman-fft`
Expected: clean compile, no errors.

- [ ] **Step 5: Commit**

```bash
git add vonkarman-fft/Cargo.toml vonkarman-fft/src/cufft.rs vonkarman-fft/src/lib.rs
git commit -m "feat(fft): add CufftError type and libloading dependency"
```

---

### Task 2: Implement CufftBackend with device memory and FFT plans

**Files:**
- Modify: `vonkarman-fft/src/cufft.rs`

- [ ] **Step 1: Write failing test for CufftBackend construction**

Append to `vonkarman-fft/src/cufft.rs`:

```rust
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
}
```

- [ ] **Step 2: Verify test fails**

Run: `cd ~/vonkarman && cargo test -p vonkarman-fft cufft_backend_constructs -- --ignored 2>&1 | tail -5`
Expected: compile error (CufftBackend does not exist).

- [ ] **Step 3: Implement CudaLibs and CufftBackend**

Add to `vonkarman-fft/src/cufft.rs` (above the tests module):

```rust
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
            let rc = (libs.cufft_plan_3d)(
                &mut plan_d2z,
                nx as i32,
                ny as i32,
                nz as i32,
                CUFFT_D2Z,
            );
            if rc != 0 {
                (libs.cuda_free)(d_real);
                (libs.cuda_free)(d_complex);
                return Err(CufftError::PlanCreate { code: rc });
            }
            let rc = (libs.cufft_plan_3d)(
                &mut plan_z2d,
                nx as i32,
                ny as i32,
                nz as i32,
                CUFFT_Z2D,
            );
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
```

- [ ] **Step 4: Add more GPU tests**

Append to the `#[cfg(test)] mod tests` block:

```rust
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
```

- [ ] **Step 5: Verify compile and run GPU tests locally**

Run: `cd ~/vonkarman && cargo check -p vonkarman-fft`
Expected: clean compile.

Run locally (GPU): `cd ~/vonkarman && cargo test -p vonkarman-fft -- --ignored`
Expected: all 3 ignored tests pass.

- [ ] **Step 6: Commit**

```bash
git add vonkarman-fft/src/cufft.rs
git commit -m "feat(fft): implement CufftBackend with device memory and R2C/C2R transforms"
```

---

### Task 3: Backend selection factory

**Files:**
- Create: `vonkarman-fft/src/select.rs`
- Modify: `vonkarman-fft/src/lib.rs`

- [ ] **Step 1: Write failing test for backend selection**

Create `vonkarman-fft/src/select.rs`:

```rust
//! Runtime FFT backend selection with automatic fallback.

use crate::backend::FftBackend;
use crate::cufft::CufftBackend;
use crate::ndrustfft_backend::NdrustfftBackend;

/// Backend selection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendMode {
    /// Try cuFFT first, fall back to CPU if CUDA unavailable.
    Auto,
    /// Force cuFFT (fails if CUDA unavailable).
    Cufft,
    /// Force CPU backend (ndrustfft).
    Cpu,
}

impl BackendMode {
    /// Parse from a string (for CLI and TOML config).
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cufft" | "gpu" | "cuda" => Self::Cufft,
            "cpu" | "ndrustfft" => Self::Cpu,
            _ => Self::Auto,
        }
    }
}

/// Create an FFT backend for the given grid dimensions.
///
/// In `Auto` mode, tries cuFFT first and silently falls back to CPU.
/// In `Cufft` mode, panics if CUDA is unavailable.
/// In `Cpu` mode, always uses ndrustfft.
pub fn create_backend(
    nx: usize,
    ny: usize,
    nz: usize,
    mode: BackendMode,
) -> Box<dyn FftBackend<f64>> {
    match mode {
        BackendMode::Cufft => {
            let backend = CufftBackend::new(nx, ny, nz)
                .expect("cuFFT requested but CUDA is unavailable");
            tracing::info!(backend = "cuFFT", nx, ny, nz, "FFT backend selected");
            Box::new(backend)
        }
        BackendMode::Cpu => {
            tracing::info!(backend = "ndrustfft", nx, ny, nz, "FFT backend selected");
            Box::new(NdrustfftBackend::new(nx, ny, nz))
        }
        BackendMode::Auto => match CufftBackend::new(nx, ny, nz) {
            Ok(backend) => {
                tracing::info!(backend = "cuFFT", nx, ny, nz, "FFT backend selected (auto)");
                Box::new(backend)
            }
            Err(e) => {
                tracing::info!(
                    backend = "ndrustfft",
                    reason = %e,
                    nx, ny, nz,
                    "cuFFT unavailable, falling back to CPU"
                );
                Box::new(NdrustfftBackend::new(nx, ny, nz))
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_mode_returns_a_backend() {
        // On CI (no GPU), this returns NdrustfftBackend.
        // Locally with CUDA, this returns CufftBackend.
        // Either way, it must not panic.
        let backend = create_backend(8, 8, 8, BackendMode::Auto);
        assert!(!backend.name().is_empty());
    }

    #[test]
    fn cpu_mode_always_returns_cpu() {
        let backend = create_backend(8, 8, 8, BackendMode::Cpu);
        assert_eq!(backend.name(), "ndrustfft (pure Rust CPU)");
    }

    #[test]
    fn parse_backend_mode() {
        assert_eq!(BackendMode::from_str_loose("cufft"), BackendMode::Cufft);
        assert_eq!(BackendMode::from_str_loose("GPU"), BackendMode::Cufft);
        assert_eq!(BackendMode::from_str_loose("cpu"), BackendMode::Cpu);
        assert_eq!(BackendMode::from_str_loose("auto"), BackendMode::Auto);
        assert_eq!(BackendMode::from_str_loose("anything"), BackendMode::Auto);
    }
}
```

- [ ] **Step 2: Update lib.rs exports**

Edit `vonkarman-fft/src/lib.rs`:

```rust
pub mod backend;
pub mod cufft;
pub mod dealiased;
pub mod ndrustfft_backend;
pub mod select;

pub use backend::FftBackend;
pub use cufft::{CufftBackend, CufftError};
pub use dealiased::dealiased_cross_product;
pub use ndrustfft_backend::NdrustfftBackend;
pub use select::{BackendMode, create_backend};
```

- [ ] **Step 3: Run tests**

Run: `cd ~/vonkarman && cargo test -p vonkarman-fft`
Expected: all non-ignored tests pass (including the 3 new select tests).

- [ ] **Step 4: Commit**

```bash
git add vonkarman-fft/src/select.rs vonkarman-fft/src/lib.rs
git commit -m "feat(fft): add BackendMode and create_backend factory with auto-fallback"
```

---

### Task 4: Refactor Periodic3D to use `Box<dyn FftBackend<f64>>`

**Files:**
- Modify: `vonkarman-periodic/src/solver.rs`
- Modify: `vonkarman-periodic/src/lib.rs` (if needed for re-exports)

- [ ] **Step 1: Change the fft fields from concrete to trait object**

In `vonkarman-periodic/src/solver.rs`, replace:

```rust
use vonkarman_fft::{FftBackend, NdrustfftBackend};
```

with:

```rust
use vonkarman_fft::{FftBackend, BackendMode, create_backend};
```

Replace the struct fields:

```rust
    /// FFT backend (original grid).
    fft: NdrustfftBackend,
    /// FFT backend (3/2-padded grid).
    fft_padded: NdrustfftBackend,
```

with:

```rust
    /// FFT backend (original grid).
    fft: Box<dyn FftBackend<f64>>,
    /// FFT backend (3/2-padded grid).
    fft_padded: Box<dyn FftBackend<f64>>,
```

- [ ] **Step 2: Update Periodic3D::new to accept BackendMode**

Change the constructor signature from:

```rust
pub fn new(grid: GridSpec, nu: f64, ic: IcType) -> Self {
```

to:

```rust
pub fn new(grid: GridSpec, nu: f64, ic: IcType, backend_mode: BackendMode) -> Self {
```

Replace the backend creation lines:

```rust
        let fft = NdrustfftBackend::new(grid.nx, grid.ny, grid.nz);
```

with:

```rust
        let fft = create_backend(grid.nx, grid.ny, grid.nz, backend_mode);
```

And:

```rust
        let fft_padded = NdrustfftBackend::new(pg.nx, pg.ny, pg.nz);
```

with:

```rust
        let fft_padded = create_backend(pg.nx, pg.ny, pg.nz, backend_mode);
```

- [ ] **Step 3: Update all fft references from value to trait object**

Wherever the code calls `self.fft.r2c_3d(...)` or `self.fft.c2r_3d(...)`, the method signatures take `&self` on the trait, so these calls should work unchanged with `Box<dyn FftBackend<f64>>`.

For code that passes `&self.fft` as `&dyn FftBackend<f64>`, change to `self.fft.as_ref()` or `&*self.fft`.

Search for all occurrences of `&self.fft` and `&self.fft_padded` in solver.rs and update:

```rust
// Before:
&self.fft
// After:
self.fft.as_ref()

// Before:
&self.fft_padded
// After:
self.fft_padded.as_ref()
```

- [ ] **Step 4: Fix test code in solver.rs and other files**

Any test that calls `Periodic3D::new(grid, nu, ic)` needs the extra argument:

```rust
// Before:
let mut solver = Periodic3D::new(grid, nu, ic);
// After:
let mut solver = Periodic3D::new(grid, nu, ic, BackendMode::Cpu);
```

Search the workspace for all `Periodic3D::new(` calls and add `BackendMode::Cpu` as the fourth argument. Files to check:
- `vonkarman-periodic/src/solver.rs` (tests)
- `vonkarman-periodic/src/rk4.rs` (tests)
- `vonkarman-periodic/src/nonlinear.rs` (tests)
- `vonkarman-periodic/src/ic/*.rs` (tests)
- `vonkarman-bin/src/run.rs`
- `vonkarman-diag/src/*.rs` (if any)
- Integration tests in `tests/` directories

For `vonkarman-bin/src/run.rs`, use `BackendMode::Auto` (default for CLI):

```rust
let mut solver = Periodic3D::new(grid, nu, ic, BackendMode::Auto);
```

- [ ] **Step 5: Verify all tests pass**

Run: `cd ~/vonkarman && cargo test --workspace`
Expected: all tests pass with the CPU backend.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(periodic): use Box<dyn FftBackend> with BackendMode selection"
```

---

### Task 5: CLI --backend flag and TOML config

**Files:**
- Modify: `vonkarman-bin/src/config.rs`
- Modify: `vonkarman-bin/src/run.rs`
- Modify: `vonkarman-bin/src/main.rs`

- [ ] **Step 1: Add backend field to config**

In `vonkarman-bin/src/config.rs`, add to `DomainConfig`:

```rust
#[derive(Debug, Deserialize)]
pub struct DomainConfig {
    #[serde(rename = "type")]
    pub domain_type: String,
    pub n: usize,
    #[serde(default = "default_domain_length")]
    pub l: f64,
    /// FFT backend: "auto" (default), "cufft", or "cpu".
    #[serde(default = "default_backend")]
    pub backend: String,
}

fn default_backend() -> String {
    "auto".to_string()
}
```

- [ ] **Step 2: Update run.rs to use the backend config**

In `vonkarman-bin/src/run.rs`, add the import:

```rust
use vonkarman_fft::BackendMode;
```

Change the solver construction to read the backend from config:

```rust
    let backend_mode = BackendMode::from_str_loose(&config.domain.backend);
    let mut solver = Periodic3D::new(grid, nu, ic, backend_mode);
```

- [ ] **Step 3: Add --backend CLI override**

In `vonkarman-bin/src/main.rs`, update the Run variant:

```rust
    Run {
        #[arg(short, long)]
        config: String,
        /// FFT backend: auto, cufft, cpu. Overrides TOML config.
        #[arg(long, default_value = "")]
        backend: String,
    },
```

In the match arm, pass the override to `run`:

```rust
        Commands::Run {
            config: config_path,
            backend,
        } => {
            let contents = std::fs::read_to_string(&config_path)?;
            let mut config: config::ExperimentConfig = toml::from_str(&contents)?;
            if !backend.is_empty() {
                config.domain.backend = backend;
            }
            run::run(&config)?;
        }
```

- [ ] **Step 4: Update config test**

Add a test in `vonkarman-bin/src/config.rs` that verifies backend parsing:

```rust
    #[test]
    fn parse_config_with_backend() {
        let toml_str = r#"
[run]
name = "test"
output_dir = "./output/test"

[domain]
type = "periodic3d"
n = 64
backend = "cufft"

[physics]
nu = 1e-3

[initial_condition]
type = "taylor-green"

[termination]
max_steps = 10
"#;
        let cfg: ExperimentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.domain.backend, "cufft");
    }

    #[test]
    fn parse_config_default_backend() {
        let toml_str = r#"
[run]
name = "test"
output_dir = "./output/test"

[domain]
type = "periodic3d"
n = 64

[physics]
nu = 1e-3

[initial_condition]
type = "taylor-green"

[termination]
max_steps = 10
"#;
        let cfg: ExperimentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.domain.backend, "auto");
    }
```

- [ ] **Step 5: Run full test suite**

Run: `cd ~/vonkarman && cargo test --workspace`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(bin): add --backend CLI flag and TOML backend config"
```

---

### Task 6: CI verification and push

**Files:**
- Modify: `.github/workflows/ci.yml` (if needed)

- [ ] **Step 1: Run clippy**

Run: `cd ~/vonkarman && cargo clippy --workspace -- -D warnings`
Expected: zero warnings.

- [ ] **Step 2: Run full test suite one more time**

Run: `cd ~/vonkarman && cargo test --workspace`
Expected: all tests pass.

- [ ] **Step 3: Run GPU tests locally**

Run: `cd ~/vonkarman && cargo test -p vonkarman-fft -- --ignored`
Expected: cufft_backend_constructs, cufft_roundtrip, cufft_matches_cpu all pass.

- [ ] **Step 4: Push and verify CI**

```bash
git push origin main
```

Check CI: `gh run list --workflow ci.yml --limit 1`
Expected: CI passes (all non-ignored tests on CPU backend).

- [ ] **Step 5: Tag release**

```bash
git tag v0.2.0
git push origin v0.2.0
```
