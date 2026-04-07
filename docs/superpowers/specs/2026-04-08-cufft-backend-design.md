# cuFFT GPU Backend for vonkarman-fft

## Goal

Add a `CufftBackend` that implements `FftBackend<f64>` via runtime-loaded cuFFT, with automatic fallback to the existing `NdrustfftBackend` when CUDA is unavailable. Hybrid approach: GPU for FFT, CPU for spectral operators. No compile-time CUDA dependency.

## Architecture

### New files

| File | Purpose |
|------|---------|
| `vonkarman-fft/src/cufft.rs` | cuFFT FFI bindings, device memory, `CufftBackend` |
| `vonkarman-fft/src/select.rs` | `BackendMode` enum, `create_backend` factory |

### Modified files

| File | Change |
|------|--------|
| `vonkarman-fft/src/lib.rs` | Add modules, re-export `BackendMode`, `create_backend` |
| `vonkarman-fft/Cargo.toml` | Add `libloading = "0.8"` |
| `vonkarman-bin/src/main.rs` | Add `--backend` CLI flag |
| `vonkarman-bin/src/config.rs` | Add `backend` field to TOML config |
| `vonkarman-bin/src/run.rs` | Use `create_backend` with config |

## FFI surface

Six symbols loaded via `libloading` from `libcufft.so` and `libcudart.so`:

```
cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32
cudaFree(devPtr: *mut c_void) -> i32
cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32
cufftPlan3d(plan: *mut cufftHandle, nx: i32, ny: i32, nz: i32, type: i32) -> i32
cufftExecD2Z(plan: cufftHandle, idata: *mut f64, odata: *mut cufftDoubleComplex) -> i32
cufftExecZ2D(plan: cufftHandle, idata: *mut cufftDoubleComplex, odata: *mut f64) -> i32
cufftDestroy(plan: cufftHandle) -> i32
```

Note: the trait uses R2C/C2R (real-to-complex and inverse). cuFFT provides `cufftExecD2Z` (double-to-Zcomplex, forward) and `cufftExecZ2D` (Zcomplex-to-double, inverse). These are the correct entry points, NOT `cufftExecZ2Z`.

cuFFT type constants:
- `CUFFT_D2Z = 0x6a` (forward R2C double)
- `CUFFT_Z2D = 0x6c` (inverse C2R double)

cudaMemcpy kinds:
- `cudaMemcpyHostToDevice = 1`
- `cudaMemcpyDeviceToHost = 2`

## Data flow per FFT call

### r2c_3d (forward)

1. Copy `input` (nx * ny * nz f64s) from host Array3 to device real buffer (H2D)
2. `cufftExecD2Z` on device (in-place: real buffer -> complex buffer)
3. Copy complex buffer (nx * ny * (nz/2+1) Complex<f64>) from device to host Array3 (D2H)

### c2r_3d (inverse)

1. Copy `input` (nx * ny * (nz/2+1) Complex<f64>s) from host to device complex buffer (H2D)
2. `cufftExecZ2D` on device (in-place: complex buffer -> real buffer)
3. Copy real buffer from device to host Array3 (D2H)
4. Normalize by 1/(nx*ny*nz) on the host (cuFFT inverse is unnormalized)

## Device memory

Two persistent buffers allocated at `CufftBackend::new`:
- Real buffer: `nx * ny * nz * sizeof(f64)` bytes
- Complex buffer: `nx * ny * (nz/2+1) * 2 * sizeof(f64)` bytes

For 128^3: ~16 MB real + ~17 MB complex = ~33 MB. For 384^3: ~432 MB + ~434 MB = ~866 MB. Well within 8 GB VRAM.

Two cuFFT plans (forward D2Z, inverse Z2D) created at init, reused.

For the dealiased cross product, the caller creates a second `CufftBackend` for the padded (3N/2)^3 grid, same as with `NdrustfftBackend`. This means two sets of device buffers and plans.

## Backend selection

```rust
pub enum BackendMode {
    Auto,
    Cufft,
    Cpu,
}

pub fn create_backend(nx: usize, ny: usize, nz: usize, mode: BackendMode) -> Box<dyn FftBackend<f64>> { ... }
```

`Auto` tries cuFFT first, falls back to CPU. `Cufft` fails hard if CUDA unavailable. `Cpu` always uses ndrustfft.

## Error handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum CufftError {
    #[error("failed to load {lib}: {source}")]
    LibraryLoad { lib: String, source: libloading::Error },
    #[error("cudaMalloc failed with code {code}")]
    Alloc { code: i32 },
    #[error("cudaMemcpy failed with code {code}")]
    Memcpy { code: i32 },
    #[error("cufftPlan3d failed with code {code}")]
    PlanCreate { code: i32 },
    #[error("cufftExec failed with code {code}")]
    Exec { code: i32 },
}
```

## Testing

All existing tests (roundtrip, Parseval, dealiased cross product) are backend-agnostic and run on whichever backend `create_backend(Auto)` returns. On CI (no GPU), this is NdrustfftBackend. Locally, it is CufftBackend.

New tests:
- `backend_fallback_without_cuda`: force `Auto` mode, verify a backend is returned (passes on CI)
- `cufft_roundtrip` (`#[ignore]`): explicit cuFFT roundtrip, run locally with `cargo test -- --ignored`
- `cufft_matches_cpu` (`#[ignore]`): compare GPU and CPU output element-wise, tolerance 1e-12

## CLI integration

```toml
# experiment config
[solver]
backend = "auto"  # or "cufft" or "cpu"
```

```
vonkarman run --config experiment.toml --backend cufft
```

CLI `--backend` flag overrides the TOML config if both are present.
