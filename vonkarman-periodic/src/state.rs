//! Device-resident spectral state container.
//!
//! [`SpectralState`] holds the three spectral velocity components `u_hat` plus a
//! minimal, reusable scratch set, all in backend-owned buffers. It is generic
//! over the [`ComputeBackend`] and the [`Float`] precision, so the same
//! container serves the CPU path (host `Vec`s) and the GPU path (device buffers)
//! without the caller touching raw storage.
//!
//! # Residency
//!
//! On the GPU backend the velocity and scratch buffers live in device memory for
//! the whole life of the state. The resident FFT round trip
//! ([`SpectralState::resident_fft_roundtrip`], `cuda` feature) runs cuFFT
//! DIRECTLY on those device buffers, so a forward then inverse transform moves
//! nothing across the PCIe bus. Setup ([`SpectralState::upload`]) and diagnostics
//! ([`SpectralState::download`]) are the only host<->device transfer points.
//!
//! # Scratch discipline (the 3/2 budget)
//!
//! The scratch set is deliberately MINIMAL: one real buffer and one complex
//! buffer, shared and reused across all three components (process one component
//! at a time). This matters for the eventual resident nonlinear term: at `N =
//! 256` the 3/2-dealiasing grid is `384^3`, about 0.45 GB per real field, so a
//! single 8 GB GPU cannot afford a padded scratch buffer per component. The
//! container therefore never bakes in per-component padded allocations; later
//! tasks size and reuse the shared scratch for the padded transforms.

use num_complex::Complex;
use vonkarman_core::{ComputeBackend, float::Float};

/// Grid dimensions a [`SpectralState`] was built for.
///
/// `nx`/`ny`/`nz` are the physical (real-space) extents on the `N` grid; the
/// spectral half-complex extent in the last axis is `nz / 2 + 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateDims {
    /// Physical extent in x.
    pub nx: usize,
    /// Physical extent in y.
    pub ny: usize,
    /// Physical extent in z.
    pub nz: usize,
}

impl StateDims {
    /// Builds dimensions for a cubic `n^3` grid.
    pub fn cubic(n: usize) -> Self {
        Self {
            nx: n,
            ny: n,
            nz: n,
        }
    }

    /// Number of real points, `nx * ny * nz`.
    pub fn real_len(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Number of half-complex spectral points, `nx * ny * (nz / 2 + 1)`.
    ///
    /// This is the r2c output extent in the last axis (`nz / 2 + 1`), which is
    /// the layout both the CPU [`vonkarman_fft::NdrustfftBackend`] and the GPU
    /// cuFFT D2Z transform produce.
    pub fn cplx_len(&self) -> usize {
        self.nx * self.ny * (self.nz / 2 + 1)
    }
}

/// Device-resident (or host-resident, on the CPU backend) spectral solver state.
///
/// Holds the three spectral velocity components and a shared scratch set in
/// buffers owned by `backend`. Generic over:
///
/// - `B: ComputeBackend` - where the buffers live (CPU `Vec`s or GPU device
///   memory).
/// - `F: Float` - the scalar precision (`f64` today; the trait is multi-precision
///   ready).
///
/// The state borrows nothing; it owns its buffers. Construct with [`Self::new`].
pub struct SpectralState<B: ComputeBackend, F: Float> {
    /// Grid dimensions.
    dims: StateDims,
    /// The three spectral velocity components, each `cplx_len()` complex points.
    u_hat: [B::CplxBuf<F>; 3],
    /// Shared real scratch, `real_len()` reals. Reused across components for one
    /// FFT round trip (the inverse-transform target).
    real_scratch: B::RealBuf<F>,
    /// Shared complex scratch, `cplx_len()` complex points. Reused across
    /// components (the forward-transform target / general working buffer).
    cplx_scratch: B::CplxBuf<F>,
}

impl<B: ComputeBackend, F: Float> SpectralState<B, F> {
    /// Allocates a fresh, zeroed spectral state on `backend` for `dims`.
    ///
    /// Allocates three spectral velocity buffers plus one real and one complex
    /// scratch buffer. All buffers are zero-initialised by the backend's alloc
    /// methods.
    pub fn new(backend: &B, dims: StateDims) -> Self {
        let cplx_len = dims.cplx_len();
        let real_len = dims.real_len();
        let u_hat = [
            backend.alloc_cplx::<F>(cplx_len),
            backend.alloc_cplx::<F>(cplx_len),
            backend.alloc_cplx::<F>(cplx_len),
        ];
        let real_scratch = backend.alloc_real::<F>(real_len);
        let cplx_scratch = backend.alloc_cplx::<F>(cplx_len);
        Self {
            dims,
            u_hat,
            real_scratch,
            cplx_scratch,
        }
    }

    /// Grid dimensions this state was built for.
    pub fn dims(&self) -> StateDims {
        self.dims
    }

    /// Borrows the three spectral velocity components.
    pub fn u_hat(&self) -> &[B::CplxBuf<F>; 3] {
        &self.u_hat
    }

    /// Mutably borrows the three spectral velocity components.
    pub fn u_hat_mut(&mut self) -> &mut [B::CplxBuf<F>; 3] {
        &mut self.u_hat
    }

    /// Borrows the shared real scratch buffer (`real_len()` reals).
    ///
    /// One buffer, reused across components and across transform stages, so the
    /// resident path never allocates a padded scratch per component (see the
    /// 3/2 budget note at the module level).
    pub fn real_scratch(&self) -> &B::RealBuf<F> {
        &self.real_scratch
    }

    /// Mutably borrows the shared real scratch buffer.
    pub fn real_scratch_mut(&mut self) -> &mut B::RealBuf<F> {
        &mut self.real_scratch
    }

    /// Borrows the shared complex scratch buffer (`cplx_len()` complex points).
    pub fn cplx_scratch(&self) -> &B::CplxBuf<F> {
        &self.cplx_scratch
    }

    /// Mutably borrows the shared complex scratch buffer.
    pub fn cplx_scratch_mut(&mut self) -> &mut B::CplxBuf<F> {
        &mut self.cplx_scratch
    }

    /// Uploads a host spectral field into velocity component `c` (setup).
    ///
    /// `host` must hold `dims.cplx_len()` complex values in the backend's flat
    /// spectral order. Delegates to [`ComputeBackend::upload_cplx`], the single
    /// host->device transfer point.
    pub fn upload(&mut self, backend: &B, c: usize, host: &[Complex<F>]) {
        debug_assert!(c < 3, "velocity component index out of range");
        backend.upload_cplx::<F>(host, &mut self.u_hat[c]);
    }

    /// Downloads velocity component `c` into a host buffer (diagnostics).
    ///
    /// `host` must hold `dims.cplx_len()` complex values. Delegates to
    /// [`ComputeBackend::download_cplx`], the single device->host transfer point.
    pub fn download(&self, backend: &B, c: usize, host: &mut [Complex<F>]) {
        debug_assert!(c < 3, "velocity component index out of range");
        backend.download_cplx::<F>(&self.u_hat[c], host);
    }
}

#[cfg(feature = "cuda")]
mod cuda_resident {
    //! GPU-resident FFT round trip for [`SpectralState`].
    //!
    //! Implemented only for the CUDA backend at `f64`. It runs cuFFT directly on
    //! the resident device buffers via the device-pointer entry points
    //! ([`vonkarman_fft::CufftBackend::r2c_3d_dev`] /
    //! [`vonkarman_fft::CufftBackend::c2r_3d_dev`]), so a forward then inverse
    //! transform performs zero host<->device copies.
    //!
    //! The CPU [`SpectralState`] stays a pure container (no resident round trip):
    //! the working CPU physics path already goes through `Periodic3D` with the
    //! `FftBackend` host transforms, so a second CPU FFT abstraction here would
    //! be redundant. Scoping the resident round trip to the CUDA path keeps this
    //! task focused on proving device residency.

    use super::SpectralState;
    use vonkarman_compute::CudaBackend;
    use vonkarman_fft::CufftBackend;

    impl SpectralState<CudaBackend, f64> {
        /// Resident forward+inverse FFT round trip on velocity component `c`,
        /// with ZERO host<->device copies.
        ///
        /// Runs, all on resident device buffers in the backend's primary context:
        ///
        /// 1. inverse-style staging: copies `u_hat[c]` into the complex scratch
        ///    is NOT needed; instead the forward transform reads the real scratch
        ///    and writes the complex scratch, and the inverse reads the complex
        ///    scratch and writes the real scratch. The round trip here takes the
        ///    resident real field already in `real_scratch`, transforms it to the
        ///    complex scratch (D2Z), then transforms back (Z2D) into the real
        ///    scratch, leaving the inverse UNNORMALISED.
        ///
        /// The result in the real scratch is the original real field scaled by
        /// `N = nx * ny * nz` (cuFFT's inverse is unscaled). The caller applies
        /// the `1/N` (a later task supplies a device scale kernel; until then the
        /// residency test normalises after its single download). The `c`
        /// argument is accepted for symmetry with the container API but the round
        /// trip operates on the shared scratch; see the residency test for the
        /// canonical usage that fills `real_scratch` first.
        ///
        /// `fft` must have been planned for the same `(nx, ny, nz)` as this state
        /// and bound to the backend stream with
        /// [`CufftBackend::set_stream`].
        ///
        /// # Panics
        ///
        /// Panics if `fft`'s planned dimensions do not match this state's.
        pub fn resident_fft_roundtrip(&mut self, backend: &CudaBackend, fft: &CufftBackend) {
            assert_eq!(
                fft.dims(),
                (self.dims.nx, self.dims.ny, self.dims.nz),
                "cuFFT plan dimensions must match the spectral state grid"
            );

            // Raw device pointers into the resident scratch buffers. No host
            // transfer: these address device memory directly.
            let d_real = backend.real_device_ptr(&self.real_scratch);
            let d_cplx = backend.cplx_device_ptr(&self.cplx_scratch);

            // SAFETY: both pointers come from this backend's own resident buffers
            // in the active primary context; `real_scratch` is `real_len()`
            // f64s and `cplx_scratch` is `2 * cplx_len()` f64s, the sizes the
            // cuFFT plan (matched above) expects. The plans are bound to this
            // backend's stream, and the buffers outlive the calls (we hold &mut
            // self). The forward writes the complex scratch; the inverse reads it
            // back into the real scratch, UNNORMALISED.
            unsafe {
                fft.r2c_3d_dev(d_real, d_cplx);
                fft.c2r_3d_dev(d_cplx, d_real);
            }
        }

        /// Uploads a host real field into the resident real scratch, for the
        /// resident round trip to consume (counts as one transfer).
        ///
        /// `host` must hold `dims.real_len()` reals. This is a setup transfer; the
        /// residency proof brackets it.
        pub fn upload_real_scratch(&mut self, backend: &CudaBackend, host: &[f64]) {
            // `B::RealBuf<f64>` is `RealBuf<f64>` for the CUDA backend, so the
            // uploaded buffer assigns straight into the resident scratch slot.
            self.real_scratch = backend.upload_real::<f64>(host);
        }

        /// Downloads the resident real scratch into a host `Vec` (counts as one
        /// transfer), for the residency test to verify the round trip.
        pub fn download_real_scratch(&self, backend: &CudaBackend) -> Vec<f64> {
            backend.download_real::<f64>(&self.real_scratch)
        }
    }
}
