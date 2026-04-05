use ndarray::Array3;
use num_complex::Complex;
use vonkarman_core::float::Float;

/// FFT backend trait.
///
/// Implementations: NdrustfftBackend (pure Rust CPU, Phase 1).
/// Future: CufftBackend, VkfftBackend, FftwBackend.
pub trait FftBackend<F: Float>: Send {
    /// Forward 3D real-to-complex transform.
    /// Input shape: (nx, ny, nz). Output shape: (nx, ny, nz/2+1).
    /// The output is NOT normalized (unnormalized forward, like scipy default).
    fn r2c_3d(&self, input: &Array3<F>, output: &mut Array3<Complex<F>>);

    /// Inverse 3D complex-to-real transform.
    /// Input shape: (nx, ny, nz/2+1). Output shape: (nx, ny, nz).
    /// Includes the 1/N^3 normalization (scipy default).
    fn c2r_3d(&self, input: &Array3<Complex<F>>, output: &mut Array3<F>);

    /// Backend name for logging.
    fn name(&self) -> &str;

    /// Number of reliable digits at this precision.
    fn precision_digits(&self) -> usize;
}
