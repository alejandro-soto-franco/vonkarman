use crate::float::Float;
use num_complex::Complex;

/// Selects where pointwise spectral work runs.
///
/// The CPU implementation owns host `Vec`s. A future GPU implementation will
/// own device buffers. Buffer types are associated types so callers stay
/// backend-agnostic and never touch raw pointers.
///
/// # Implementation notes
///
/// - All methods take `&self` so a backend can be cheaply shared (e.g. wrapped
///   in `Arc`) without a mutable lock on every call.
/// - `RealBuf` and `CplxBuf` are generic over `F: Float` so a single backend
///   instance can serve both `f64` and future `DDReal` precisions.
pub trait ComputeBackend: Send + Sync {
    /// Backend-owned buffer of real scalars of type `F`.
    type RealBuf<F: Float>: Send;
    /// Backend-owned buffer of complex scalars of type `F`.
    type CplxBuf<F: Float>: Send;

    /// Short human-readable name for logging (e.g. `"cpu (rayon)"`).
    fn name(&self) -> &str;

    /// Allocate a real buffer of `len` elements, zero-initialised.
    fn alloc_real<F: Float>(&self, len: usize) -> Self::RealBuf<F>;

    /// Allocate a complex buffer of `len` elements, zero-initialised.
    fn alloc_cplx<F: Float>(&self, len: usize) -> Self::CplxBuf<F>;

    /// Copy `host` into the backend buffer `dst`.
    ///
    /// Panics in debug mode if `host.len() != dst` capacity.
    fn upload_cplx<F: Float>(&self, host: &[Complex<F>], dst: &mut Self::CplxBuf<F>);

    /// Copy the backend buffer `src` into `host`.
    ///
    /// Panics in debug mode if `host.len() != src` capacity.
    fn download_cplx<F: Float>(&self, src: &Self::CplxBuf<F>, host: &mut [Complex<F>]);

    /// Compensated (Kahan or pairwise) reduction: sum of |z|^2 over the buffer.
    ///
    /// Uses compensated summation to reduce floating-point error on large
    /// spectral arrays (comparable to `kahan_sum` in `vonkarman-core::kahan`).
    fn sum_norm_sq<F: Float>(&self, buf: &Self::CplxBuf<F>) -> F;
}
