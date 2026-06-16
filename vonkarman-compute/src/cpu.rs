//! CPU compute backend: rayon over host `Vec`s.
//!
//! Buffers are plain `Vec`s, so upload and download are simple copies. The
//! reduction uses Kahan-compensated summation (the same algorithm as
//! [`vonkarman_core::kahan`]) so that the CPU backend can serve as the
//! deterministic differential oracle for a future GPU backend.

use num_complex::Complex;
use vonkarman_core::{ComputeBackend, float::Float};

/// CPU backend. Cheap to clone and share; holds no state.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl CpuBackend {
    /// Construct a CPU backend.
    pub fn new() -> Self {
        Self
    }

    /// Borrow a complex buffer as a slice (test and oracle helper).
    pub fn as_slice<'a, F: Float>(&self, buf: &'a Vec<Complex<F>>) -> &'a [Complex<F>] {
        buf.as_slice()
    }
}

impl ComputeBackend for CpuBackend {
    type RealBuf<F: Float> = Vec<F>;
    type CplxBuf<F: Float> = Vec<Complex<F>>;

    fn name(&self) -> &str {
        "cpu (rayon)"
    }

    fn alloc_real<F: Float>(&self, len: usize) -> Self::RealBuf<F> {
        vec![F::ZERO; len]
    }

    fn alloc_cplx<F: Float>(&self, len: usize) -> Self::CplxBuf<F> {
        vec![
            Complex {
                re: F::ZERO,
                im: F::ZERO
            };
            len
        ]
    }

    fn upload_cplx<F: Float>(&self, host: &[Complex<F>], dst: &mut Self::CplxBuf<F>) {
        debug_assert_eq!(host.len(), dst.len(), "upload length mismatch");
        dst.copy_from_slice(host);
    }

    fn download_cplx<F: Float>(&self, src: &Self::CplxBuf<F>, host: &mut [Complex<F>]) {
        debug_assert_eq!(host.len(), src.len(), "download length mismatch");
        host.copy_from_slice(src);
    }

    fn sum_norm_sq<F: Float>(&self, buf: &Self::CplxBuf<F>) -> F {
        // Kahan-compensated sum of |z|^2 = re*re + im*im, kept sequential for
        // determinism so the GPU reduction can be checked against this exactly.
        let mut sum = F::ZERO;
        let mut comp = F::ZERO;
        for z in buf.iter() {
            let term = z.re * z.re + z.im * z.im;
            let y = term - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_sum_norm_sq_matches_kahan() {
        let be = CpuBackend::new();
        let host: Vec<Complex<f64>> = (1..=1000).map(|k| Complex::new(k as f64, 0.0)).collect();
        let mut buf = be.alloc_cplx::<f64>(host.len());
        be.upload_cplx(&host, &mut buf);
        // sum_{k=1}^{1000} k^2 = 333_833_500
        assert!((be.sum_norm_sq::<f64>(&buf).to_f64() - 333_833_500.0).abs() < 1e-6);
    }

    #[test]
    fn cpu_sum_norm_sq_counts_both_parts() {
        let be = CpuBackend::new();
        // |3 + 4i|^2 = 25, four of them => 100.
        let host = vec![Complex::new(3.0_f64, 4.0); 4];
        let mut buf = be.alloc_cplx::<f64>(host.len());
        be.upload_cplx(&host, &mut buf);
        assert!((be.sum_norm_sq::<f64>(&buf) - 100.0).abs() < 1e-12);
    }

    #[test]
    fn cpu_upload_download_roundtrip() {
        let be = CpuBackend::new();
        let host: Vec<Complex<f64>> = (0..64)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let mut buf = be.alloc_cplx::<f64>(host.len());
        be.upload_cplx(&host, &mut buf);
        let mut back = vec![Complex::new(0.0_f64, 0.0); host.len()];
        be.download_cplx(&buf, &mut back);
        assert_eq!(host, back);
    }
}
