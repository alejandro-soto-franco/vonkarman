//! Runtime FFT backend selection with automatic fallback.

use crate::backend::FftBackend;
use crate::cufft::CufftBackend;
#[cfg(feature = "gpufft-cuda")]
use crate::gpufft_backend::GpufftBackend;
use crate::ndrustfft_backend::NdrustfftBackend;

/// Backend selection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendMode {
    /// Try GPU backends first (gpufft when built with that feature, then
    /// the legacy libloading-based CufftBackend), then fall back to CPU.
    Auto,
    /// Force the standalone `gpufft` crate's CUDA backend. Only available
    /// when vonkarman-fft is built with the `gpufft-cuda` feature.
    Gpufft,
    /// Force the legacy libloading-based `CufftBackend`. Fails if CUDA
    /// libraries are not present at runtime.
    Cufft,
    /// Force CPU backend (ndrustfft).
    Cpu,
}

impl BackendMode {
    /// Parse from a string (for CLI and TOML config).
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "gpufft" => Self::Gpufft,
            "cufft" | "gpu" | "cuda" => Self::Cufft,
            "cpu" | "ndrustfft" => Self::Cpu,
            _ => Self::Auto,
        }
    }
}

/// Create an FFT backend for the given grid dimensions.
///
/// - `Auto`: try gpufft (if compiled in), then legacy CufftBackend, then
///   fall back to the CPU backend.
/// - `Gpufft`: force the standalone `gpufft` crate. Requires the
///   `gpufft-cuda` feature.
/// - `Cufft`: force the legacy libloading cuFFT path.
/// - `Cpu`: always use `NdrustfftBackend`.
pub fn create_backend(
    nx: usize,
    ny: usize,
    nz: usize,
    mode: BackendMode,
) -> Box<dyn FftBackend<f64>> {
    match mode {
        #[cfg(feature = "gpufft-cuda")]
        BackendMode::Gpufft => {
            let backend = GpufftBackend::new(nx, ny, nz)
                .expect("gpufft backend requested but CUDA is unavailable");
            tracing::info!(backend = "gpufft", nx, ny, nz, "FFT backend selected");
            Box::new(backend)
        }
        #[cfg(not(feature = "gpufft-cuda"))]
        BackendMode::Gpufft => {
            panic!(
                "gpufft backend requested, but vonkarman-fft was built without \
                 the `gpufft-cuda` feature"
            );
        }
        BackendMode::Cufft => {
            let backend =
                CufftBackend::new(nx, ny, nz).expect("cuFFT requested but CUDA is unavailable");
            tracing::info!(backend = "cuFFT (libloading)", nx, ny, nz, "FFT backend selected");
            Box::new(backend)
        }
        BackendMode::Cpu => {
            tracing::info!(backend = "ndrustfft", nx, ny, nz, "FFT backend selected");
            Box::new(NdrustfftBackend::new(nx, ny, nz))
        }
        BackendMode::Auto => {
            #[cfg(feature = "gpufft-cuda")]
            if let Ok(backend) = GpufftBackend::new(nx, ny, nz) {
                tracing::info!(backend = "gpufft", nx, ny, nz, "FFT backend selected (auto)");
                return Box::new(backend);
            }

            match CufftBackend::new(nx, ny, nz) {
                Ok(backend) => {
                    tracing::info!(
                        backend = "cuFFT (libloading)",
                        nx,
                        ny,
                        nz,
                        "FFT backend selected (auto)"
                    );
                    Box::new(backend)
                }
                Err(e) => {
                    tracing::info!(
                        backend = "ndrustfft",
                        reason = %e,
                        nx, ny, nz,
                        "GPU backends unavailable, falling back to CPU"
                    );
                    Box::new(NdrustfftBackend::new(nx, ny, nz))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_mode_returns_a_backend() {
        // On CI (no GPU), this returns NdrustfftBackend.
        // Locally with CUDA, this returns CufftBackend or GpufftBackend.
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
        assert_eq!(BackendMode::from_str_loose("gpufft"), BackendMode::Gpufft);
        assert_eq!(BackendMode::from_str_loose("cufft"), BackendMode::Cufft);
        assert_eq!(BackendMode::from_str_loose("GPU"), BackendMode::Cufft);
        assert_eq!(BackendMode::from_str_loose("cpu"), BackendMode::Cpu);
        assert_eq!(BackendMode::from_str_loose("auto"), BackendMode::Auto);
        assert_eq!(BackendMode::from_str_loose("anything"), BackendMode::Auto);
    }
}
