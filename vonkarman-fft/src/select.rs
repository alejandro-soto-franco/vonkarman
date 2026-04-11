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
            let backend =
                CufftBackend::new(nx, ny, nz).expect("cuFFT requested but CUDA is unavailable");
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
