# vonkarman Cerebrum

## Preferences
- Rust 2024 edition, MIT license
- thiserror for error types
- Heavy doc comments on every function
- ndarray for arrays, num-complex for Complex<f64>
- rayon for parallelism
- Conventional commits: feat/, fix/, patch/, chore/, docs/
- Integration tests in /tests directories
- No actor model; tokio::spawn + Arc<Mutex> if async needed

## Learnings
- FFT normalization: ndrustfft c2r already includes 1/N^3; cuFFT does not (must apply manually)
- SpectralOps stores k_mag_sq as flat Array3 indexed [ix, iy, iz], not flattened Vec
- EtdCoeffs must be recomputed whenever dt changes (adaptive CFL)
- dealiased_cross_product needs separate 3/2-padded FFT backend
- Periodic3D uses Box<dyn FftBackend<f64>> with BackendMode selection (refactored from concrete type)

## Do-Not-Repeat
- (none yet)
