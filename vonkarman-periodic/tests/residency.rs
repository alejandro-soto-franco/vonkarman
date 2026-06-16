//! Residency proof: a cuFFT round trip on device-resident buffers performs
//! ZERO host<->device copies.
//!
//! Gated behind the `cuda` feature and SKIPS gracefully (early return) when no
//! GPU or CUDA toolkit is present, matching the skip pattern in
//! `vonkarman-compute`'s GPU tests. Run on the RTX 5060 with:
//!
//! ```text
//! CUDA_TOOLKIT_PATH=/usr/local/cuda \
//!   cargo +nightly-2026-04-03 test -p vonkarman-periodic --features cuda
//! ```

#![cfg(feature = "cuda")]

use vonkarman_compute::CudaBackend;
use vonkarman_fft::CufftBackend;
use vonkarman_periodic::{SpectralState, StateDims};

/// Forward then inverse FFT entirely on resident device buffers, proving (a)
/// correctness of the round trip after the deferred `1/N` normalisation and (b)
/// that the round trip itself copies nothing across the bus.
#[test]
fn resident_fft_roundtrip_is_zero_copy() {
    // Build the GPU backend, or SKIP if no usable device / driver.
    let backend = match CudaBackend::new(0) {
        Ok(be) => be,
        Err(e) => {
            eprintln!("skipping residency test: CudaBackend::new(0) failed: {e}");
            return;
        }
    };

    // A small cube keeps the test quick; residency is independent of size.
    let n = 16usize;
    let dims = StateDims::cubic(n);

    // The cuFFT backend loads libcufft/libcudart at runtime; SKIP if absent.
    let fft = match CufftBackend::new(n, n, n) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("skipping residency test: CufftBackend::new failed: {e}");
            return;
        }
    };

    // Bind both plans to the backend stream (same primary context). A failure
    // here means cuFFT rejected the stream; surface it rather than paper over.
    fft.set_stream(backend.stream_raw())
        .expect("cufftSetStream on the backend stream failed");

    let mut state = SpectralState::<CudaBackend, f64>::new(&backend, dims);

    // Known real field on the N grid (flat [ix][iy][iz] order, the layout the
    // CPU NdrustfftBackend and cuFFT D2Z share).
    let real_len = dims.real_len();
    let mut host: Vec<f64> = Vec::with_capacity(real_len);
    for ix in 0..n {
        let x = 2.0 * std::f64::consts::PI * (ix as f64) / (n as f64);
        for iy in 0..n {
            let y = 2.0 * std::f64::consts::PI * (iy as f64) / (n as f64);
            for iz in 0..n {
                let z = 2.0 * std::f64::consts::PI * (iz as f64) / (n as f64);
                host.push(x.sin() + 0.5 * y.cos() - 0.25 * (x + z).sin());
            }
        }
    }

    // Bracket the resident work with the transfer counter. The ONE upload below
    // and the ONE download afterwards are the only permitted transfers.
    let before = backend.transfer_count();

    // Upload 1 (setup): fill the resident real scratch.
    state.upload_real_scratch(&backend, &host);

    // Resident round trip: D2Z then Z2D on the device buffers, no host copy.
    state.resident_fft_roundtrip(&backend, &fft);

    // Download 1 (verify): pull the resident real scratch back once.
    let mut got = state.download_real_scratch(&backend);

    let after = backend.transfer_count();

    // Residency proof: exactly one upload + one download, nothing in between.
    // The FFT round trip itself issued zero host<->device copies.
    assert_eq!(
        after - before,
        2,
        "resident FFT round trip must be zero-copy: counter moved by {} (expected exactly 1 upload + 1 download)",
        after - before
    );

    // cuFFT's inverse is unnormalised; apply the deferred 1/N here (a later task
    // supplies a device scale kernel that fuses this with other scaling).
    let inv_n = 1.0 / (real_len as f64);
    for v in got.iter_mut() {
        *v *= inv_n;
    }

    // The normalised round trip must reproduce the original field.
    let mut max_rel = 0.0_f64;
    for (g, h) in got.iter().zip(host.iter()) {
        let rel = (g - h).abs() / h.abs().max(1.0);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel < 1e-12,
        "resident round trip relative error {max_rel:e} exceeds 1e-12"
    );
    eprintln!(
        "residency proof: round-trip max relative error {max_rel:e}, transfers {} (= 1 upload + 1 download)",
        after - before
    );
}
