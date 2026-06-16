//! Real-scalar scaling of an interleaved-complex buffer, pointwise in place.
//!
//! Multiplies every `f64` of an interleaved-complex buffer
//! (`[re0, im0, re1, im1, ...]`) by one real scalar `s`. Because the scalar is
//! real, the real and imaginary slots of each complex element follow the
//! identical multiply, so the body just scales the flat buffer element by
//! element.
//!
//! This is the device-resident replacement for the host
//! `mapv_inplace(|v| Complex { re: v.re * scale, im: v.im * scale })` calls in
//! [`vonkarman_fft::dealiased_cross_product`]: the resident dealiasing path
//! folds the host's two amplitude factors (`scale` before the inverse transform,
//! `inv_scale` after the forward transform) into a single device multiply each.
//! It is deliberately a SEPARATE operator from pad/truncate, which are pure
//! index remaps (bit-identity tested); the scaling never touches their kernels.

use vonkarman_core::float::Float;

/// Scale every element of an interleaved-complex buffer by a real scalar `s`.
///
/// `buf` is the interleaved-complex buffer (`[re, im, ...]`, length `2 n`) and
/// `s` is the real factor applied to both interleaved slots of every complex
/// element. Pure pointwise multiply, so the GPU result differs from this body
/// by at most one rounding (a single multiply per slot, no fusion to exploit).
pub fn scale_cplx_inplace<F: Float>(buf: &mut [F], s: F) {
    for v in buf.iter_mut() {
        *v *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_matches_formula() {
        let n = 17;
        let mut buf: Vec<f64> = (0..2 * n)
            .map(|i| (0.031 * i as f64).sin() * 1.3 + 0.2)
            .collect();
        let orig = buf.clone();
        let s = 0.375_f64;
        scale_cplx_inplace::<f64>(&mut buf, s);
        for j in 0..2 * n {
            let want = orig[j] * s;
            assert!(
                (buf[j] - want).abs() < 1e-15,
                "scale idx {j}: got {}, want {want}",
                buf[j]
            );
        }
    }

    #[test]
    fn scale_by_one_is_identity() {
        let mut buf = vec![1.5_f64, -2.25, 3.0, 0.125];
        let orig = buf.clone();
        scale_cplx_inplace::<f64>(&mut buf, 1.0);
        assert_eq!(buf, orig);
    }
}
