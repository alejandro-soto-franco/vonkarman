//! ETD-RK4 phi-multiply per-mode operators, pointwise over a complex field.
//!
//! These mirror the per-mode arithmetic of
//! [`vonkarman_periodic`]'s `Periodic3D::etd_rk4_step`: the exponential
//! time-differencing Runge-Kutta scheme applies, per spectral mode and per
//! velocity component, real per-mode scalar coefficients (`exp_half`,
//! `exp_full`, `a21 = a31`, `a41`, `b1`, `b23`, `b4` from `EtdCoeffs`) to
//! complex field values. Every coefficient is REAL, so the real and imaginary
//! interleaved slots follow the identical linear combination; the bodies below
//! treat both slots `2 i` and `2 i + 1` the same way.
//!
//! Each operator acts on ONE velocity component at a time. The complex buffers
//! (`u`, `n`, `out`) are interleaved `[re0, im0, re1, im1, ...]`, length `2 n`;
//! the coefficient buffers (`exp_half`, `a`, `b1`, ...) are real, length `n`,
//! one value per spectral mode applied to both slots of that mode. The
//! solver/residency layer loops over the three components.
//!
//! There are three distinct patterns in the scheme:
//!
//! 1. Stages 2 and 3: `out = exp_half * u + dt * a * n` (`a` is `a21` or `a31`).
//! 2. Stage 4: `out = exp_full * u + dt * a41 * (2 n3 - n1)`.
//! 3. Final update (in place): `u = exp_full * u + dt * (b1 n1 + b23 (n2 + n3) + b4 n4)`.

use vonkarman_core::float::Float;

/// Stage 2 / stage 3 multiply: `out = exp_half * u + dt * a * n`, out of place.
///
/// `exp_half` and `a` are real per-mode buffers (length `n`); `u`, `n`, `out`
/// are interleaved complex buffers (length `2 n`). For mode `i` and slot `s`
/// in `{0, 1}` this writes `out[2 i + s] = exp_half[i] * u[2 i + s]
/// + dt * a[i] * n[2 i + s]`. With `a = a21` this is stage 2; with `a = a31`,
/// stage 3 (`a21 == a31` in the scheme, but the argument is kept general).
#[allow(clippy::too_many_arguments)]
pub fn etd_stage_axpy_inplace<F: Float>(
    exp_half: &[F],
    a: &[F],
    dt: F,
    u: &[F],
    n: &[F],
    out: &mut [F],
) {
    let m = exp_half.len();
    debug_assert!(
        a.len() == m && u.len() == 2 * m && n.len() == 2 * m && out.len() == 2 * m,
        "etd_stage_axpy_inplace: complex buffers must be length 2n, coeff buffers length n"
    );

    for i in 0..m {
        let (re, im) = (2 * i, 2 * i + 1);
        let eh = exp_half[i];
        let dta = dt * a[i];
        // out = exp_half * u + (dt a) * n, same combination on re and im.
        out[re] = eh * u[re] + dta * n[re];
        out[im] = eh * u[im] + dta * n[im];
    }
}

/// Stage 4 multiply: `out = exp_full * u + dt * a41 * (2 n3 - n1)`, out of place.
///
/// `exp_full` and `a41` are real per-mode buffers (length `n`); `u`, `n1`,
/// `n3`, `out` are interleaved complex buffers (length `2 n`). For mode `i`
/// and slot `s` this writes
/// `out[2 i + s] = exp_full[i] * u[2 i + s]
/// + dt * a41[i] * (2 n3[2 i + s] - n1[2 i + s])`.
#[allow(clippy::too_many_arguments)]
pub fn etd_stage4_inplace<F: Float>(
    exp_full: &[F],
    a41: &[F],
    dt: F,
    u: &[F],
    n1: &[F],
    n3: &[F],
    out: &mut [F],
) {
    let m = exp_full.len();
    debug_assert!(
        a41.len() == m
            && u.len() == 2 * m
            && n1.len() == 2 * m
            && n3.len() == 2 * m
            && out.len() == 2 * m,
        "etd_stage4_inplace: complex buffers must be length 2n, coeff buffers length n"
    );

    let two = F::ONE + F::ONE;
    for i in 0..m {
        let (re, im) = (2 * i, 2 * i + 1);
        let ef = exp_full[i];
        let dta = dt * a41[i];
        // dn = 2 n3 - n1; out = exp_full * u + (dt a41) * dn.
        let dn_re = two * n3[re] - n1[re];
        let dn_im = two * n3[im] - n1[im];
        out[re] = ef * u[re] + dta * dn_re;
        out[im] = ef * u[im] + dta * dn_im;
    }
}

/// Final ETD-RK4 update, in place on `u`:
/// `u = exp_full * u + dt * (b1 n1 + b23 (n2 + n3) + b4 n4)`.
///
/// `exp_full`, `b1`, `b23`, `b4` are real per-mode buffers (length `n`); `n1`,
/// `n2`, `n3`, `n4` are interleaved complex buffers (length `2 n`); `u` is the
/// interleaved complex component (length `2 n`) updated in place. For mode `i`
/// and slot `s` this sets `u[2 i + s] = exp_full[i] * u[2 i + s]
/// + dt * (b1[i] n1[2 i + s] + b23[i] (n2[2 i + s] + n3[2 i + s])
/// + b4[i] n4[2 i + s])`.
#[allow(clippy::too_many_arguments)]
pub fn etd_final_inplace<F: Float>(
    exp_full: &[F],
    b1: &[F],
    b23: &[F],
    b4: &[F],
    dt: F,
    n1: &[F],
    n2: &[F],
    n3: &[F],
    n4: &[F],
    u: &mut [F],
) {
    let m = exp_full.len();
    debug_assert!(
        b1.len() == m
            && b23.len() == m
            && b4.len() == m
            && n1.len() == 2 * m
            && n2.len() == 2 * m
            && n3.len() == 2 * m
            && n4.len() == 2 * m
            && u.len() == 2 * m,
        "etd_final_inplace: complex buffers must be length 2n, coeff buffers length n"
    );

    for i in 0..m {
        let (re, im) = (2 * i, 2 * i + 1);
        let ef = exp_full[i];
        let (b1i, b23i, b4i) = (b1[i], b23[i], b4[i]);
        // rhs = b1 n1 + b23 (n2 + n3) + b4 n4; u = exp_full * u + dt * rhs.
        let rhs_re = b1i * n1[re] + b23i * (n2[re] + n3[re]) + b4i * n4[re];
        let rhs_im = b1i * n1[im] + b23i * (n2[im] + n3[im]) + b4i * n4[im];
        u[re] = ef * u[re] + dt * rhs_re;
        u[im] = ef * u[im] + dt * rhs_im;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random real buffer of `len` values.
    fn make(seed: f64, len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| (seed + 0.031 * i as f64).sin() * 1.3 + 0.2)
            .collect()
    }

    #[test]
    fn stage_axpy_matches_formula() {
        let m = 17;
        let exp_half = make(0.1, m);
        let a = make(0.7, m);
        let dt = 0.013_f64;
        let u = make(1.3, 2 * m);
        let n = make(2.7, 2 * m);
        let mut out = vec![0.0; 2 * m];
        etd_stage_axpy_inplace::<f64>(&exp_half, &a, dt, &u, &n, &mut out);
        for i in 0..m {
            for s in 0..2 {
                let j = 2 * i + s;
                let want = exp_half[i] * u[j] + dt * a[i] * n[j];
                assert!(
                    (out[j] - want).abs() < 1e-13,
                    "stage axpy idx {j}: got {}, want {want}",
                    out[j]
                );
            }
        }
    }

    #[test]
    fn stage4_matches_formula() {
        let m = 19;
        let exp_full = make(0.2, m);
        let a41 = make(0.9, m);
        let dt = 0.021_f64;
        let u = make(1.1, 2 * m);
        let n1 = make(2.2, 2 * m);
        let n3 = make(3.3, 2 * m);
        let mut out = vec![0.0; 2 * m];
        etd_stage4_inplace::<f64>(&exp_full, &a41, dt, &u, &n1, &n3, &mut out);
        for i in 0..m {
            for s in 0..2 {
                let j = 2 * i + s;
                let dn = 2.0 * n3[j] - n1[j];
                let want = exp_full[i] * u[j] + dt * a41[i] * dn;
                assert!(
                    (out[j] - want).abs() < 1e-13,
                    "stage4 idx {j}: got {}, want {want}",
                    out[j]
                );
            }
        }
    }

    #[test]
    fn final_update_matches_formula() {
        let m = 23;
        let exp_full = make(0.3, m);
        let b1 = make(0.5, m);
        let b23 = make(0.6, m);
        let b4 = make(0.8, m);
        let dt = 0.017_f64;
        let n1 = make(1.0, 2 * m);
        let n2 = make(1.5, 2 * m);
        let n3 = make(2.0, 2 * m);
        let n4 = make(2.5, 2 * m);
        let u0 = make(3.0, 2 * m);
        let mut u = u0.clone();
        etd_final_inplace::<f64>(&exp_full, &b1, &b23, &b4, dt, &n1, &n2, &n3, &n4, &mut u);
        for i in 0..m {
            for s in 0..2 {
                let j = 2 * i + s;
                let rhs = b1[i] * n1[j] + b23[i] * (n2[j] + n3[j]) + b4[i] * n4[j];
                let want = exp_full[i] * u0[j] + dt * rhs;
                assert!(
                    (u[j] - want).abs() < 1e-13,
                    "final idx {j}: got {}, want {want}",
                    u[j]
                );
            }
        }
    }
}
