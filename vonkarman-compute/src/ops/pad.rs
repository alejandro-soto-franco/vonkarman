//! Spectral pad and truncate for the 3/2 dealiasing rule, pointwise index remap.
//!
//! These mirror the private `zero_pad_spectral` and `truncate_spectral`
//! functions of [`vonkarman_fft::dealiased`], extracted as flat-slice bodies so
//! the same logic can run on either backend (the GPU kernels in
//! `kernels/src/main.rs` reproduce them). Both are pure data movement: a
//! complex coefficient is copied from one grid position to another with no
//! arithmetic on the value, so the GPU result is BIT-IDENTICAL to the CPU body.
//!
//! The `(3/2)^3` amplitude scaling that the dealiasing caller applies around
//! these steps is a SEPARATE pointwise multiply (see `dealiased_cross_product`),
//! deliberately NOT folded in here: pad and truncate are index remaps only.
//!
//! # Layout
//!
//! Buffers are interleaved-complex flat (`[re0, im0, re1, im1, ...]`,
//! length `2 * num_points`), row-major over `[ix, iy, iz]` (the same layout the
//! other operators use). A complex element at logical `[ix, iy, iz]` of a grid
//! with dimensions `(_, ny_dim, nz_dim)` lives at flat offset
//! `2 * (((ix * ny_dim) + iy) * nz_dim + iz)` for the real part, `+ 1` for the
//! imaginary part.
//!
//! The negative-frequency remap matches the host exactly: along the `x` and `y`
//! axes the first `n/2` positive frequencies map to the same index, and the
//! remaining (negative) frequencies map to the tail of the destination axis.
//! The `z` axis is the R2C half-spectrum (positive frequencies only) and is
//! copied directly.

use vonkarman_core::float::Float;

/// Flat real offset of the real slot of logical `[ix, iy, iz]` in a grid with
/// dimensions `(_, ny_dim, nz_dim)`. The imaginary slot is `+ 1`.
#[inline]
fn cplx_offset(ix: usize, iy: usize, iz: usize, ny_dim: usize, nz_dim: usize) -> usize {
    2 * (((ix * ny_dim) + iy) * nz_dim + iz)
}

/// Zero-pad an `N`-grid spectral field into a larger `(3N/2)`-grid buffer.
///
/// `src` is the interleaved-complex source of shape `(snx, sny, snz)`
/// (length `2 * snx * sny * snz`); `dst` is the interleaved-complex destination
/// of shape `(dnx, dny, dnz)` (length `2 * dnx * dny * dnz`). `nx`/`ny` are the
/// ORIGINAL `N`-grid extents used to split positive from negative frequencies.
///
/// `dst` is fully overwritten: it is zeroed first, then every `src` coefficient
/// is scattered into its padded position, mapping negative `x`/`y` frequencies
/// to the tail of the destination axis (the `z` axis is the R2C half-spectrum
/// and is copied directly). This matches `vonkarman_fft`'s `zero_pad_spectral`.
#[allow(clippy::too_many_arguments)]
pub fn zero_pad_inplace<F: Float>(
    src: &[F],
    dst: &mut [F],
    snx: usize,
    sny: usize,
    snz: usize,
    dnx: usize,
    dny: usize,
    dnz: usize,
    nx: usize,
    ny: usize,
) {
    debug_assert!(
        src.len() == 2 * snx * sny * snz && dst.len() == 2 * dnx * dny * dnz,
        "zero_pad_inplace: src must be length 2*snx*sny*snz, dst length 2*dnx*dny*dnz"
    );
    debug_assert!(
        snz <= dnz && snx <= dnx && sny <= dny,
        "zero_pad_inplace: destination grid must be no smaller than source"
    );

    // dst is fully rewritten; zero it so the padding region is empty (the GPU
    // relies on alloc_cplx already zeroing, but the CPU body owns it here).
    for v in dst.iter_mut() {
        *v = F::ZERO;
    }

    for ix in 0..snx {
        let dix = if ix < nx / 2 { ix } else { dnx - (snx - ix) };
        for iy in 0..sny {
            let diy = if iy < ny / 2 { iy } else { dny - (sny - iy) };
            for iz in 0..snz {
                // z is the R2C half-spectrum (positive freqs only): copy direct.
                let s = cplx_offset(ix, iy, iz, sny, snz);
                let d = cplx_offset(dix, diy, iz, dny, dnz);
                dst[d] = src[s];
                dst[d + 1] = src[s + 1];
            }
        }
    }
}

/// Truncate a padded `(3N/2)`-grid spectral field back to an `N`-grid buffer.
///
/// `src` is the interleaved-complex padded source of shape `(snx, sny, snz)`
/// (length `2 * snx * sny * snz`); `dst` is the interleaved-complex destination
/// of shape `(dnx, dny, dnz)` (length `2 * dnx * dny * dnz`). `nx`/`ny` are the
/// ORIGINAL `N`-grid extents (`= dnx`/`dny`) used to split positive from
/// negative frequencies.
///
/// Every `dst` coefficient is written (gather): for each destination position
/// the matching source position is computed, mapping negative `x`/`y`
/// frequencies back from the tail of the source axis (the `z` axis is copied
/// directly). This matches `vonkarman_fft`'s `truncate_spectral`, so no pre-zero
/// of `dst` is needed.
#[allow(clippy::too_many_arguments)]
pub fn truncate_inplace<F: Float>(
    src: &[F],
    dst: &mut [F],
    snx: usize,
    sny: usize,
    snz: usize,
    dnx: usize,
    dny: usize,
    dnz: usize,
    nx: usize,
    ny: usize,
) {
    debug_assert!(
        src.len() == 2 * snx * sny * snz && dst.len() == 2 * dnx * dny * dnz,
        "truncate_inplace: src must be length 2*snx*sny*snz, dst length 2*dnx*dny*dnz"
    );
    debug_assert!(
        dnz <= snz && dnx <= snx && dny <= sny,
        "truncate_inplace: destination grid must be no larger than source"
    );

    for ix in 0..dnx {
        let six = if ix < nx / 2 { ix } else { snx - (dnx - ix) };
        for iy in 0..dny {
            let siy = if iy < ny / 2 { iy } else { sny - (dny - iy) };
            for iz in 0..dnz {
                let s = cplx_offset(six, siy, iz, sny, snz);
                let d = cplx_offset(ix, iy, iz, dny, dnz);
                dst[d] = src[s];
                dst[d + 1] = src[s + 1];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use num_complex::Complex;
    use vonkarman_core::field::GridSpec;

    /// Flatten an `Array3<Complex<f64>>` into interleaved `[re, im, ...]`.
    fn flatten(a: &Array3<Complex<f64>>) -> Vec<f64> {
        let mut v = Vec::with_capacity(a.len() * 2);
        for c in a.iter() {
            v.push(c.re);
            v.push(c.im);
        }
        v
    }

    /// Host oracle for `zero_pad_spectral` (the vonkarman-fft body, which is
    /// private to that crate, so reproduced here over `Array3`).
    fn pad_oracle(
        src: &Array3<Complex<f64>>,
        dst_shape: (usize, usize, usize),
        nx: usize,
        ny: usize,
    ) -> Array3<Complex<f64>> {
        let (snx, sny, snz) = (src.shape()[0], src.shape()[1], src.shape()[2]);
        let (dnx, dny, _dnz) = dst_shape;
        let mut dst = Array3::<Complex<f64>>::zeros(dst_shape);
        for ix in 0..snx {
            let dix = if ix < nx / 2 { ix } else { dnx - (snx - ix) };
            for iy in 0..sny {
                let diy = if iy < ny / 2 { iy } else { dny - (sny - iy) };
                for iz in 0..snz {
                    dst[[dix, diy, iz]] = src[[ix, iy, iz]];
                }
            }
        }
        dst
    }

    /// Host oracle for `truncate_spectral`.
    fn truncate_oracle(
        src: &Array3<Complex<f64>>,
        dst_shape: (usize, usize, usize),
        nx: usize,
        ny: usize,
    ) -> Array3<Complex<f64>> {
        let (snx, sny, _snz) = (src.shape()[0], src.shape()[1], src.shape()[2]);
        let (dnx, dny, dnz) = dst_shape;
        let mut dst = Array3::<Complex<f64>>::zeros(dst_shape);
        for ix in 0..dnx {
            let six = if ix < nx / 2 { ix } else { snx - (dnx - ix) };
            for iy in 0..dny {
                let siy = if iy < ny / 2 { iy } else { sny - (dny - iy) };
                for iz in 0..dnz {
                    dst[[ix, iy, iz]] = src[[six, siy, iz]];
                }
            }
        }
        dst
    }

    /// Deterministic pseudo-random complex field of the given shape.
    fn make(shape: (usize, usize, usize), seed: f64) -> Array3<Complex<f64>> {
        let mut a = Array3::from_elem(shape, Complex::new(0.0, 0.0));
        let mut t = seed;
        for c in a.iter_mut() {
            *c = Complex::new(t.sin() * 1.3 + 0.2, (t * 1.7).cos() * 0.9 - 0.1);
            t += 0.031;
        }
        a
    }

    #[test]
    fn zero_pad_matches_host_body() {
        let n = 8;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let pg = grid.padded_3half();
        let (snx, sny, snz) = grid.spectral_shape();
        let (dnx, dny, dnz) = pg.spectral_shape();

        let src = make((snx, sny, snz), 0.7);
        let oracle = pad_oracle(&src, (dnx, dny, dnz), grid.nx, grid.ny);

        let fsrc = flatten(&src);
        let mut fdst = vec![0.0; 2 * dnx * dny * dnz];
        zero_pad_inplace::<f64>(
            &fsrc, &mut fdst, snx, sny, snz, dnx, dny, dnz, grid.nx, grid.ny,
        );

        let want = flatten(&oracle);
        assert_eq!(fdst.len(), want.len());
        for (j, (g, w)) in fdst.iter().zip(want.iter()).enumerate() {
            assert_eq!(g, w, "pad mismatch at flat idx {j}: got {g}, want {w}");
        }
    }

    #[test]
    fn truncate_matches_host_body() {
        let n = 8;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let pg = grid.padded_3half();
        let (snx, sny, snz) = pg.spectral_shape();
        let (dnx, dny, dnz) = grid.spectral_shape();

        let src = make((snx, sny, snz), 1.9);
        let oracle = truncate_oracle(&src, (dnx, dny, dnz), grid.nx, grid.ny);

        let fsrc = flatten(&src);
        let mut fdst = vec![0.0; 2 * dnx * dny * dnz];
        truncate_inplace::<f64>(
            &fsrc, &mut fdst, snx, sny, snz, dnx, dny, dnz, grid.nx, grid.ny,
        );

        let want = flatten(&oracle);
        assert_eq!(fdst.len(), want.len());
        for (j, (g, w)) in fdst.iter().zip(want.iter()).enumerate() {
            assert_eq!(g, w, "truncate mismatch at flat idx {j}: got {g}, want {w}");
        }
    }

    #[test]
    fn pad_then_truncate_roundtrips() {
        // Truncating a padded field back to N must recover the original N-grid
        // field exactly (the pad region is the only thing dropped).
        let n = 8;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let pg = grid.padded_3half();
        let (snx, sny, snz) = grid.spectral_shape();
        let (pnx, pny, pnz) = pg.spectral_shape();

        let src = make((snx, sny, snz), 2.4);
        let fsrc = flatten(&src);
        let mut fpad = vec![0.0; 2 * pnx * pny * pnz];
        zero_pad_inplace::<f64>(
            &fsrc, &mut fpad, snx, sny, snz, pnx, pny, pnz, grid.nx, grid.ny,
        );
        let mut fback = vec![0.0; 2 * snx * sny * snz];
        truncate_inplace::<f64>(
            &fpad, &mut fback, pnx, pny, pnz, snx, sny, snz, grid.nx, grid.ny,
        );
        for (j, (a, b)) in fsrc.iter().zip(fback.iter()).enumerate() {
            assert_eq!(a, b, "roundtrip mismatch at flat idx {j}: {a} vs {b}");
        }
    }
}
