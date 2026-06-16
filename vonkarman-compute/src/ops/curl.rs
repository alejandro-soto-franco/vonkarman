//! Spectral curl `omega_hat = i k x u_hat`, pointwise over a complex field.
//!
//! This mirrors [`vonkarman_core::spectral_ops::SpectralOps::curl`], extracted
//! as a flat-slice body so it can run on either backend. The three velocity
//! components and the three vorticity outputs are interleaved complex buffers
//! (`[re0, im0, re1, im1, ...]`, the same layout `cuda.rs` uses); the
//! wavenumbers `kx`, `ky`, `kz` are real, one value per spectral grid point in
//! the same flat order as the complex buffers.
//!
//! In spectral space the curl is a wavenumber cross product, then a
//! multiplication by `i` (which maps `(re, im) -> (-im, re)`):
//!
//! ```text
//! omega_x = i (ky uz - kz uy)
//! omega_y = i (kz ux - kx uz)
//! omega_z = i (kx uy - ky ux)
//! ```

use vonkarman_core::float::Float;

/// Compute `omega_hat = i k x u_hat` pointwise over interleaved complex fields.
///
/// `ux`/`uy`/`uz` and the outputs `ox`/`oy`/`oz` are interleaved complex
/// buffers of length `2 * n` (`n` spectral grid points). `kx`/`ky`/`kz` are
/// real buffers of length `n`. All lengths are checked in debug builds.
#[allow(clippy::too_many_arguments)]
pub fn curl_inplace<F: Float>(
    ux: &[F],
    uy: &[F],
    uz: &[F],
    kx: &[F],
    ky: &[F],
    kz: &[F],
    ox: &mut [F],
    oy: &mut [F],
    oz: &mut [F],
) {
    let n = kx.len();
    debug_assert!(
        ky.len() == n
            && kz.len() == n
            && ux.len() == 2 * n
            && uy.len() == 2 * n
            && uz.len() == 2 * n
            && ox.len() == 2 * n
            && oy.len() == 2 * n
            && oz.len() == 2 * n,
        "curl_inplace: complex buffers must be length 2n, wavenumber buffers length n"
    );

    for i in 0..n {
        let (re, im) = (2 * i, 2 * i + 1);
        let (kxi, kyi, kzi) = (kx[i], ky[i], kz[i]);
        let (uxr, uxi) = (ux[re], ux[im]);
        let (uyr, uyi) = (uy[re], uy[im]);
        let (uzr, uzi) = (uz[re], uz[im]);

        // omega_x = i (ky uz - kz uy):
        //   re = -(ky uz.im - kz uy.im),  im = ky uz.re - kz uy.re
        ox[re] = -(kyi * uzi - kzi * uyi);
        ox[im] = kyi * uzr - kzi * uyr;

        // omega_y = i (kz ux - kx uz)
        oy[re] = -(kzi * uxi - kxi * uzi);
        oy[im] = kzi * uxr - kxi * uzr;

        // omega_z = i (kx uy - ky ux)
        oz[re] = -(kxi * uyi - kyi * uxi);
        oz[im] = kxi * uyr - kyi * uxr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use num_complex::Complex;
    use vonkarman_core::field::GridSpec;
    use vonkarman_core::spectral_ops::SpectralOps;

    /// Expand the per-axis wavenumbers `kx[ix]`, `ky[iy]`, `kz[iz]` into flat
    /// per-spectral-point buffers in the row-major `[ix, iy, iz]` order that
    /// `Array3` (and our interleaved complex layout) use.
    fn flat_k(ops: &SpectralOps<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (snx, sny, snz) = ops.grid.spectral_shape();
        let mut fkx = Vec::with_capacity(snx * sny * snz);
        let mut fky = Vec::with_capacity(snx * sny * snz);
        let mut fkz = Vec::with_capacity(snx * sny * snz);
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    fkx.push(ops.kx[ix]);
                    fky.push(ops.ky[iy]);
                    fkz.push(ops.kz[iz]);
                }
            }
        }
        (fkx, fky, fkz)
    }

    /// Flatten an `Array3<Complex<f64>>` into interleaved `[re, im, ...]`.
    fn flatten(a: &Array3<Complex<f64>>) -> Vec<f64> {
        let mut v = Vec::with_capacity(a.len() * 2);
        for c in a.iter() {
            v.push(c.re);
            v.push(c.im);
        }
        v
    }

    #[test]
    fn matches_spectral_ops_curl_on_random_field() {
        let grid = GridSpec::cubic(8, 2.0 * std::f64::consts::PI);
        let ops = SpectralOps::<f64>::new(&grid);
        let (snx, sny, snz) = grid.spectral_shape();
        let shape = (snx, sny, snz);

        // Deterministic pseudo-random complex field per component.
        let make = |seed: f64| -> Array3<Complex<f64>> {
            let mut a = Array3::from_elem(shape, Complex::new(0.0, 0.0));
            let mut t = seed;
            for c in a.iter_mut() {
                *c = Complex::new((t).sin() * 1.3 + 0.2, (t * 1.7).cos() * 0.9 - 0.1);
                t += 0.031;
            }
            a
        };
        let u_hat = [make(0.1), make(1.3), make(2.7)];
        let mut omega_ref = [
            Array3::from_elem(shape, Complex::new(0.0, 0.0)),
            Array3::from_elem(shape, Complex::new(0.0, 0.0)),
            Array3::from_elem(shape, Complex::new(0.0, 0.0)),
        ];
        ops.curl(&u_hat, &mut omega_ref);

        let (fkx, fky, fkz) = flat_k(&ops);
        let (fux, fuy, fuz) = (flatten(&u_hat[0]), flatten(&u_hat[1]), flatten(&u_hat[2]));
        let n = snx * sny * snz;
        let (mut fox, mut foy, mut foz) = (vec![0.0; 2 * n], vec![0.0; 2 * n], vec![0.0; 2 * n]);
        curl_inplace::<f64>(
            &fux, &fuy, &fuz, &fkx, &fky, &fkz, &mut fox, &mut foy, &mut foz,
        );

        let ref_flat = [
            flatten(&omega_ref[0]),
            flatten(&omega_ref[1]),
            flatten(&omega_ref[2]),
        ];
        let got = [&fox, &foy, &foz];
        for comp in 0..3 {
            for j in 0..2 * n {
                let d = (got[comp][j] - ref_flat[comp][j]).abs();
                assert!(
                    d < 1e-13,
                    "curl mismatch comp {comp} idx {j}: got {}, ref {}, |d| {d:e}",
                    got[comp][j],
                    ref_flat[comp][j]
                );
            }
        }
    }
}
