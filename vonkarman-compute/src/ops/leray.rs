//! Leray projection `u_hat -= k (k . u_hat) / |k|^2`, pointwise in place.
//!
//! This mirrors [`vonkarman_core::spectral_ops::SpectralOps::leray_project`],
//! extracted as a flat-slice body so it can run on either backend. It projects
//! a spectral vector field onto the divergence-free subspace
//! (`P = I - k k^T / |k|^2`). The `k = 0` mode (where `|k|^2` is below a tiny
//! floor) is left unchanged so the mean flow is preserved.
//!
//! `ux`/`uy`/`uz` are interleaved complex buffers (`[re0, im0, re1, im1, ...]`,
//! length `2 n`); `kx`/`ky`/`kz` and `k2` (`= |k|^2`, the value the CPU stores
//! in `k_mag_sq`) are real, length `n`, one value per spectral grid point in
//! the same flat order.

use vonkarman_core::float::Float;

/// In-place Leray projection over interleaved complex velocity components.
///
/// `kx`/`ky`/`kz`/`k2` are real (length `n`); `ux`/`uy`/`uz` are interleaved
/// complex (length `2 n`) and are modified in place. The `k = 0` mode
/// (`k2 < 1e-30`) is left untouched, matching the reference.
#[allow(clippy::too_many_arguments)]
pub fn leray_inplace<F: Float>(
    kx: &[F],
    ky: &[F],
    kz: &[F],
    k2: &[F],
    ux: &mut [F],
    uy: &mut [F],
    uz: &mut [F],
) {
    let n = kx.len();
    debug_assert!(
        ky.len() == n
            && kz.len() == n
            && k2.len() == n
            && ux.len() == 2 * n
            && uy.len() == 2 * n
            && uz.len() == 2 * n,
        "leray_inplace: complex buffers must be length 2n, wavenumber buffers length n"
    );

    let floor = F::from_f64(1e-30);
    for i in 0..n {
        let k2i = k2[i];
        if k2i < floor {
            continue; // skip k = 0 (mean flow)
        }
        let (re, im) = (2 * i, 2 * i + 1);
        let (kxi, kyi, kzi) = (kx[i], ky[i], kz[i]);
        let inv_k2 = F::ONE / k2i;

        let (uxr, uxi) = (ux[re], ux[im]);
        let (uyr, uyi) = (uy[re], uy[im]);
        let (uzr, uzi) = (uz[re], uz[im]);

        // k . u_hat (complex dot product with real k).
        let kdu_re = kxi * uxr + kyi * uyr + kzi * uzr;
        let kdu_im = kxi * uxi + kyi * uyi + kzi * uzi;

        // u_hat -= k (k . u_hat) / |k|^2.
        ux[re] = uxr - kxi * kdu_re * inv_k2;
        ux[im] = uxi - kxi * kdu_im * inv_k2;
        uy[re] = uyr - kyi * kdu_re * inv_k2;
        uy[im] = uyi - kyi * kdu_im * inv_k2;
        uz[re] = uzr - kzi * kdu_re * inv_k2;
        uz[im] = uzi - kzi * kdu_im * inv_k2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use num_complex::Complex;
    use vonkarman_core::field::GridSpec;
    use vonkarman_core::spectral_ops::SpectralOps;

    fn flat_k(ops: &SpectralOps<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (snx, sny, snz) = ops.grid.spectral_shape();
        let cap = snx * sny * snz;
        let (mut fkx, mut fky, mut fkz, mut fk2) = (
            Vec::with_capacity(cap),
            Vec::with_capacity(cap),
            Vec::with_capacity(cap),
            Vec::with_capacity(cap),
        );
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    fkx.push(ops.kx[ix]);
                    fky.push(ops.ky[iy]);
                    fkz.push(ops.kz[iz]);
                    fk2.push(ops.k_mag_sq[[ix, iy, iz]]);
                }
            }
        }
        (fkx, fky, fkz, fk2)
    }

    fn flatten(a: &Array3<Complex<f64>>) -> Vec<f64> {
        let mut v = Vec::with_capacity(a.len() * 2);
        for c in a.iter() {
            v.push(c.re);
            v.push(c.im);
        }
        v
    }

    #[test]
    fn matches_spectral_ops_leray_and_is_divergence_free() {
        let grid = GridSpec::cubic(8, 2.0 * std::f64::consts::PI);
        let ops = SpectralOps::<f64>::new(&grid);
        let (snx, sny, snz) = grid.spectral_shape();
        let shape = (snx, sny, snz);

        let make = |seed: f64| -> Array3<Complex<f64>> {
            let mut a = Array3::from_elem(shape, Complex::new(0.0, 0.0));
            let mut t = seed;
            for c in a.iter_mut() {
                *c = Complex::new((t).sin() * 1.1 + 0.4, (t * 1.3).cos() * 0.8 - 0.2);
                t += 0.029;
            }
            a
        };
        let mut u_ref = [make(0.2), make(1.1), make(3.3)];
        let (fux, fuy, fuz) = (flatten(&u_ref[0]), flatten(&u_ref[1]), flatten(&u_ref[2]));

        ops.leray_project(&mut u_ref);

        let (fkx, fky, fkz, fk2) = flat_k(&ops);
        let (mut mux, mut muy, mut muz) = (fux.clone(), fuy.clone(), fuz.clone());
        leray_inplace::<f64>(&fkx, &fky, &fkz, &fk2, &mut mux, &mut muy, &mut muz);

        let ref_flat = [flatten(&u_ref[0]), flatten(&u_ref[1]), flatten(&u_ref[2])];
        let got = [&mux, &muy, &muz];
        let n = snx * sny * snz;
        for comp in 0..3 {
            for j in 0..2 * n {
                let d = (got[comp][j] - ref_flat[comp][j]).abs();
                assert!(
                    d < 1e-13,
                    "leray mismatch comp {comp} idx {j}: got {}, ref {}, |d| {d:e}",
                    got[comp][j],
                    ref_flat[comp][j]
                );
            }
        }

        // Divergence-free: k . u_hat ~ 0 at every non-zero mode.
        for i in 0..n {
            let (re, im) = (2 * i, 2 * i + 1);
            let div_re = fkx[i] * mux[re] + fky[i] * muy[re] + fkz[i] * muz[re];
            let div_im = fkx[i] * mux[im] + fky[i] * muy[im] + fkz[i] * muz[im];
            let mag = (div_re * div_re + div_im * div_im).sqrt();
            assert!(mag < 1e-13, "divergence at flat idx {i} = {mag:e}");
        }
    }
}
