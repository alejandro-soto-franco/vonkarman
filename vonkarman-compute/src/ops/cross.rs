//! Physical-space cross product `c = u x omega`, pointwise.
//!
//! This mirrors the inner loop of `vonkarman_fft::dealiased_cross_product`
//! (the `u x omega` step in rotation form), extracted as a flat-slice body so
//! it can run on either backend. Components are passed separately, in the
//! `[x, y, z]` order the solver uses:
//!
//! ```text
//! c_x = u_y * o_z - u_z * o_y
//! c_y = u_z * o_x - u_x * o_z
//! c_z = u_x * o_y - u_y * o_x
//! ```

use vonkarman_core::float::Float;

/// Compute `c = u x omega` pointwise over equal-length physical fields.
///
/// All nine slices must have the same length (one entry per physical grid
/// point). Panics in debug builds on a length mismatch.
#[allow(clippy::too_many_arguments)]
pub fn cross_product_inplace<F: Float>(
    ux: &[F],
    uy: &[F],
    uz: &[F],
    ox: &[F],
    oy: &[F],
    oz: &[F],
    cx: &mut [F],
    cy: &mut [F],
    cz: &mut [F],
) {
    let n = ux.len();
    debug_assert!(
        uy.len() == n
            && uz.len() == n
            && ox.len() == n
            && oy.len() == n
            && oz.len() == n
            && cx.len() == n
            && cy.len() == n
            && cz.len() == n,
        "cross_product_inplace: all slices must have equal length"
    );

    for i in 0..n {
        cx[i] = uy[i] * oz[i] - uz[i] * oy[i];
        cy[i] = uz[i] * ox[i] - ux[i] * oz[i];
        cz[i] = ux[i] * oy[i] - uy[i] * ox[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_of_x_and_y_is_z() {
        // u = x_hat, omega = y_hat  =>  u x omega = z_hat, at every point.
        let n = 8;
        let (ux, uy, uz) = (vec![1.0_f64; n], vec![0.0; n], vec![0.0; n]);
        let (ox, oy, oz) = (vec![0.0_f64; n], vec![1.0; n], vec![0.0; n]);
        let (mut cx, mut cy, mut cz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        cross_product_inplace::<f64>(&ux, &uy, &uz, &ox, &oy, &oz, &mut cx, &mut cy, &mut cz);
        for i in 0..n {
            assert_eq!((cx[i], cy[i], cz[i]), (0.0, 0.0, 1.0));
        }
    }

    #[test]
    fn cross_is_antisymmetric() {
        // u x o = -(o x u) for arbitrary fields.
        let n = 5;
        let ux: Vec<f64> = (0..n).map(|i| 0.1 * i as f64 + 0.3).collect();
        let uy: Vec<f64> = (0..n).map(|i| -0.2 * i as f64 + 1.0).collect();
        let uz: Vec<f64> = (0..n).map(|i| 0.7 * i as f64).collect();
        let ox: Vec<f64> = (0..n).map(|i| 0.5 - 0.05 * i as f64).collect();
        let oy: Vec<f64> = (0..n).map(|i| 0.9 * i as f64 - 0.4).collect();
        let oz: Vec<f64> = (0..n).map(|i| 0.05 * i as f64 + 0.2).collect();

        let (mut ax, mut ay, mut az) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let (mut bx, mut by, mut bz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        cross_product_inplace::<f64>(&ux, &uy, &uz, &ox, &oy, &oz, &mut ax, &mut ay, &mut az);
        cross_product_inplace::<f64>(&ox, &oy, &oz, &ux, &uy, &uz, &mut bx, &mut by, &mut bz);
        for i in 0..n {
            assert!((ax[i] + bx[i]).abs() < 1e-14);
            assert!((ay[i] + by[i]).abs() < 1e-14);
            assert!((az[i] + bz[i]).abs() < 1e-14);
        }
    }
}
