//! Rectangular-box transforms, wavenumbers and the Poisson solve.

use ndarray::Array2;
use std::f64::consts::TAU;
use vonkarman_2d::Spectral2D;

/// Grid coordinates for a `(nx, ny)` box of size `lx x ly`.
fn coords(nx: usize, ny: usize, lx: f64, ly: f64) -> (Vec<f64>, Vec<f64>) {
    (
        (0..nx).map(|i| i as f64 * lx / nx as f64).collect(),
        (0..ny).map(|j| j as f64 * ly / ny as f64).collect(),
    )
}

#[test]
fn forward_inverse_roundtrip_on_a_rectangular_box() {
    let (nx, ny, lx, ly) = (64, 32, 2.0 * TAU, TAU);
    let s = Spectral2D::new(nx, ny, lx, ly);
    let (xs, ys) = coords(nx, ny, lx, ly);
    let mut f = Array2::<f64>::zeros((nx, ny));
    for i in 0..nx {
        for j in 0..ny {
            f[[i, j]] = (TAU * 2.0 * xs[i] / lx).sin() * (TAU * 3.0 * ys[j] / ly).cos()
                + 0.3 * (TAU * xs[i] / lx).cos();
        }
    }
    let back = s.inverse(&s.forward(&f));
    let err = f
        .iter()
        .zip(back.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(err < 1e-12, "roundtrip error {err:e}");
}

#[test]
fn velocity_matches_the_analytic_field_on_a_rectangular_box() {
    // psi = sin(2 pi x / lx) sin(2 pi y / ly)
    // omega = -lap(psi) = (a^2 + b^2) psi,   a = 2 pi / lx,  b = 2 pi / ly
    // u = d(psi)/dy =  b sin(a x) cos(b y)
    // v = -d(psi)/dx = -a cos(a x) sin(b y)
    let (nx, ny, lx, ly) = (64, 32, 2.0 * TAU, TAU);
    let s = Spectral2D::new(nx, ny, lx, ly);
    let (xs, ys) = coords(nx, ny, lx, ly);
    let (a, b) = (TAU / lx, TAU / ly);
    let mut omega = Array2::<f64>::zeros((nx, ny));
    let mut ue = Array2::<f64>::zeros((nx, ny));
    let mut ve = Array2::<f64>::zeros((nx, ny));
    for i in 0..nx {
        for j in 0..ny {
            let (sa, ca) = ((a * xs[i]).sin(), (a * xs[i]).cos());
            let (sb, cb) = ((b * ys[j]).sin(), (b * ys[j]).cos());
            omega[[i, j]] = (a * a + b * b) * sa * sb;
            ue[[i, j]] = b * sa * cb;
            ve[[i, j]] = -a * ca * sb;
        }
    }
    let (u, v) = s.velocity(&s.forward(&omega));
    let eu = u
        .iter()
        .zip(ue.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    let ev = v
        .iter()
        .zip(ve.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(eu < 1e-12, "u error {eu:e}");
    assert!(ev < 1e-12, "v error {ev:e}");
}

#[test]
fn new_square_reproduces_the_legacy_box() {
    let s = Spectral2D::new_square(32);
    assert_eq!(s.nx(), 32);
    assert_eq!(s.ny(), 32);
    assert!((s.lx() - TAU).abs() < 1e-15);
    assert!((s.ly() - TAU).abs() < 1e-15);
}
