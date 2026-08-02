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

use vonkarman_2d::Sim;

/// A compact vorticity blob in a uniform stream translates at the stream speed.
#[test]
fn mean_flow_advects_a_blob_downstream() {
    let (nx, ny, lx, ly) = (128, 64, 4.0 * TAU, 2.0 * TAU);
    let s = Spectral2D::new(nx, ny, lx, ly);
    let (dx, dy) = s.spacing();
    let (cx, cy) = (0.25 * lx, 0.5 * ly);
    let core = 0.5;
    let mut omega = Array2::<f64>::zeros((nx, ny));
    for i in 0..nx {
        for j in 0..ny {
            let (x, y) = (i as f64 * dx, j as f64 * dy);
            let r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            omega[[i, j]] = (-r2 / (core * core)).exp();
        }
    }
    let wh = s.forward(&omega);
    let zero = Array2::<num_complex::Complex<f64>>::zeros(wh.raw_dim());
    let u_mean = 1.0;
    let dt = 2.0e-3;
    let steps = 500;
    let mut sim = Sim::new(s, wh, zero.clone(), zero, dt, 1.0e-4, 1.0e-4);
    sim.set_mean_flow(u_mean);
    for _ in 0..steps {
        sim.step();
    }

    // Vorticity-weighted centroid in x, on a box wide enough that the blob
    // does not wrap within the integration time.
    let w = sim.vorticity();
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for i in 0..nx {
        let x = i as f64 * dx;
        for j in 0..ny {
            let a = w[[i, j]].max(0.0);
            num += a * x;
            den += a;
        }
    }
    let centroid = num / den;
    let expected = cx + u_mean * dt * steps as f64;
    assert!(
        (centroid - expected).abs() < 2.0 * dx,
        "centroid {centroid:.4}, expected {expected:.4}"
    );
}
