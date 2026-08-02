//! Streamfunction reconstruction and the .npy frame interchange.

use ndarray::Array2;
use ndarray_npy::read_npy;
use num_complex::Complex;
use std::f64::consts::TAU;
use vonkarman_2d::{Sim, Spectral2D, export};

#[test]
fn the_streamfunction_carries_the_mean_flow() {
    // With no vorticity, psi = u_mean * y exactly, so d(psi)/dy = u_mean.
    let (nx, ny, lx, ly) = (64, 32, 2.0 * TAU, TAU);
    let s = Spectral2D::new(nx, ny, lx, ly);
    let (_dx, dy) = s.spacing();
    let zero = Array2::<Complex<f64>>::zeros((nx, ny / 2 + 1));
    let mut sim = Sim::new(s, zero.clone(), zero.clone(), zero, 1e-3, 1e-4, 1e-4);
    sim.set_mean_flow(1.5);
    let psi = sim.streamfunction();
    for i in 0..nx {
        for j in 0..ny - 1 {
            let slope = (psi[[i, j + 1]] - psi[[i, j]]) / dy;
            assert!((slope - 1.5).abs() < 1e-12, "d(psi)/dy = {slope}");
        }
    }
}

#[test]
fn the_streamfunction_solves_the_poisson_equation() {
    // psi = sin(a x) sin(b y) with omega = (a^2 + b^2) psi and no mean flow.
    let (nx, ny, lx, ly) = (64, 32, 2.0 * TAU, TAU);
    let s = Spectral2D::new(nx, ny, lx, ly);
    let (dx, dy) = s.spacing();
    let (a, b) = (TAU / lx, TAU / ly);
    let mut omega = Array2::<f64>::zeros((nx, ny));
    let mut expected = Array2::<f64>::zeros((nx, ny));
    for i in 0..nx {
        for j in 0..ny {
            let p = (a * (i as f64 * dx)).sin() * (b * (j as f64 * dy)).sin();
            expected[[i, j]] = p;
            omega[[i, j]] = (a * a + b * b) * p;
        }
    }
    let wh = s.forward(&omega);
    let zero = Array2::<Complex<f64>>::zeros(wh.raw_dim());
    let sim = Sim::new(s, wh, zero.clone(), zero, 1e-3, 1e-4, 1e-4);
    let psi = sim.streamfunction();
    let err = psi
        .iter()
        .zip(expected.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(err < 1e-12, "psi error {err:e}");
}

#[test]
fn frames_round_trip_through_npy() {
    let dir = tempdir();
    let (nx, ny) = (16, 8);
    let mut psi = Array2::<f64>::zeros((nx, ny));
    for i in 0..nx {
        for j in 0..ny {
            psi[[i, j]] = (i * ny + j) as f64;
        }
    }
    let omega = psi.clone();
    let speed = psi.clone();
    export::write_frame(&dir, 7, &psi, &omega, &speed).unwrap();
    let back: Array2<f32> = read_npy(dir.join("psi_00007.npy")).unwrap();
    assert_eq!(back.shape(), &[nx, ny]);
    for i in 0..nx {
        for j in 0..ny {
            assert_eq!(back[[i, j]], psi[[i, j]] as f32);
        }
    }
}

/// A scratch directory that does not need a dev-dependency.
fn tempdir() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("vk2d-export-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}
