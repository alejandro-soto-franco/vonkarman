//! Flow past a penalised circular cylinder in a uniform stream, written out as
//! `.npy` frames for the flowforms renderer.
//!
//!   cylinder <config.toml>
//!
//! Config keys and their defaults are documented on [`Config`].

use std::path::PathBuf;

use ndarray::Array2;
use num_complex::Complex;
use serde::Deserialize;
use vonkarman_2d::{Penalisation, Sim, Spectral2D, export};

/// Run configuration. Lengths are in body diameters.
#[derive(Debug, Deserialize)]
struct Config {
    /// Grid points along `x`.
    nx: usize,
    /// Grid points along `y`.
    ny: usize,
    /// Box length along `x`, in diameters.
    lx_d: f64,
    /// Box length along `y`, in diameters.
    ly_d: f64,
    /// Reynolds number on the diameter.
    re: f64,
    /// Stream speed.
    u_mean: f64,
    /// Total steps.
    steps: usize,
    /// Steps between written frames.
    stride: usize,
    /// Steps run before the first frame is written, letting the wake develop.
    spin_up: usize,
    /// Penalisation time constant.
    eta_p: f64,
    /// Peak fringe relaxation rate.
    sigma_max: f64,
    /// Where frames go.
    output_dir: PathBuf,
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: cylinder <config.toml>");
    let text = std::fs::read_to_string(&path).expect("read config");
    let cfg: Config = toml::from_str(&text).expect("parse config");

    let d = 1.0_f64;
    let (lx, ly) = (cfg.lx_d * d, cfg.ly_d * d);
    let spec = Spectral2D::new(cfg.nx, cfg.ny, lx, ly);
    let (dx, _dy) = spec.spacing();
    let (cx, cy) = (0.25 * lx, 0.5 * ly);
    let radius = 0.5 * d;
    let nu = cfg.u_mean * d / cfg.re;
    let dt = 0.25 * dx / cfg.u_mean;

    let zero = Array2::<Complex<f64>>::zeros((cfg.nx, cfg.ny / 2 + 1));
    let body = Penalisation::cylinder(
        &spec,
        cx,
        cy,
        radius,
        cfg.eta_p,
        0.85 * lx,
        0.15 * lx,
        cfg.sigma_max,
    );
    let chi = body.chi().clone();
    let mut sim = Sim::new(spec, zero.clone(), zero.clone(), zero, dt, nu, nu);
    sim.set_mean_flow(cfg.u_mean);
    sim.set_body(body);

    export::write_mask(&cfg.output_dir, &chi).expect("write mask");

    let t0 = std::time::Instant::now();
    let mut frame = 0usize;
    for step in 0..=cfg.steps {
        if step >= cfg.spin_up && (step - cfg.spin_up).is_multiple_of(cfg.stride) {
            let (u, v) = sim.total_velocity();
            let mut speed = Array2::<f64>::zeros(u.raw_dim());
            for ((s, &u), &v) in speed.iter_mut().zip(u.iter()).zip(v.iter()) {
                *s = (u * u + v * v).sqrt();
            }
            export::write_frame(
                &cfg.output_dir,
                frame,
                &sim.streamfunction(),
                &sim.vorticity(),
                &speed,
            )
            .expect("write frame");
            frame += 1;
            if frame.is_multiple_of(20) {
                println!("  frame {frame} (step {step}, {:.0?})", t0.elapsed());
            }
        }
        if step < cfg.steps {
            sim.step();
        }
    }

    export::write_meta(
        &cfg.output_dir,
        &export::Meta {
            nx: cfg.nx,
            ny: cfg.ny,
            lx,
            ly,
            u_mean: cfg.u_mean,
            radius,
            cx,
            cy,
            dt,
            stride: cfg.stride,
            re: cfg.re,
            frames: frame,
        },
    )
    .expect("write meta");
    println!("done: {frame} frames in {:.0?}", t0.elapsed());
}
