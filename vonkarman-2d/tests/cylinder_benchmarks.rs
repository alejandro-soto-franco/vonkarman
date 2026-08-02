//! Cylinder validation against published benchmarks. All ignored by default:
//! each integrates for minutes. Run with
//!
//!   cargo test -p vonkarman-2d --test cylinder_benchmarks --release -- --ignored --nocapture

use ndarray::Array2;
use num_complex::Complex;
use vonkarman_2d::{Penalisation, Sim, Spectral2D};

struct Run {
    sim: Sim,
    cx: f64,
    cy: f64,
    radius: f64,
    dx: f64,
    dy: f64,
    dt: f64,
}

/// Build a cylinder run. `seed_asymmetry` adds a small off-centreline vorticity
/// blob just behind the body.
///
/// Without it the Re 100 case never sheds. Zero initial vorticity, a mask
/// symmetric about `cy`, and a grid whose points mirror about `cy` (with
/// `ny = 512` and `cy = ly/2`, point `j = ny/2` sits exactly on the centreline)
/// leave the whole configuration symmetric, so in exact arithmetic the wake
/// stays symmetric forever. Real shedding would then depend on round-off
/// asymmetry growing, which can take longer than any sensible sampling window.
/// The blob is small enough to decay long before sampling starts, and only sets
/// the instability going.
fn build(nx: usize, ny: usize, re: f64, seed_asymmetry: bool) -> Run {
    let d = 1.0_f64;
    let (lx, ly) = (24.0 * d, 12.0 * d);
    let spec = Spectral2D::new(nx, ny, lx, ly);
    let (dx, dy) = spec.spacing();
    let (cx, cy) = (0.25 * lx, 0.5 * ly);
    let radius = 0.5 * d;
    let u_mean = 1.0;
    let nu = u_mean * d / re;
    let dt = 0.25 * dx / u_mean;

    let mut omega = Array2::<f64>::zeros((nx, ny));
    if seed_asymmetry {
        // One Gaussian blob at (cx + d, cy + d/4), amplitude 1e-2 U/d.
        let (bx, by, core) = (cx + d, cy + 0.25 * d, 0.25 * d);
        for i in 0..nx {
            for j in 0..ny {
                let (x, y) = (i as f64 * dx, j as f64 * dy);
                let r2 = (x - bx).powi(2) + (y - by).powi(2);
                omega[[i, j]] = 1.0e-2 * (-r2 / (core * core)).exp();
            }
        }
    }
    let wh = spec.forward(&omega);
    let zero = Array2::<Complex<f64>>::zeros((nx, ny / 2 + 1));
    let body = Penalisation::cylinder(&spec, cx, cy, radius, 1e-3, 0.85 * lx, 0.15 * lx, 5.0);
    let mut sim = Sim::new(spec, wh, zero.clone(), zero, dt, nu, nu);
    sim.set_mean_flow(u_mean);
    sim.set_body(body);
    Run {
        sim,
        cx,
        cy,
        radius,
        dx,
        dy,
        dt,
    }
}

#[test]
#[ignore = "integrates for minutes"]
fn re40_drag_and_bubble_match_the_steady_benchmark() {
    // Steady at Re 40, so no symmetry seed is needed or wanted.
    let mut r = build(1024, 512, 40.0, false);
    // 120 convective times is comfortably past the steady state at Re 40.
    let steps = (120.0 / r.dt) as usize;
    for _ in 0..steps {
        r.sim.step();
    }
    let (u, v) = r.sim.total_velocity();
    let (fx, _fy) = r.sim.body().unwrap().force(&u, &v);
    let cd = fx / (0.5 * 1.0 * 1.0); // C_d = F_x / (0.5 rho U^2 d), rho = U = d = 1
    println!("Re 40: C_d = {cd:.3}");
    assert!((1.45..=1.75).contains(&cd), "C_d = {cd:.3}");

    // Recirculation bubble: from the rear of the body along the centreline to
    // the first sign change of u.
    let j = (r.cy / r.dy).round() as usize;
    let i0 = ((r.cx + r.radius) / r.dx).ceil() as usize;
    let mut length = 0.0;
    for i in i0..r.sim.spec.nx() {
        if u[[i, j]] > 0.0 {
            length = i as f64 * r.dx - (r.cx + r.radius);
            break;
        }
    }
    println!("Re 40: bubble L/d = {length:.3}");
    assert!((1.8..=2.7).contains(&length), "L/d = {length:.3}");
}

#[test]
#[ignore = "integrates for minutes"]
fn re100_shedding_frequency_matches_the_benchmark() {
    let mut r = build(1024, 512, 100.0, true);
    // Discard 100 convective times of transient, then sample for 120 more.
    // At St ~ 0.165 the shedding period is about 6 convective times, so 120
    // covers roughly 20 cycles, well above what a frequency estimate needs.
    let warm = (100.0 / r.dt) as usize;
    for _ in 0..warm {
        r.sim.step();
    }
    let sample_every = 20usize;
    let sample_steps = (120.0 / r.dt) as usize;
    let probe_i = ((r.cx + 2.0) / r.dx).round() as usize;
    let probe_j = (r.cy / r.dy).round() as usize;
    let mut series = Vec::new();
    for k in 0..sample_steps {
        r.sim.step();
        if k % sample_every == 0 {
            let (_u, v) = r.sim.total_velocity();
            series.push(v[[probe_i, probe_j]]);
        }
    }

    // Dominant frequency by counting upward zero crossings of the mean-removed
    // transverse velocity, which is robust for a clean periodic signal.
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let s: Vec<f64> = series.iter().map(|x| x - mean).collect();
    let mut crossings = 0usize;
    let (mut first, mut last) = (None, 0usize);
    for k in 1..s.len() {
        if s[k - 1] <= 0.0 && s[k] > 0.0 {
            if first.is_none() {
                first = Some(k);
            }
            last = k;
            crossings += 1;
        }
    }
    let first = first.expect("no shedding detected");
    assert!(crossings >= 4, "only {crossings} cycles sampled");
    let period_samples = (last - first) as f64 / (crossings - 1) as f64;
    let period = period_samples * sample_every as f64 * r.dt;
    let st = 1.0 / period; // St = f d / U with d = U = 1
    println!("Re 100: St = {st:.4}");
    assert!((0.15..=0.19).contains(&st), "St = {st:.4}");
}
