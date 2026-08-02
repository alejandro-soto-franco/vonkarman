//! Body mask, no-slip enforcement and fringe damping.

use ndarray::Array2;
use num_complex::Complex;
use vonkarman_2d::{Penalisation, Sim, Spectral2D};

/// Domain 24 d x 12 d with the body at a quarter of the streamwise extent,
/// matching the production geometry at a test-sized resolution.
fn setup(nx: usize, ny: usize) -> (Spectral2D, f64, f64, f64, f64) {
    let d = 1.0_f64; // body diameter
    let (lx, ly) = (24.0 * d, 12.0 * d);
    let (cx, cy) = (0.25 * lx, 0.5 * ly);
    (Spectral2D::new(nx, ny, lx, ly), cx, cy, 0.5 * d, d)
}

#[test]
fn the_mask_is_one_inside_a_half_at_the_edge_and_zero_outside() {
    // nx = 384, not the brief's 256. At nx = 256 the nearest grid point to
    // the edge sample lands 0.03125 off the true edge,
    // which is 0.44 delta given delta = 0.75 dx, so chi there is 0.71, not
    // 0.5. The chi-within-0.05-of-0.5 window is only about 0.014 wide against
    // a grid spacing of 0.094, far too narrow for the assertion to be
    // reliably satisfiable at that resolution regardless of the mask
    // implementation. nx = 384 makes 24 / 384 = 0.0625 divide the radius
    // exactly, so the edge sample lands exactly on the mathematical edge.
    let (s, cx, cy, radius, _d) = setup(384, 128);
    let body = Penalisation::cylinder(&s, cx, cy, radius, 1e-3, 20.0, 3.0, 5.0);
    let chi = body.chi();
    let (dx, dy) = s.spacing();
    let at = |x: f64, y: f64| {
        let i = (x / dx).round() as usize % s.nx();
        let j = (y / dy).round() as usize % s.ny();
        chi[[i, j]]
    };
    assert!(at(cx, cy) > 0.999, "centre {}", at(cx, cy));
    assert!(
        (at(cx + radius, cy) - 0.5).abs() < 0.05,
        "edge {}",
        at(cx + radius, cy)
    );
    assert!(
        at(cx + 3.0 * radius, cy) < 1e-3,
        "outside {}",
        at(cx + 3.0 * radius, cy)
    );
}

#[test]
#[ignore = "2000 steps at 512 x 256 is far too slow in debug; run with \
            cargo test -p vonkarman-2d --test penalisation --release -- --ignored"]
fn the_stream_is_brought_to_rest_inside_the_body() {
    let (s, cx, cy, radius, d) = setup(512, 256);
    let (dx, dy) = s.spacing();
    let u_mean = 1.0_f64;
    let nu = u_mean * d / 100.0; // Re = 100
    let dt = 0.25 * dx / u_mean;
    let zero = Array2::<Complex<f64>>::zeros(s.forward(&Array2::zeros((s.nx(), s.ny()))).raw_dim());
    let body = Penalisation::cylinder(&s, cx, cy, radius, 1e-3, 20.0, 3.0, 5.0);
    let mut sim = Sim::new(s, zero.clone(), zero.clone(), zero, dt, nu, nu);
    sim.set_mean_flow(u_mean);
    sim.set_body(body);
    for _ in 0..2000 {
        sim.step();
    }

    // Deep inside the body the total velocity must be a small fraction of the
    // free stream. The interface itself is smoothed, so sample within 0.8 a.
    let (u, v) = sim.total_velocity();
    let mut worst = 0.0_f64;
    for i in 0..sim.spec.nx() {
        for j in 0..sim.spec.ny() {
            let (x, y) = (i as f64 * dx, j as f64 * dy);
            let r = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            if r < 0.8 * radius {
                worst = worst.max((u[[i, j]].powi(2) + v[[i, j]].powi(2)).sqrt());
            }
        }
    }
    assert!(worst < 0.05 * u_mean, "no-slip residual {worst:.4} of U");
}

/// Cheap always-run stand-in for [`the_stream_is_brought_to_rest_inside_the_body`]:
/// a small grid over few steps, just enough to show the penalisation substep
/// is wired up and pulling velocity toward rest inside the body. It does not
/// aim for the strict 0.05 U bound the heavy, release-only test uses, only a
/// substantial drop from the free stream, since `eta_p = 1e-3` is far smaller
/// than even this coarse grid's `dt`, so suppression inside `chi = 1` is
/// already close to total after a single half substep.
#[test]
fn the_no_slip_substep_pulls_velocity_toward_rest_inside_the_body() {
    let (s, cx, cy, radius, d) = setup(64, 32);
    let (dx, dy) = s.spacing();
    let u_mean = 1.0_f64;
    let nu = u_mean * d / 100.0; // Re = 100
    let dt = 0.25 * dx / u_mean;
    let zero = Array2::<Complex<f64>>::zeros(s.forward(&Array2::zeros((s.nx(), s.ny()))).raw_dim());
    let body = Penalisation::cylinder(&s, cx, cy, radius, 1e-3, 20.0, 3.0, 5.0);
    let mut sim = Sim::new(s, zero.clone(), zero.clone(), zero, dt, nu, nu);
    sim.set_mean_flow(u_mean);
    sim.set_body(body);
    for _ in 0..20 {
        sim.step();
    }
    let (u, v) = sim.total_velocity();
    let mut worst = 0.0_f64;
    for i in 0..sim.spec.nx() {
        for j in 0..sim.spec.ny() {
            let (x, y) = (i as f64 * dx, j as f64 * dy);
            let r = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            if r < 0.8 * radius {
                worst = worst.max((u[[i, j]].powi(2) + v[[i, j]].powi(2)).sqrt());
            }
        }
    }
    assert!(
        worst < 0.5 * u_mean,
        "no-slip substep did not pull velocity toward rest: {worst:.4} of U"
    );
}

/// Cheap always-run check on [`Sim::body_force`]: unlike the Angot estimator
/// it replaced, the momentum-removed force should not depend strongly on
/// `eta_p`. A body in a uniform stream should feel drag along `+x`, and that
/// drag should barely move when `eta_p` is halved (a correct estimator is
/// exact for the scheme regardless of `eta_p`; the broken one this replaced
/// scaled roughly as `1 / eta_p`, so halving `eta_p` would have roughly
/// doubled it).
#[test]
fn the_body_force_estimate_is_stable_under_a_change_in_eta_p() {
    let drag_at = |eta_p: f64| -> f64 {
        let (s, cx, cy, radius, d) = setup(96, 48);
        let (dx, _dy) = s.spacing();
        let u_mean = 1.0_f64;
        let nu = u_mean * d / 100.0; // Re = 100
        let dt = 0.25 * dx / u_mean;
        let zero =
            Array2::<Complex<f64>>::zeros(s.forward(&Array2::zeros((s.nx(), s.ny()))).raw_dim());
        let body = Penalisation::cylinder(&s, cx, cy, radius, eta_p, 20.0, 3.0, 5.0);
        let mut sim = Sim::new(s, zero.clone(), zero.clone(), zero, dt, nu, nu);
        sim.set_mean_flow(u_mean);
        sim.set_body(body);
        for _ in 0..8 {
            sim.step();
        }
        sim.body_force().0
    };
    let fx_full = drag_at(1e-3);
    let fx_half = drag_at(5e-4);
    assert!(fx_full > 0.0, "drag should be positive along +x: {fx_full}");
    assert!(fx_half > 0.0, "drag should be positive along +x: {fx_half}");
    let rel_diff = (fx_full - fx_half).abs() / fx_full.abs().max(fx_half.abs());
    assert!(
        rel_diff < 0.08,
        "force estimate depends on eta_p: fx(1e-3) = {fx_full:.6}, fx(5e-4) = {fx_half:.6}, \
         relative difference {:.2}%",
        100.0 * rel_diff
    );
}

#[test]
#[ignore = "500 steps at 256 x 128 does not finish within the debug test budget; \
            run with cargo test -p vonkarman-2d --test penalisation --release -- --ignored"]
fn the_fringe_removes_vorticity_carried_into_it() {
    let (s, _cx, _cy, _radius, _d) = setup(256, 128);
    let (dx, dy) = s.spacing();
    let (lx, ly) = (s.lx(), s.ly());
    // A blob placed inside the fringe strip.
    let (bx, by) = (0.92 * lx, 0.5 * ly);
    let mut omega = Array2::<f64>::zeros((s.nx(), s.ny()));
    for i in 0..s.nx() {
        for j in 0..s.ny() {
            let (x, y) = (i as f64 * dx, j as f64 * dy);
            omega[[i, j]] = (-((x - bx).powi(2) + (y - by).powi(2)) / 0.25).exp();
        }
    }
    let enstrophy = |w: &Array2<f64>| w.iter().map(|a| a * a).sum::<f64>();
    let before = enstrophy(&omega);
    let wh = s.forward(&omega);
    let zero = Array2::<Complex<f64>>::zeros(wh.raw_dim());
    let dt = 2.0e-3;
    let body = Penalisation::cylinder(
        &s,
        0.25 * lx,
        0.5 * ly,
        0.5,
        1e-3,
        0.8 * lx,
        0.15 * lx,
        20.0,
    );
    let mut sim = Sim::new(s, wh, zero.clone(), zero, dt, 1e-4, 1e-4);
    sim.set_body(body);
    for _ in 0..500 {
        sim.step();
    }
    let after = enstrophy(&sim.vorticity());
    assert!(
        after < 0.05 * before,
        "fringe left {:.1}% of the enstrophy",
        100.0 * after / before
    );
}

/// Cheap always-run stand-in for [`the_fringe_removes_vorticity_carried_into_it`]:
/// a small grid over few steps, just enough to show `sigma`, `vorticity_decay`
/// and the fringe half of `penalisation_half_step` are wired up and removing
/// vorticity from the strip. It does not aim for the heavy test's 95%
/// reduction, only a healthy majority, since `sigma_max = 20` decays vorticity
/// by `exp(-sigma * dt)` a step, comfortably visible after a handful of steps.
#[test]
fn the_fringe_substep_removes_a_healthy_fraction_of_the_enstrophy() {
    let (s, _cx, _cy, _radius, _d) = setup(64, 32);
    let (dx, dy) = s.spacing();
    let (lx, ly) = (s.lx(), s.ly());
    let (bx, by) = (0.92 * lx, 0.5 * ly);
    let mut omega = Array2::<f64>::zeros((s.nx(), s.ny()));
    for i in 0..s.nx() {
        for j in 0..s.ny() {
            let (x, y) = (i as f64 * dx, j as f64 * dy);
            omega[[i, j]] = (-((x - bx).powi(2) + (y - by).powi(2)) / 0.25).exp();
        }
    }
    let enstrophy = |w: &Array2<f64>| w.iter().map(|a| a * a).sum::<f64>();
    let before = enstrophy(&omega);
    let wh = s.forward(&omega);
    let zero = Array2::<Complex<f64>>::zeros(wh.raw_dim());
    let dt = 2.0e-3;
    let body = Penalisation::cylinder(
        &s,
        0.25 * lx,
        0.5 * ly,
        0.5,
        1e-3,
        0.8 * lx,
        0.15 * lx,
        20.0,
    );
    let mut sim = Sim::new(s, wh, zero.clone(), zero, dt, 1e-4, 1e-4);
    sim.set_body(body);
    for _ in 0..30 {
        sim.step();
    }
    let after = enstrophy(&sim.vorticity());
    assert!(
        after < 0.5 * before,
        "fringe substep did not remove a healthy fraction: {:.1}% remained",
        100.0 * after / before
    );
}
