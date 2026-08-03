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
    // which is 0.44 delta given delta = 0.75 dx.max(dy), so chi there is 0.71, not
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

/// The mask edge is smoothed along `y` as well as along `x`.
///
/// The edge width is one physical length sampled along both axes, so sizing it
/// from `dx` alone leaves it under-resolved in `y` whenever cells are taller
/// than they are wide, and a sample line along `x` cannot see that. This runs
/// 512 x 128 over the same 24 d x 12 d domain, cell aspect ratio `dy / dx = 2`,
/// the highest [`Penalisation::cylinder`] admits, which is where the difference
/// is largest. Two shape measures are taken through the body centre along `y`:
/// grid samples in the transition band `0.05 < chi < 0.95`, and the largest
/// jump in `chi` between adjacent samples.
///
/// Measured on this configuration, edge width from `dx.max(dy)` against
/// `dx` alone:
///
/// ```text
///   dx.max(dy)   band 4 samples   largest jump 0.5641
///   dx           band 2 samples   largest jump 0.8277
/// ```
///
/// The thresholds below sit between the two pairs, at 3 samples and 0.70, so
/// the test passes the edge sized from the coarser axis and fails a return to
/// the square-grid assumption. The 384 x 128 grid the shape test above uses
/// cannot carry this guard: at aspect ratio 1.5 the band holds 4 samples either
/// way.
#[test]
fn the_mask_edge_is_smoothed_along_y() {
    let (s, cx, cy, radius, _d) = setup(512, 128);
    let (dx, dy) = s.spacing();
    assert!(
        (dy / dx - 2.0).abs() < 1e-12,
        "this test needs anisotropic cells: dy / dx = {}",
        dy / dx
    );
    let body = Penalisation::cylinder(&s, cx, cy, radius, 1e-3, 20.0, 3.0, 5.0);
    let chi = body.chi();
    let i = (cx / dx).round() as usize % s.nx();
    let band = (0..s.ny())
        .filter(|&j| {
            let c = chi[[i, j]];
            c > 0.05 && c < 0.95
        })
        .count();
    let jump = (0..s.ny())
        .map(|j| (chi[[i, (j + 1) % s.ny()]] - chi[[i, j]]).abs())
        .fold(0.0_f64, f64::max);
    println!("transition band samples along y: {band}, largest jump: {jump:.4}");
    assert!(
        band >= 3,
        "mask edge under-resolved along y: only {band} samples in 0.05 < chi < 0.95"
    );
    assert!(
        jump <= 0.70,
        "mask edge steps along y: largest jump between adjacent samples is {jump:.4}"
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
/// `eta_p`. A body in a uniform stream should feel drag along `+x` at every
/// `eta_p` in `{2.5e-4, 1e-3, 4e-3}`, the sixteenfold range the plan's
/// acceptance criterion sweeps, and the spread `(max - min) / max` across
/// those three drags should stay well clear of the eightfold swing the
/// broken estimator produced over the same range (a correct estimator is
/// exact for the scheme regardless of `eta_p`; the broken one this replaced
/// scaled roughly as `1 / eta_p`).
///
/// At this grid (96 x 48, 8 steps) the spread measured 22.29% when this test
/// was written. The broken estimator was measured at the same grid and step
/// count, from the same states, and gave 89.10%. The tolerance below is set to
/// 35%, a round number with headroom above the first and clear of the second,
/// so the guard passes the corrected estimator and fails the form a regression
/// would reintroduce. Both figures come from this configuration rather than
/// from the full-resolution sweep, whose numbers do not transfer.
#[test]
fn the_body_force_estimate_is_stable_under_a_change_in_eta_p() {
    let drag_at = |eta_p: f64| -> f64 {
        let (s, cx, cy, radius, d) = setup(96, 48);
        let (dx, _dy) = s.spacing();
        let u_mean = 1.0_f64;
        let nu = u_mean * d / 100.0; // Re = 100
        let dt = 0.25 * dx / u_mean;
        let zero = Array2::<Complex<f64>>::zeros((s.nx(), s.ny() / 2 + 1));
        let body = Penalisation::cylinder(&s, cx, cy, radius, eta_p, 20.0, 3.0, 5.0);
        let mut sim = Sim::new(s, zero.clone(), zero.clone(), zero, dt, nu, nu);
        sim.set_mean_flow(u_mean);
        sim.set_body(body);
        for _ in 0..8 {
            sim.step();
        }
        sim.body_force().0
    };
    let drags = [2.5e-4, 1e-3, 4e-3].map(drag_at);
    for (eta_p, fx) in [2.5e-4, 1e-3, 4e-3].iter().zip(drags) {
        assert!(
            fx > 0.0,
            "drag should be positive along +x at eta_p = {eta_p}: {fx}"
        );
    }
    let max = drags.iter().cloned().fold(f64::MIN, f64::max);
    let min = drags.iter().cloned().fold(f64::MAX, f64::min);
    let spread = (max - min) / max;
    println!(
        "eta_p sweep drags: {drags:?}, spread (max - min) / max = {:.4} ({:.2}%)",
        spread,
        100.0 * spread
    );
    assert!(
        spread < 0.35,
        "force estimate depends on eta_p: drags at eta_p in {{2.5e-4, 1e-3, 4e-3}} = {drags:?}, \
         spread {:.2}%",
        100.0 * spread
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
