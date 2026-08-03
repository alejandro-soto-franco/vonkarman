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

/// [`Sim::body_force`] against a closed-form value, pinning its magnitude.
///
/// The `eta_p` sweep below checks only sign and flatness, both of which a stub
/// returning a constant satisfies, so it cannot catch a wrong `cell_area`, a
/// missing half substep, a swapped axis pair or any constant factor. The only
/// other magnitude check on the branch lives in the ignored benchmarks, at 50
/// minutes and 2 hours.
///
/// From rest the first half substep acts on `u = u_mean` and `v = 0` exactly,
/// so the momentum it removes is
/// `sum(u_mean * (1 - exp(-chi * h / eta_p))) * dx * dy` with `h = dt/2`, in
/// closed form. `curl` discards the `k = 0` mode, so `u_mean` is restored at
/// full strength before the second half substep, and as `dt` tends to zero the
/// second half removes the same amount. `body_force().0 * dt` divided by that
/// closed form therefore tends to 2.
///
/// Measured at 96 x 48 over 24 d x 12 d, Re 100, `eta_p = 1e-3`, one step:
///
/// ```text
///   cfl 0.250   ratio 1.5886   fy/fx 2.8e-5
///   cfl 0.050   ratio 1.6101   fy/fx 1.8e-5
///   cfl 0.010   ratio 1.7548   fy/fx 3.0e-6
///   cfl 0.002   ratio 1.9269   fy/fx 4.5e-7
/// ```
///
/// This test runs the last row, where the measurement is 1.9269 and
/// 4.526e-7. The band is 1.85 to 2.05, clearing the measurement on both sides,
/// and the symmetry bound is 1e-5, over twentyfold above what was measured. A
/// stub `fn body_force(&self) -> (f64, f64) { (1.0, 0.0) }` gives a ratio of
/// 2.5e-3 here and fails.
#[test]
fn the_body_force_matches_the_closed_form_first_substep() {
    let (s, cx, cy, radius, d) = setup(96, 48);
    let (dx, dy) = s.spacing();
    let u_mean = 1.0_f64;
    let eta_p = 1e-3_f64;
    let nu = u_mean * d / 100.0; // Re = 100
    let cfl = 0.002_f64;
    let dt = cfl * dx / u_mean;
    let zero = Array2::<Complex<f64>>::zeros((s.nx(), s.ny() / 2 + 1));
    let body = Penalisation::cylinder(&s, cx, cy, radius, eta_p, 20.0, 3.0, 5.0);
    let mut sim = Sim::new(s, zero.clone(), zero.clone(), zero, dt, nu, nu);
    sim.set_mean_flow(u_mean);
    sim.set_body(body);

    // Momentum the first half substep removes, in closed form. From rest the
    // state is u = u_mean and v = 0 exactly at every point, so the removed
    // velocity is u_mean * (1 - exp(-chi * h / eta_p)) per point with h = dt/2.
    let h = 0.5 * dt;
    let chi = sim.body().expect("body attached").chi();
    let dp1: f64 = chi
        .iter()
        .map(|&c| u_mean * (1.0 - (-c * h / eta_p).exp()))
        .sum::<f64>()
        * dx
        * dy;

    sim.step();
    let (fx, fy) = sim.body_force();
    let ratio = fx * dt / dp1;
    let symmetry = fy.abs() / fx.abs();
    println!("dp1 = {dp1:.6e}, fx = {fx:.6e}, ratio = {ratio:.4}, fy/fx = {symmetry:.3e}");
    assert!(
        (1.85..=2.05).contains(&ratio),
        "body force over the closed-form first substep is {ratio:.4}, outside 1.85 to 2.05: \
         fx = {fx:.6e}, dt = {dt:.6e}, dp1 = {dp1:.6e}"
    );
    assert!(
        symmetry < 1e-5,
        "transverse force should vanish by symmetry: fy / fx = {symmetry:.3e}"
    );
}

/// Attaching a body clears the force recorded for the previous one.
///
/// `body_force()` gates on a body being attached, not on a step having run
/// with that body, so leaving `last_body_force` in place across `set_body`
/// makes a public accessor report the old body's drag until the next step
/// overwrites it. Measured at 96 x 48 with a radius 2.0 body replaced by a
/// radius 0.1 one, the stale value overstated the new body's drag by 68 times.
#[test]
fn attaching_a_body_clears_the_previous_body_s_force() {
    let (s, cx, cy, _radius, d) = setup(96, 48);
    let (dx, _dy) = s.spacing();
    let u_mean = 1.0_f64;
    let nu = u_mean * d / 100.0; // Re = 100
    let dt = 0.25 * dx / u_mean;
    let zero = Array2::<Complex<f64>>::zeros((s.nx(), s.ny() / 2 + 1));
    let big = Penalisation::cylinder(&s, cx, cy, 2.0, 1e-3, 20.0, 3.0, 5.0);
    let mut sim = Sim::new(s, zero.clone(), zero.clone(), zero, dt, nu, nu);
    sim.set_mean_flow(u_mean);
    sim.set_body(big);
    sim.step();
    let big_force = sim.body_force();
    assert!(big_force.0 > 0.0, "big body drag {:?}", big_force);

    let small = Penalisation::cylinder(&sim.spec, cx, cy, 0.1, 1e-3, 20.0, 3.0, 5.0);
    sim.set_body(small);
    let right_after = sim.body_force();
    println!(
        "after step with the big body: {big_force:?}, immediately after set_body: {right_after:?}"
    );
    assert_eq!(
        right_after,
        (0.0, 0.0),
        "body_force reported {right_after:?} for a body no step has run with"
    );

    sim.step();
    let small_force = sim.body_force();
    println!("after one step with the new body: {small_force:?}");
    assert!(
        small_force.0 > 0.0 && small_force.0 < 0.5 * big_force.0,
        "the new body's drag should be its own and much smaller: {small_force:?} against \
         {big_force:?}"
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
///
/// This checks sign and flatness only.
/// [`the_body_force_matches_the_closed_form_first_substep`] above pins the
/// magnitude, which a stub returning a constant would otherwise slip past.
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

/// The exponential penalisation substep leaves the state close to band-limited.
///
/// The substep multiplies the physical velocity by `exp(-chi * h / eta_p)`,
/// takes the curl and applies the fringe factor, and none of those transforms
/// applies the 2/3 mask. The 2/3 rule the advective term does apply removes
/// aliasing only from a state band-limited to two thirds of `k_max`; content
/// above that cutoff produces aliases that fold back below it, where the mask
/// cannot reach. Masking after the substep is ruled out, since it would
/// truncate the body's vorticity sheet, so the residue is held down by the mask
/// edge width `delta` and by viscosity instead. This measures it, so a change
/// to either trips a guard.
///
/// Measured at 96 x 48 over 24 d x 12 d, Re 100, `eta_p = 1e-3`, `cfl = 0.25`,
/// as the fraction of `sum(|omega_hat|^2)` above the 2/3 cutoff:
///
/// ```text
///   delta = 0.75 dx.max(dy)   step 1  16.13%   step 8   8.69%
///   delta = 0.35 dx.max(dy)   step 1  32.63%   step 8  17.58%
/// ```
///
/// The threshold below is 12% on the step-8 fraction, 38% of headroom above the
/// 8.69% the current edge width produces and well under the 17.58% a mask
/// sharpened to 0.35 cells produces, the second figure measured at this same
/// configuration to confirm the guard discriminates rather than only passing.
/// The grid is rectangular, so this also exercises the dealias mask away from
/// square cells.
#[test]
fn the_penalisation_substep_leaves_the_state_close_to_band_limited() {
    let (s, cx, cy, radius, d) = setup(96, 48);
    let (nx, ny) = (s.nx(), s.ny());
    let (dx, _dy) = s.spacing();
    let u_mean = 1.0_f64;
    let nu = u_mean * d / 100.0; // Re = 100
    let dt = 0.25 * dx / u_mean;
    let zero = Array2::<Complex<f64>>::zeros((nx, ny / 2 + 1));
    let body = Penalisation::cylinder(&s, cx, cy, radius, 1e-3, 20.0, 3.0, 5.0);
    let mut sim = Sim::new(s, zero.clone(), zero.clone(), zero, dt, nu, nu);
    sim.set_mean_flow(u_mean);
    sim.set_body(body);

    // Fraction of the enstrophy sum(|omega_hat|^2) carried above the 2/3
    // cutoff, in mode-index space, matching how Spectral2D builds its mask.
    // Half-spectrum entries other than j = 0 and j = ny/2 stand for a
    // conjugate pair, so they count twice.
    let above_cutoff = |sim: &Sim| -> f64 {
        let (cutx, cuty) = (nx as f64 / 3.0, ny as f64 / 3.0);
        let (mut total, mut above) = (0.0_f64, 0.0_f64);
        for i in 0..nx {
            let m = if i <= nx / 2 {
                i as f64
            } else {
                i as f64 - nx as f64
            };
            for j in 0..(ny / 2 + 1) {
                let weight = if j == 0 || j == ny / 2 { 1.0 } else { 2.0 };
                let e = weight * sim.wh[[i, j]].norm_sqr();
                total += e;
                if m.abs() > cutx || (j as f64) > cuty {
                    above += e;
                }
            }
        }
        above / total
    };

    sim.step();
    let after_1 = above_cutoff(&sim);
    for _ in 0..7 {
        sim.step();
    }
    let after_8 = above_cutoff(&sim);
    println!(
        "above-cutoff enstrophy fraction: step 1 = {:.4}%, step 8 = {:.4}%",
        100.0 * after_1,
        100.0 * after_8
    );
    assert!(
        after_8 < 0.12,
        "penalisation substep left {:.2}% of the enstrophy above the 2/3 cutoff \
         after 8 steps, against 8.69% when this guard was written",
        100.0 * after_8
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
