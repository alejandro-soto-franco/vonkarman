use vonkarman_core::domain::Domain;
use vonkarman_core::field::GridSpec;
use vonkarman_fft::BackendMode;
use vonkarman_periodic::{IcType, Periodic3D};

#[test]
fn taylor_green_re100_energy_conservation() {
    let n = 16;
    let nu = 0.01;
    let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
    let mut solver = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);

    let e0 = solver.energy();
    assert!(e0 > 0.0);
    assert!(e0.is_finite());

    let mut prev_e = e0;
    let mut step_count = 0;

    for _ in 0..100 {
        solver.step();
        step_count += 1;
        let e = solver.energy();

        assert!(e.is_finite(), "energy is NaN/Inf at step {step_count}");
        assert!(
            e <= prev_e + 1e-12 * prev_e.abs().max(1e-30),
            "energy increased at step {step_count}: {prev_e} -> {e}"
        );
        prev_e = e;
    }

    let final_e = solver.energy();
    let t = solver.time();

    assert!(
        final_e < 0.9 * e0,
        "energy didn't decay enough: {e0} -> {final_e} at t={t}"
    );

    // Enstrophy positive and finite
    let ens = solver.enstrophy();
    assert!(ens > 0.0);
    assert!(ens.is_finite());

    // Taylor-Green has zero helicity by symmetry
    let hel = solver.helicity();
    assert!(
        hel.abs() < 1e-10,
        "Taylor-Green helicity should be ~0, got {hel}"
    );
}

#[test]
fn taylor_green_diagnostics_consistency() {
    let n = 16;
    let nu = 0.01;
    let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
    let mut solver = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);

    for _ in 0..20 {
        solver.step();
    }

    let e1 = solver.energy();
    let ens = solver.enstrophy();
    let dt = solver.dt();

    solver.step();
    let e2 = solver.energy();

    let computed_rate = (e2 - e1) / dt;
    let expected_rate = -2.0 * nu * ens;

    let rel_err = (computed_rate - expected_rate).abs() / expected_rate.abs().max(1e-30);
    assert!(
        rel_err < 0.5,
        "dissipation rate mismatch: computed={computed_rate}, expected={expected_rate}, rel_err={rel_err}"
    );
}
