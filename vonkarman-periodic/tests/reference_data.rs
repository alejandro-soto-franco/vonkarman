//! Reference data validation against Brachet et al. (1983) and van Rees et al. (2011).
//!
//! Taylor-Green vortex at Re=1600. Checks enstrophy peak timing and magnitude.
//! This test is `#[ignore]` because it takes several minutes at N=128.

use vonkarman_core::domain::Domain;
use vonkarman_core::field::GridSpec;
use vonkarman_fft::BackendMode;
use vonkarman_periodic::{IcType, Periodic3D};

#[test]
#[ignore] // ~5-10 minutes; run with: cargo test --test reference_data -- --ignored --nocapture
fn taylor_green_re1600_reference() {
    let n = 128;
    let nu = 6.25e-4; // Re = 1600
    let t_final = 10.0;
    let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
    let mut solver = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);

    let mut peak_enstrophy = 0.0_f64;
    let mut peak_time = 0.0_f64;
    let mut peak_epsilon = 0.0_f64;
    let mut prev_energy = solver.energy();
    let mut step = 0u64;

    eprintln!("t,energy,enstrophy,epsilon,max_vorticity");

    while solver.time() < t_final {
        solver.step();
        step += 1;
        let e = solver.energy();
        let enstrophy = solver.enstrophy();
        let epsilon = 2.0 * nu * enstrophy;
        let t = solver.time();

        // Check no NaN
        assert!(e.is_finite(), "NaN/Inf energy at step {step}, t={t}");

        // Check no energy increase (within tolerance)
        assert!(
            e <= prev_energy + 1e-12 * prev_energy.abs().max(1e-30),
            "energy increased at step {step}: {prev_energy} -> {e}"
        );

        if enstrophy > peak_enstrophy {
            peak_enstrophy = enstrophy;
            peak_time = t;
            peak_epsilon = epsilon;
        }

        if step.is_multiple_of(500) {
            eprintln!(
                "{t:.4},{e:.8e},{enstrophy:.8e},{epsilon:.8e},{}",
                solver.max_vorticity()
            );
        }

        prev_energy = e;
    }

    eprintln!("\n=== Results (N={n}, Re=1600) ===");
    eprintln!("Peak enstrophy: {peak_enstrophy:.6e} at t={peak_time:.4}");
    eprintln!("Peak dissipation rate: {peak_epsilon:.6e}");
    eprintln!("Final energy: {:.6e}", solver.energy());
    eprintln!("Total steps: {step}");

    // Brachet et al. (1983): enstrophy peaks around t ~ 8-10
    assert!(
        peak_time > 7.0 && peak_time < 11.0,
        "enstrophy peak at t={peak_time}, expected in [7, 11]"
    );

    // Peak dissipation rate: approximately 0.008-0.010 at Re=1600
    // At N=128 we accept wider tolerance since it's under-resolved vs N=512
    assert!(
        peak_epsilon > 0.005 && peak_epsilon < 0.015,
        "peak epsilon={peak_epsilon:.6e}, expected in [0.005, 0.015]"
    );

    // Energy should have decayed significantly
    let final_energy = solver.energy();
    let initial_energy = 0.125; // TG analytical: E0 = 1/8
    assert!(
        final_energy < 0.5 * initial_energy,
        "insufficient energy decay: E0={initial_energy}, E_final={final_energy}"
    );
}
