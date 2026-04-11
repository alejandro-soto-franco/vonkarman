//! Verifies that checkpoint + restart produces bitwise-identical results
//! to a continuous run.

use vonkarman_core::domain::Domain;
use vonkarman_core::field::GridSpec;
use vonkarman_fft::BackendMode;
use vonkarman_io::{read_checkpoint, write_checkpoint};
use vonkarman_periodic::{IcType, Periodic3D};

#[test]
fn checkpoint_restart_bitwise_match() {
    let dir = std::env::temp_dir().join("vonkarman_restart_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let n = 16;
    let nu = 0.01;
    let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);

    // Run 100 steps continuously
    let mut continuous = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
    for _ in 0..100 {
        continuous.step();
    }

    // Run 50, checkpoint, restart, run 50 more
    let mut first_half = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
    for _ in 0..50 {
        first_half.step();
    }

    let cp_path = dir.join("checkpoint.h5");
    let cp_data = first_half.checkpoint_data();
    write_checkpoint(&cp_path, &cp_data).unwrap();

    let loaded = read_checkpoint(&cp_path).unwrap();
    let mut restarted = Periodic3D::from_checkpoint(loaded, BackendMode::Cpu);

    for _ in 0..50 {
        restarted.step();
    }

    // Bitwise comparison
    let u_cont = continuous.u_hat();
    let u_rest = restarted.u_hat();
    for c in 0..3 {
        assert_eq!(
            u_cont[c], u_rest[c],
            "u_hat[{c}] diverged: continuous vs restarted"
        );
    }

    assert!(
        (continuous.time() - restarted.time()).abs() < 1e-14,
        "time diverged: {} vs {}",
        continuous.time(),
        restarted.time()
    );

    assert_eq!(continuous.step_count(), restarted.step_count());

    let _ = std::fs::remove_dir_all(&dir);
}
