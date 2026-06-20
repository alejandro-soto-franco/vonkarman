//! Fixture generator for flowforms integration tests.
//!
//! Usage: cargo run -p vonkarman-io --example fixture -- <output-dir>
//!
//! Writes into <output-dir>:
//!   grid.vti      - single snapshot
//!   f_0000.vti    - frame 0 (timestep 0.0)
//!   f_0001.vti    - frame 1 (timestep 0.5)
//!   seq.pvd       - PVD collection listing f_0000.vti and f_0001.vti

use ndarray::Array3;
use num_complex::Complex;
use std::path::PathBuf;
use vonkarman_core::domain::{DomainType, PhysicsParams, Snapshot};
use vonkarman_core::field::{GridSpec, VectorField};
use vonkarman_io::{write_pvd, write_vti};

fn make_snapshot(time: f64, step: u64) -> Snapshot<f64> {
    let grid = GridSpec::cubic(4, std::f64::consts::TAU);
    let zero_vec = || VectorField {
        data: [
            Array3::zeros((4, 4, 4)),
            Array3::zeros((4, 4, 4)),
            Array3::zeros((4, 4, 4)),
        ],
        grid,
    };
    Snapshot {
        time,
        step,
        dt: 0.01,
        velocity: zero_vec(),
        vorticity: zero_vec(),
        u_hat: [
            Array3::<Complex<f64>>::zeros((4, 4, 3)),
            Array3::<Complex<f64>>::zeros((4, 4, 3)),
            Array3::<Complex<f64>>::zeros((4, 4, 3)),
        ],
        grid,
        params: PhysicsParams {
            nu: 1e-3,
            re: 1000.0,
            domain: DomainType::Periodic3D,
        },
    }
}

fn main() {
    let out_dir: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: fixture <output-dir>")
        .into();

    std::fs::create_dir_all(&out_dir).expect("could not create output dir");

    // Single snapshot
    let snap0 = make_snapshot(0.0, 0);
    write_vti(&out_dir.join("grid.vti"), &snap0).expect("write grid.vti");

    // Two frames for the PVD series
    let snap1 = make_snapshot(0.5, 1);
    write_vti(&out_dir.join("f_0000.vti"), &snap0).expect("write f_0000.vti");
    write_vti(&out_dir.join("f_0001.vti"), &snap1).expect("write f_0001.vti");

    // PVD collection with relative paths
    write_pvd(
        &out_dir.join("seq.pvd"),
        &[(0.0, "f_0000.vti".into()), (0.5, "f_0001.vti".into())],
    )
    .expect("write seq.pvd");

    println!("fixtures written to {}", out_dir.display());
}
