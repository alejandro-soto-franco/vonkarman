//! Integration test for the export-vtk subcommand: author two synthetic
//! spectral checkpoints, export them, and assert the .vti/.pvd outputs.

use ndarray::Array3;
use num_complex::Complex;
use vonkarman_bin::export::export_vtk;
use vonkarman_core::field::GridSpec;
use vonkarman_io::{CheckpointData, write_checkpoint};

fn write_synthetic_checkpoint(path: &std::path::Path, grid: GridSpec, time: f64, step: u64) {
    let (snx, sny, snz) = grid.spectral_shape();
    let mut u_hat = [
        Array3::<Complex<f64>>::zeros((snx, sny, snz)),
        Array3::<Complex<f64>>::zeros((snx, sny, snz)),
        Array3::<Complex<f64>>::zeros((snx, sny, snz)),
    ];
    // A single non-trivial shear mode so curl (hence vorticity) is non-zero:
    // u_x has a kx=1 component, giving omega_z = i*kx*u_y - ... non-trivial.
    u_hat[0][[1, 0, 0]] = Complex { re: 1.0, im: 0.5 };
    u_hat[1][[0, 1, 0]] = Complex { re: 0.3, im: -0.2 };

    let data = CheckpointData {
        u_hat,
        time,
        step_count: step,
        dt: 0.01,
        grid,
        nu: 1e-3,
        config_toml: "[run]\nname = \"export_test\"".to_string(),
    };
    write_checkpoint(path, &data).unwrap();
}

#[test]
fn export_vtk_produces_frames_and_series() {
    let dir = std::env::temp_dir().join("vonkarman_export_vtk_test");
    let _ = std::fs::remove_dir_all(&dir);
    let input = dir.join("ckpts");
    let output = dir.join("vti");
    std::fs::create_dir_all(&input).unwrap();

    let grid = GridSpec::cubic(8, 2.0 * std::f64::consts::PI);
    write_synthetic_checkpoint(&input.join("checkpoint_00000000.h5"), grid, 0.0, 0);
    write_synthetic_checkpoint(&input.join("checkpoint_00000010.h5"), grid, 0.1, 10);

    export_vtk(input.to_str().unwrap(), output.to_str().unwrap()).unwrap();

    // Two frame_*.vti files exist.
    let mut vtis: Vec<_> = std::fs::read_dir(&output)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("frame_") && n.ends_with(".vti"))
                .unwrap_or(false)
        })
        .collect();
    vtis.sort();
    assert_eq!(vtis.len(), 2, "expected two frame_*.vti files");

    // The .vti is valid ImageData and contains a velocity DataArray.
    let body = std::fs::read_to_string(&vtis[0]).unwrap();
    assert!(body.contains("ImageData"), "vti missing ImageData");
    assert!(
        body.contains(r#"Name="velocity""#),
        "vti missing velocity DataArray"
    );

    // series.pvd exists and lists two timesteps mapping to the two frames.
    let pvd = std::fs::read_to_string(output.join("series.pvd")).unwrap();
    assert!(pvd.contains("Collection"), "pvd missing Collection");
    let ndatasets = pvd.matches("<DataSet").count();
    assert_eq!(ndatasets, 2, "pvd should list two timesteps");
    assert!(pvd.contains(r#"file="frame_0000.vti""#));
    assert!(pvd.contains(r#"file="frame_0001.vti""#));

    let _ = std::fs::remove_dir_all(&dir);
}
