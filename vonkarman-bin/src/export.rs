//! Convert spectral HDF5 checkpoints to a VTK ImageData (.vti) per frame plus
//! a PVD time-series, reconstructing physical velocity and vorticity by inverse
//! FFT. This is the bridge that lets external viz tools consume vonkarman runs.

use ndarray::Array3;
use num_complex::Complex;
use std::path::Path;
use vonkarman_core::domain::{DomainType, PhysicsParams, Snapshot};
use vonkarman_core::field::VectorField;
use vonkarman_core::spectral_ops::SpectralOps;
use vonkarman_fft::{BackendMode, FftBackend, create_backend};
use vonkarman_io::{read_checkpoint, write_pvd, write_vti};

/// Export every `checkpoint_*.h5` in `input` (sorted by step) to a
/// `frame_NNNN.vti` per frame plus a `series.pvd` collection in `output`.
pub fn export_vtk(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output)?;

    // Collect and sort checkpoint files. Names are zero-padded
    // (checkpoint_NNNNNNNN.h5), so a lexical sort is also a sort by step.
    let mut paths: Vec<_> = std::fs::read_dir(input)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("checkpoint_") && n.ends_with(".h5"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();

    let mut entries: Vec<(f64, String)> = Vec::new();
    let mut backend: Option<Box<dyn FftBackend<f64>>> = None;
    let mut ops: Option<SpectralOps<f64>> = None;

    for (frame, path) in paths.iter().enumerate() {
        let ckpt = read_checkpoint(path)?;
        let grid = ckpt.grid;
        let u_hat = ckpt.u_hat;

        // The grid is constant across a run, so the FFT backend and spectral
        // operators are built once and reused.
        if backend.is_none() {
            backend = Some(create_backend(grid.nx, grid.ny, grid.nz, BackendMode::Cpu));
            ops = Some(SpectralOps::<f64>::new(&grid));
        }
        let fft = backend.as_ref().unwrap();
        let spectral = ops.as_ref().unwrap();

        // velocity = c2r(u_hat)
        let mut velocity = VectorField::<f64>::zeros(grid);
        for c in 0..3 {
            fft.c2r_3d(&u_hat[c], &mut velocity.data[c]);
        }

        // vorticity = c2r(curl(u_hat))
        let (snx, sny, snz) = grid.spectral_shape();
        let mut omega_hat = [
            Array3::<Complex<f64>>::zeros((snx, sny, snz)),
            Array3::<Complex<f64>>::zeros((snx, sny, snz)),
            Array3::<Complex<f64>>::zeros((snx, sny, snz)),
        ];
        spectral.curl(&u_hat, &mut omega_hat);
        let mut vorticity = VectorField::<f64>::zeros(grid);
        for c in 0..3 {
            fft.c2r_3d(&omega_hat[c], &mut vorticity.data[c]);
        }

        let snapshot = Snapshot::<f64> {
            time: ckpt.time,
            step: ckpt.step_count,
            dt: ckpt.dt,
            velocity,
            vorticity,
            u_hat,
            grid,
            params: PhysicsParams {
                nu: ckpt.nu,
                // re is not persisted in checkpoints; derive from nu as Re = 1/nu
                // (matching the run config convention where nu = 1/Re).
                re: if ckpt.nu > 0.0 { 1.0 / ckpt.nu } else { 0.0 },
                domain: DomainType::Periodic3D,
            },
        };

        let fname = format!("frame_{frame:04}.vti");
        write_vti(&Path::new(output).join(&fname), &snapshot)?;
        entries.push((ckpt.time, fname));
    }

    write_pvd(&Path::new(output).join("series.pvd"), &entries)?;
    println!("exported {} frames to {output}", entries.len());
    Ok(())
}
