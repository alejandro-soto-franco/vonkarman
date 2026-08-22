//! GPU-resident run path: the fast Clay-setting numerics engine.
//!
//! Time-steps entirely on the `ResidentSolver` (nonlinear, FFTs, ETD-RK4, and the
//! CFL timestep all on device), and downloads `u_hat` only at the diagnostics
//! cadence to compute the frame / coherence / pressure diagnostics on the host
//! (`frame_diagnostics_uhat`). This is what makes n=256 tractable: the CPU
//! `Periodic3D` path is FFT-bound (~5 s/step at n=128), the resident path steps
//! ~20x faster. Requires building with `--features cuda` on a nightly toolchain.

use crate::config::ExperimentConfig;
use crate::frame_writer::FrameWriter;
use ndarray::Array3;
use num_complex::Complex;
use std::path::Path;
use std::time::Instant;
use tracing::{info, warn};
use vonkarman_compute::CudaBackend;
use vonkarman_core::domain::Domain;
use vonkarman_core::field::GridSpec;
use vonkarman_core::spectral_ops::SpectralOps;
use vonkarman_fft::BackendMode;
use vonkarman_periodic::{IcType, Periodic3D, ResidentSolver, frame_diagnostics_uhat};

/// Parse the config IC string into an `IcType` (mirrors `run::run`).
fn parse_ic(s: &str) -> Result<IcType, Box<dyn std::error::Error>> {
    Ok(match s {
        "taylor-green" => IcType::TaylorGreen { shift: 0.0 },
        "anti-parallel-tubes" | "anti-parallel" => IcType::AntiParallelTubes {
            circulation: 1.0,
            core_radius: 0.3,
            separation: 1.0,
            perturbation: 0.1,
        },
        "colliding-vortex-rings" | "colliding-rings" => IcType::CollidingVortexRings {
            ring_radius: 1.0,
            core_radius: 0.35,
            circulation: 1.0,
            separation: 2.0,
            axis: 2,
        },
        other => return Err(format!("unknown IC type: {other}").into()),
    })
}

pub fn run_resident(config: &ExperimentConfig) -> Result<(), Box<dyn std::error::Error>> {
    let grid = GridSpec::cubic(config.domain.n, config.domain.l);
    let nu = config.physics.nu;
    let cfl_safety = 0.5;

    let backend = CudaBackend::new(0)
        .map_err(|e| format!("resident GPU path requires a CUDA device: {e}"))?;

    // Seed the initial spectral state through the CPU IC machinery, then hand the
    // flat host `u_hat` to the resident solver (the seed solver is dropped).
    let ic = parse_ic(&config.initial_condition.ic_type)?;
    let seed = Periodic3D::new(grid, nu, ic, BackendMode::Cpu);
    let u0 = seed.u_hat();
    let u_hat_flat: [Vec<Complex<f64>>; 3] = [
        u0[0].iter().copied().collect(),
        u0[1].iter().copied().collect(),
        u0[2].iter().copied().collect(),
    ];
    drop(seed);
    let mut solver = ResidentSolver::new(backend, grid, nu, &u_hat_flat);

    // Host-side machinery for the periodic frame diagnostics. The heavy stepping
    // stays on the GPU; only the diagnostic snapshots come to the host, where the
    // ~22 diagnostic FFTs run in parallel across cores (frame_diagnostics_uhat).
    let ops = SpectralOps::<f64>::new(&grid);
    let (snx, sny, snz) = grid.spectral_shape();
    let cplx_len = snx * sny * snz;
    let mut host: [Vec<Complex<f64>>; 3] = [
        vec![Complex::new(0.0, 0.0); cplx_len],
        vec![Complex::new(0.0, 0.0); cplx_len],
        vec![Complex::new(0.0, 0.0); cplx_len],
    ];
    let to_arr = |host: &[Vec<Complex<f64>>; 3]| -> [Array3<Complex<f64>>; 3] {
        std::array::from_fn(|c| {
            Array3::from_shape_vec((snx, sny, snz), host[c].clone())
                .expect("u_hat flat length matches spectral shape")
        })
    };

    // Output
    let output_dir = Path::new(&config.run.output_dir);
    std::fs::create_dir_all(output_dir)?;
    let frame_enabled = config
        .diagnostics
        .as_ref()
        .map(|d| d.frame_diagnostics)
        .unwrap_or(false);
    let mut frame_writer = if frame_enabled {
        let p = output_dir.join("frame_diagnostics.csv");
        info!(path = %p.display(), "frame diagnostics enabled (resident)");
        Some(FrameWriter::new(&p)?)
    } else {
        None
    };

    let diag_interval = config
        .commit_cycle
        .as_ref()
        .map(|c| c.diagnostics_interval)
        .unwrap_or(1);
    let max_steps = config.termination.max_steps.unwrap_or(u64::MAX);
    let max_time = config.termination.max_time.unwrap_or(f64::INFINITY);
    let max_wall_hours = config.termination.max_wall_hours.unwrap_or(f64::INFINITY);
    let max_vort = config
        .termination
        .max_vorticity_threshold
        .unwrap_or(f64::INFINITY);

    info!(
        name = %config.run.name,
        n = config.domain.n,
        nu = nu,
        re = 1.0 / nu,
        backend = "resident-cuda",
        "starting resident GPU simulation"
    );

    let wall_start = Instant::now();
    let mut time = 0.0_f64;
    let mut step: u64 = 0;

    // Initial diagnostics.
    if frame_enabled {
        solver.download_u_hat_all(&mut host);
        let arr = to_arr(&host);
        let fd = frame_diagnostics_uhat(&arr, &ops, &grid, nu, time, step);
        info!(
            step = 0,
            enstrophy = fd.enstrophy,
            max_vorticity = fd.max_vorticity,
            "initial state"
        );
        if let Some(fw) = frame_writer.as_mut() {
            fw.write_row(&fd)?;
        }
    }

    loop {
        let dt = solver.cfl_dt(cfl_safety, nu);
        solver.step(dt);
        time += dt;
        step += 1;

        if step.is_multiple_of(diag_interval) && frame_enabled {
            solver.download_u_hat_all(&mut host);
            let arr = to_arr(&host);
            let fd = frame_diagnostics_uhat(&arr, &ops, &grid, nu, time, step);
            if let Some(fw) = frame_writer.as_mut() {
                fw.write_row(&fd)?;
            }
            if step.is_multiple_of(100) {
                info!(
                    step = step,
                    time = time,
                    dt = dt,
                    enstrophy = fd.enstrophy,
                    max_vorticity = fd.max_vorticity,
                    "progress"
                );
            }
            if fd.max_vorticity >= max_vort {
                warn!(
                    max_vorticity = fd.max_vorticity,
                    "vorticity threshold exceeded"
                );
                break;
            }
            if fd.enstrophy.is_nan() {
                warn!("NaN detected, aborting");
                break;
            }
        }

        if step >= max_steps {
            info!(step = step, "reached max_steps");
            break;
        }
        if time >= max_time {
            info!(time = time, "reached max_time");
            break;
        }
        if wall_start.elapsed().as_secs_f64() / 3600.0 >= max_wall_hours {
            info!("reached max_wall_hours");
            break;
        }
    }

    if let Some(fw) = frame_writer {
        fw.finish()?;
    }
    info!(
        steps = step,
        time = time,
        wall_secs = wall_start.elapsed().as_secs_f64(),
        "resident simulation complete"
    );
    Ok(())
}
