use crate::config::ExperimentConfig;
use crate::diagnostics_writer::DiagnosticsWriter;
use std::path::Path;
use std::time::Instant;
use tracing::{info, warn};
use vonkarman_core::domain::Domain;
use vonkarman_core::field::GridSpec;
use vonkarman_diag::audit::ConservationAudit;
use vonkarman_diag::scalar::ScalarDiagnostics;
use vonkarman_periodic::{IcType, Periodic3D};

pub fn run(config: &ExperimentConfig) -> Result<(), Box<dyn std::error::Error>> {
    let grid = GridSpec::cubic(config.domain.n, config.domain.l);
    let nu = config.physics.nu;

    info!(
        name = %config.run.name,
        n = config.domain.n,
        nu = nu,
        re = 1.0 / nu,
        "starting simulation"
    );

    // Parse IC type
    let ic = match config.initial_condition.ic_type.as_str() {
        "taylor-green" => IcType::TaylorGreen,
        other => return Err(format!("unknown IC type: {other}").into()),
    };

    let mut solver = Periodic3D::new(grid, nu, ic);

    // Set up output
    let output_dir = Path::new(&config.run.output_dir);
    std::fs::create_dir_all(output_dir)?;
    let parquet_path = output_dir.join("diagnostics.parquet");
    let mut writer = DiagnosticsWriter::new(&parquet_path)?;

    let diag_interval = config
        .commit_cycle
        .as_ref()
        .map(|c| c.diagnostics_interval)
        .unwrap_or(1);

    let audit_enabled = config
        .diagnostics
        .as_ref()
        .map(|d| d.conservation_audit)
        .unwrap_or(true);
    let mut audit = ConservationAudit::new();

    let max_steps = config.termination.max_steps.unwrap_or(u64::MAX);
    let max_time = config.termination.max_time.unwrap_or(f64::INFINITY);
    let max_wall_hours = config.termination.max_wall_hours.unwrap_or(f64::INFINITY);
    let max_vort = config
        .termination
        .max_vorticity_threshold
        .unwrap_or(f64::INFINITY);

    let wall_start = Instant::now();

    // Write initial diagnostics
    let diag = ScalarDiagnostics::from_domain(&solver);
    writer.write_row(&diag)?;
    info!(
        step = 0,
        energy = diag.energy,
        enstrophy = diag.enstrophy,
        max_vorticity = diag.max_vorticity,
        "initial state"
    );

    // Main simulation loop
    loop {
        solver.step();
        let step = solver.step_count();
        let time = solver.time();

        // Diagnostics
        if step % diag_interval == 0 {
            let diag = ScalarDiagnostics::from_domain(&solver);
            writer.write_row(&diag)?;

            if audit_enabled {
                audit.check_diagnostics(&diag, nu);
            }

            if step % 100 == 0 {
                info!(
                    step = step,
                    time = time,
                    dt = solver.dt(),
                    energy = diag.energy,
                    enstrophy = diag.enstrophy,
                    max_vorticity = diag.max_vorticity,
                    "progress"
                );
            }
        }

        // Termination checks
        if step >= max_steps {
            info!(step = step, "reached max_steps");
            break;
        }
        if time >= max_time {
            info!(time = time, "reached max_time");
            break;
        }
        let wall_hours = wall_start.elapsed().as_secs_f64() / 3600.0;
        if wall_hours >= max_wall_hours {
            info!(wall_hours = wall_hours, "reached max_wall_hours");
            break;
        }
        if solver.max_vorticity() >= max_vort {
            warn!(
                max_vorticity = solver.max_vorticity(),
                "vorticity threshold exceeded"
            );
            break;
        }
        if solver.energy().is_nan() {
            warn!("NaN detected in energy, aborting");
            break;
        }
    }

    // Finalize
    writer.finish()?;

    if audit.has_violations() {
        warn!(
            count = audit.violations.len(),
            "conservation violations detected"
        );
        for v in &audit.violations {
            warn!(?v, "violation");
        }
    }

    info!(
        steps = solver.step_count(),
        time = solver.time(),
        wall_secs = wall_start.elapsed().as_secs_f64(),
        "simulation complete"
    );

    Ok(())
}
