//! Flow past a penalised circular cylinder in a uniform stream, written out as
//! `.npy` frames for the flowforms renderer.
//!
//!   cylinder <config.toml>
//!
//! Config keys are documented on [`Config`]. Set `checkpoint_every` to a
//! positive step count to survive an interruption, and `resume = true` to
//! continue from the checkpoint left in `output_dir` rather than starting
//! over.

use std::path::PathBuf;

use ndarray::Array2;
use num_complex::Complex;
use serde::Deserialize;
use vonkarman_2d::{Penalisation, Sim, Spectral2D, export};

/// Run configuration. Lengths are in body diameters.
#[derive(Debug, Deserialize)]
struct Config {
    /// Grid points along `x`.
    nx: usize,
    /// Grid points along `y`.
    ny: usize,
    /// Box length along `x`, in diameters.
    lx_d: f64,
    /// Box length along `y`, in diameters.
    ly_d: f64,
    /// Reynolds number on the diameter.
    re: f64,
    /// Stream speed.
    u_mean: f64,
    /// Total steps.
    steps: usize,
    /// Steps between written frames.
    stride: usize,
    /// Steps run before the first frame is written, letting the wake develop.
    spin_up: usize,
    /// Penalisation time constant.
    eta_p: f64,
    /// Peak fringe relaxation rate.
    sigma_max: f64,
    /// Where frames go.
    output_dir: PathBuf,
    /// Steps between checkpoint writes. `0` (the default, so an existing
    /// config that omits this key is unaffected) disables checkpointing.
    #[serde(default)]
    checkpoint_every: usize,
    /// Resume from the checkpoint in `output_dir` instead of starting from a
    /// zero initial state. Defaults to `false`, so an existing config that
    /// omits this key starts fresh as before.
    #[serde(default)]
    resume: bool,
}

impl Config {
    /// Checks the invariants nothing else in `main` reproves before using
    /// them. `u_mean` and `re` are divided into below (`dt`, `nu`); zero or
    /// negative either one turns those into an infinity or a NaN and the run
    /// proceeds to write frames of garbage rather than failing. `stride` is
    /// divided into by `usize::is_multiple_of` in the frame-write guard,
    /// which never panics: unlike the plain `%` form it used to be, `stride
    /// = 0` would silently write exactly one frame (at `step == spin_up`)
    /// and then none again, rather than failing loudly. A negative
    /// `sigma_max` makes the fringe factor `exp(-sigma * h)` exceed one, so
    /// the strip amplifies vorticity instead of draining it; zero is allowed
    /// and means no fringe. Grid size and box length are already asserted by
    /// `Spectral2D::new`, so are not repeated here.
    ///
    /// `spin_up` is checked against `steps`: the planned frame count
    /// (written into `meta.json` before the loop starts) is `(steps -
    /// spin_up) / stride + 1`, unchecked `usize` subtraction, which panics
    /// on underflow rather than the loop's own guard (`step >= spin_up`),
    /// which simply never writes a frame when `spin_up` exceeds `steps`.
    /// Asserting here turns that panic into a message that names the actual
    /// offending values instead of an opaque arithmetic overflow.
    ///
    /// `checkpoint_every` and `resume` need no assertion here.
    /// `checkpoint_every = 0` is a valid, meaningful value (checkpointing
    /// disabled), not an error to catch. Whether `resume` is satisfiable
    /// depends on whether a checkpoint file already exists in `output_dir`,
    /// which is filesystem state this function has no access to and cannot
    /// check; `main` checks it directly and fails loudly rather than
    /// silently starting over.
    fn validate(&self) {
        assert!(
            self.u_mean > 0.0,
            "u_mean must be positive, got {}",
            self.u_mean
        );
        assert!(self.re > 0.0, "re must be positive, got {}", self.re);
        assert!(
            self.stride > 0,
            "stride must be positive, got {}",
            self.stride
        );
        assert!(
            self.sigma_max >= 0.0,
            "sigma_max must not be negative, got {}",
            self.sigma_max
        );
        assert!(
            self.spin_up <= self.steps,
            "spin_up ({}) must not exceed steps ({})",
            self.spin_up,
            self.steps
        );
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: cylinder <config.toml>");
    let text = std::fs::read_to_string(&path).expect("read config");
    let cfg: Config = toml::from_str(&text).expect("parse config");
    cfg.validate();

    let d = 1.0_f64;
    let (lx, ly) = (cfg.lx_d * d, cfg.ly_d * d);
    let spec = Spectral2D::new(cfg.nx, cfg.ny, lx, ly);
    let (dx, dy) = spec.spacing();
    let (cx, cy) = (0.25 * lx, 0.5 * ly);
    let radius = 0.5 * d;
    let nu = cfg.u_mean * d / cfg.re;
    // The CFL limit is set by the finer cell axis, so the step follows
    // `dx.min(dy)`. On the square cells every validated run uses, the two agree.
    let dt = 0.25 * dx.min(dy) / cfg.u_mean;

    let params = export::CheckpointParams {
        nx: cfg.nx,
        ny: cfg.ny,
        lx,
        ly,
        re: cfg.re,
        u_mean: cfg.u_mean,
        dt,
        eta_p: cfg.eta_p,
        sigma_max: cfg.sigma_max,
        stride: cfg.stride,
        spin_up: cfg.spin_up,
    };

    // Read and verify the checkpoint, if resuming, before anything below
    // writes into output_dir. A resume config that disagrees with the
    // checkpoint must be refused before the mask and the pre-loop meta.json
    // are written, not after: writing first and refusing second would
    // clobber a completed run's mask.npy and mark it incomplete in
    // meta.json purely from a mistyped resume config, on top of failing to
    // resume at all.
    let resumed = if cfg.resume {
        let checkpoint = export::read_checkpoint(&cfg.output_dir).unwrap_or_else(|e| {
            panic!(
                "resume = true but {:?} has no usable checkpoint: {e}. Refusing to \
                 silently start over: fix output_dir, or set resume = false to start fresh.",
                cfg.output_dir
            )
        });
        if let Err(msg) = checkpoint.params.verify(params) {
            panic!(
                "checkpoint in {:?} does not match this config, refusing to resume: {msg}",
                cfg.output_dir
            );
        }
        // A config whose params agree can still name a `steps` shorter than
        // the checkpoint's own `step`: the loop below is `start_step
        // ..=cfg.steps`, which is empty when `start_step > cfg.steps`, so
        // with no check here the process would run zero steps, exit 0, and
        // overwrite meta.json's `frames`/`complete` to relabel a finished,
        // longer run as this shorter, complete one.
        if checkpoint.step > cfg.steps {
            panic!(
                "checkpoint in {:?} is already at step {} but this config only runs to step \
                 {}, refusing to resume: that would relabel the run already recorded there \
                 as a shorter, complete one instead",
                cfg.output_dir, checkpoint.step, cfg.steps
            );
        }
        Some(checkpoint)
    } else {
        None
    };

    let zero = Array2::<Complex<f64>>::zeros((cfg.nx, cfg.ny / 2 + 1));
    let body = Penalisation::cylinder(
        &spec,
        cx,
        cy,
        radius,
        cfg.eta_p,
        0.85 * lx,
        0.15 * lx,
        cfg.sigma_max,
    );
    let chi = body.chi().clone();
    export::write_mask(&cfg.output_dir, &chi).expect("write mask");

    // Mirrors the loop's own frame-write guard below: a frame is written at
    // step = spin_up, spin_up + stride, ..., up to the largest such step no
    // greater than steps. `Config::validate` asserts `spin_up <= steps`, so
    // this subtraction cannot underflow.
    let planned_frames = (cfg.steps - cfg.spin_up) / cfg.stride + 1;

    // Written now, before the loop runs a single step, so an interrupted run
    // still leaves a readable meta.json instead of none at all: everything
    // here except the frame count is already known, and the frame count is
    // the planned one until the final rewrite at the end of this function
    // marks the run complete. See the doc comment on `export::Meta`.
    export::write_meta(
        &cfg.output_dir,
        &export::Meta {
            nx: cfg.nx,
            ny: cfg.ny,
            lx,
            ly,
            u_mean: cfg.u_mean,
            radius,
            cx,
            cy,
            dt,
            stride: cfg.stride,
            re: cfg.re,
            frames: planned_frames,
            complete: false,
        },
    )
    .expect("write meta");

    let (mut sim, start_step, mut frame) = if let Some(checkpoint) = resumed {
        let mut sim = Sim::new(
            spec,
            checkpoint.wh,
            checkpoint.gh,
            checkpoint.rh,
            dt,
            nu,
            nu,
        );
        sim.set_mean_flow(cfg.u_mean);
        sim.set_body(body);
        (sim, checkpoint.step, checkpoint.frame)
    } else {
        let mut sim = Sim::new(spec, zero.clone(), zero.clone(), zero, dt, nu, nu);
        sim.set_mean_flow(cfg.u_mean);
        sim.set_body(body);
        (sim, 0usize, 0usize)
    };

    let t0 = std::time::Instant::now();
    for step in start_step..=cfg.steps {
        if step >= cfg.spin_up && (step - cfg.spin_up).is_multiple_of(cfg.stride) {
            let (u, v) = sim.total_velocity();
            let mut speed = Array2::<f64>::zeros(u.raw_dim());
            for ((s, &u), &v) in speed.iter_mut().zip(u.iter()).zip(v.iter()) {
                *s = (u * u + v * v).sqrt();
            }
            export::write_frame(
                &cfg.output_dir,
                frame,
                &sim.streamfunction(),
                &sim.vorticity(),
                &speed,
            )
            .expect("write frame");
            frame += 1;
            if frame.is_multiple_of(20) {
                println!("  frame {frame} (step {step}, {:.0?})", t0.elapsed());
            }
        }
        if step < cfg.steps {
            sim.step();
            // Checkpointed right after the step it advances past, so a
            // resume restarting the loop at `step + 1` (see `start_step`
            // above) picks up exactly where this run left off, neither
            // repeating nor skipping this iteration's frame check.
            if cfg.checkpoint_every > 0 && (step + 1).is_multiple_of(cfg.checkpoint_every) {
                export::write_checkpoint(
                    &cfg.output_dir,
                    step + 1,
                    frame,
                    params,
                    &sim.wh,
                    &sim.gh,
                    &sim.rh,
                )
                .expect("write checkpoint");
            }
        }
    }

    export::write_meta(
        &cfg.output_dir,
        &export::Meta {
            nx: cfg.nx,
            ny: cfg.ny,
            lx,
            ly,
            u_mean: cfg.u_mean,
            radius,
            cx,
            cy,
            dt,
            stride: cfg.stride,
            re: cfg.re,
            frames: frame,
            complete: true,
        },
    )
    .expect("write meta");
    println!("done: {frame} frames in {:.0?}", t0.elapsed());
}
