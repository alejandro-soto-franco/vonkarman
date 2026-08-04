//! Checkpoint round trip, config-mismatch refusal, corruption rejection, and
//! the property this task exists to prove: a run split by a checkpoint
//! reproduces the uninterrupted run's final state. Also covers the
//! `cylinder` binary's resume-with-no-checkpoint failure and the early
//! `meta.json` write, both driven through the compiled binary since neither
//! is observable from a library-level call into `export`.

use ndarray::Array2;
use num_complex::Complex;
use vonkarman_2d::export::{self, CheckpointParams};
use vonkarman_2d::{Sim, Spectral2D, co_rotating_vortices};

type C = Complex<f64>;

/// A scratch directory unique to the calling test, so parallel `cargo test`
/// runs never collide. Mirrors `export_roundtrip.rs`'s helper.
fn tempdir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("vk2d-checkpoint-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A small, non-trivial spectral state: two co-rotating vortices with dye
/// and band-limited noise, so `wh`, `gh` and `rh` all carry varied non-zero
/// content across many modes. A checkpoint round trip or a restart against a
/// state this rich cannot pass by accident the way it could against an
/// all-zero field, which a `write_checkpoint` that writes zeros or a
/// `read_checkpoint` that ignores the file would reproduce exactly.
///
/// Deterministic: `co_rotating_vortices` takes a fixed seed and no other
/// source of randomness, so two calls with the same grid size produce
/// bit-identical output, which the fidelity test below depends on to build
/// two independent `Sim`s from the same starting state.
fn seed_state() -> (Spectral2D, Array2<C>, Array2<C>, Array2<C>) {
    let spec = Spectral2D::new_square(32);
    let (wh, gh, rh) = co_rotating_vortices(&spec, 1.7, 0.5, 10.0, 0.12, 12345);
    (spec, wh, gh, rh)
}

/// Arbitrary but internally consistent [`CheckpointParams`] for a `spec`,
/// standing in for a driver's config. Physics/penalisation values are not
/// otherwise used by the tests here; only their agreement/disagreement
/// across two [`CheckpointParams`] values is exercised.
fn params_for(spec: &Spectral2D, dt: f64) -> CheckpointParams {
    CheckpointParams {
        nx: spec.nx(),
        ny: spec.ny(),
        lx: spec.lx(),
        ly: spec.ly(),
        re: 100.0,
        u_mean: 1.0,
        dt,
        eta_p: 1e-3,
        sigma_max: 5.0,
    }
}

#[test]
fn checkpoint_round_trips_the_spectral_state_exactly() {
    let (spec, wh, gh, rh) = seed_state();
    let params = params_for(&spec, 1e-3);
    let dir = tempdir("roundtrip");

    export::write_checkpoint(&dir, 7, 3, params, &wh, &gh, &rh).unwrap();
    let back = export::read_checkpoint(&dir).unwrap();

    assert_eq!(back.step, 7);
    assert_eq!(back.frame, 3);
    assert_eq!(back.params, params);
    // Bit-exact against seed_state's varied, non-zero content: a
    // write_checkpoint that wrote zeros, a read_checkpoint that ignored the
    // file and returned a zero state, or a swap of wh/gh/rh, would all fail
    // this, where they would pass against an all-zero or single-value field.
    assert_eq!(back.wh, wh);
    assert_eq!(back.gh, gh);
    assert_eq!(back.rh, rh);
}

/// `write_checkpoint`'s atomicity rests on the temporary file living in the
/// same directory as `checkpoint.bin` (so the rename cannot cross a
/// filesystem) and on nothing else being left behind to confuse a later
/// `read_checkpoint`. A unit test cannot interrupt a write mid-flight to
/// observe the rename itself taking effect atomically (that guarantee is
/// POSIX `rename(2)`'s, documented on `write_checkpoint`), but it can check
/// the part within reach: after a successful write, the directory holds
/// exactly `checkpoint.bin` and no stray `.checkpoint.*.tmp` file, so a
/// later `read_checkpoint` in the same directory has nothing but the
/// intended target to find.
#[test]
fn write_checkpoint_leaves_no_temporary_file_behind() {
    let (spec, wh, gh, rh) = seed_state();
    let params = params_for(&spec, 1e-3);
    let dir = tempdir("no-leftover-tmp");
    export::write_checkpoint(&dir, 1, 0, params, &wh, &gh, &rh).unwrap();

    let names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        vec!["checkpoint.bin".to_string()],
        "a successful write_checkpoint should leave exactly one file behind, got: {names:?}"
    );
}

#[test]
fn a_checkpoint_truncated_below_the_header_is_rejected() {
    let (spec, wh, gh, rh) = seed_state();
    let params = params_for(&spec, 1e-3);
    let dir = tempdir("truncated-header");
    export::write_checkpoint(&dir, 1, 0, params, &wh, &gh, &rh).unwrap();

    let path = dir.join("checkpoint.bin");
    let full = std::fs::read(&path).unwrap();
    std::fs::write(&path, &full[..40]).unwrap(); // header is 100 bytes

    let err =
        export::read_checkpoint(&dir).expect_err("a header-truncated checkpoint must be rejected");
    assert!(
        err.to_string().contains("truncated"),
        "error should say the checkpoint is truncated, got: {err}"
    );
}

#[test]
fn a_checkpoint_truncated_mid_array_is_rejected() {
    let (spec, wh, gh, rh) = seed_state();
    let params = params_for(&spec, 1e-3);
    let dir = tempdir("truncated-array");
    export::write_checkpoint(&dir, 1, 0, params, &wh, &gh, &rh).unwrap();

    let path = dir.join("checkpoint.bin");
    let full = std::fs::read(&path).unwrap();
    assert!(
        full.len() > 300,
        "a 32x32 grid's checkpoint should be far larger than the header"
    );
    // Cuts well past the header (so the array data itself is short) but
    // leaves the header intact and correct, exercising the length check
    // rather than the magic/version checks.
    std::fs::write(&path, &full[..full.len() - 64]).unwrap();

    let err =
        export::read_checkpoint(&dir).expect_err("an array-truncated checkpoint must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("truncated") || msg.contains("corrupt"),
        "error should say the checkpoint is truncated/corrupt, got: {msg}"
    );
}

#[test]
fn resume_accepts_a_checkpoint_written_under_identical_params() {
    let (spec, wh, gh, rh) = seed_state();
    let params = params_for(&spec, 1e-3);
    let dir = tempdir("identical");
    export::write_checkpoint(&dir, 1, 0, params, &wh, &gh, &rh).unwrap();
    let checkpoint = export::read_checkpoint(&dir).unwrap();

    // A control for the four refusal tests below: without this, a
    // `verify` that always returns `Err` would make every one of them pass
    // for the wrong reason.
    checkpoint
        .params
        .verify(params)
        .expect("identical params must be accepted");
}

#[test]
fn resume_refuses_a_grid_mismatch() {
    let (spec, wh, gh, rh) = seed_state();
    let params = params_for(&spec, 1e-3);
    let dir = tempdir("mismatch-grid");
    export::write_checkpoint(&dir, 1, 0, params, &wh, &gh, &rh).unwrap();
    let checkpoint = export::read_checkpoint(&dir).unwrap();

    let mut current = params;
    current.nx += 8;
    let err = checkpoint
        .params
        .verify(current)
        .expect_err("a grid mismatch must be refused");
    assert!(err.contains("nx"), "error should name 'nx', got: {err}");
}

#[test]
fn resume_refuses_a_reynolds_number_mismatch() {
    let (spec, wh, gh, rh) = seed_state();
    let params = params_for(&spec, 1e-3);
    let dir = tempdir("mismatch-re");
    export::write_checkpoint(&dir, 1, 0, params, &wh, &gh, &rh).unwrap();
    let checkpoint = export::read_checkpoint(&dir).unwrap();

    let mut current = params;
    current.re *= 2.0;
    let err = checkpoint
        .params
        .verify(current)
        .expect_err("a re mismatch must be refused");
    assert!(err.contains("re"), "error should name 're', got: {err}");
}

#[test]
fn resume_refuses_a_time_step_mismatch() {
    let (spec, wh, gh, rh) = seed_state();
    let params = params_for(&spec, 1e-3);
    let dir = tempdir("mismatch-dt");
    export::write_checkpoint(&dir, 1, 0, params, &wh, &gh, &rh).unwrap();
    let checkpoint = export::read_checkpoint(&dir).unwrap();

    let mut current = params;
    current.dt *= 2.0;
    let err = checkpoint
        .params
        .verify(current)
        .expect_err("a dt mismatch must be refused");
    assert!(err.contains("dt"), "error should name 'dt', got: {err}");
}

#[test]
fn resume_refuses_a_penalisation_parameter_mismatch() {
    let (spec, wh, gh, rh) = seed_state();
    let params = params_for(&spec, 1e-3);
    let dir = tempdir("mismatch-penalisation");
    export::write_checkpoint(&dir, 1, 0, params, &wh, &gh, &rh).unwrap();
    let checkpoint = export::read_checkpoint(&dir).unwrap();

    let mut current = params;
    current.eta_p *= 10.0;
    let err = checkpoint
        .params
        .verify(current)
        .expect_err("an eta_p mismatch must be refused");
    assert!(
        err.contains("eta_p"),
        "error should name 'eta_p', got: {err}"
    );
}

/// The load-bearing test: a run split by a checkpoint must reproduce the
/// uninterrupted run's final spectral state.
///
/// Three independent `Sim`s are built from three independently-constructed
/// but deterministic `Spectral2D`/IC pairs (`seed_state` is pure), so
/// nothing here shares memory across the "straight through" and "split"
/// paths other than the checkpoint file itself. A `write_checkpoint`/
/// `read_checkpoint` pair that rounds through anything other than the exact
/// bits (truncation to `f32`, a dropped array, a transposed shape) would
/// show up here as a non-zero difference, not just in the dedicated
/// round-trip test above, since it would also perturb every step taken
/// after the resume.
#[test]
fn a_checkpointed_restart_reproduces_the_straight_through_trajectory() {
    let dt = 1e-3;
    let (nu, kappa) = (1e-3, 1e-3);
    let (steps_total, steps_half) = (20, 10);

    let (spec_a, wh0, gh0, rh0) = seed_state();
    let params = params_for(&spec_a, dt);
    let mut straight = Sim::new(spec_a, wh0.clone(), gh0.clone(), rh0.clone(), dt, nu, kappa);
    for _ in 0..steps_total {
        straight.step();
    }

    let (spec_b, wh0b, gh0b, rh0b) = seed_state();
    assert_eq!(wh0b, wh0, "seed_state must be deterministic between calls");
    let mut first_half = Sim::new(spec_b, wh0b, gh0b, rh0b, dt, nu, kappa);
    for _ in 0..steps_half {
        first_half.step();
    }

    let dir = tempdir("restart-fidelity");
    export::write_checkpoint(
        &dir,
        steps_half,
        0,
        params,
        &first_half.wh,
        &first_half.gh,
        &first_half.rh,
    )
    .unwrap();
    drop(first_half);

    let checkpoint = export::read_checkpoint(&dir).unwrap();
    checkpoint
        .params
        .verify(params)
        .expect("the checkpoint should match the params it was written under");

    let (spec_c, _, _, _) = seed_state();
    let mut resumed = Sim::new(
        spec_c,
        checkpoint.wh,
        checkpoint.gh,
        checkpoint.rh,
        dt,
        nu,
        kappa,
    );
    for _ in 0..(steps_total - steps_half) {
        resumed.step();
    }

    let max_diff = |a: &Array2<C>, b: &Array2<C>| -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).norm()))
    };
    let dw = max_diff(&straight.wh, &resumed.wh);
    let dg = max_diff(&straight.gh, &resumed.gh);
    let dr = max_diff(&straight.rh, &resumed.rh);
    let worst = dw.max(dg).max(dr);
    eprintln!(
        "restart fidelity over {steps_total} steps (split {steps_half}/{}): \
         max |wh diff| = {dw:e}, max |gh diff| = {dg:e}, max |rh diff| = {dr:e}, worst = {worst:e}",
        steps_total - steps_half
    );
    assert_eq!(
        worst, 0.0,
        "a checkpointed restart must reproduce the straight-through trajectory bit-exactly \
         (no rayon, no RNG and no other non-determinism sits between Sim::new and Sim::step \
         in this crate); worst observed difference was {worst:e}"
    );
}

#[test]
fn resume_with_no_checkpoint_present_fails_loudly() {
    let dir = tempdir("resume-no-checkpoint");
    let out = dir.join("out");
    let cfg_path = dir.join("config.toml");
    std::fs::write(
        &cfg_path,
        format!(
            "nx = 16\nny = 8\nlx_d = 4.0\nly_d = 2.0\nre = 100.0\nu_mean = 1.0\n\
             steps = 1\nstride = 1\nspin_up = 0\neta_p = 1e-3\nsigma_max = 5.0\n\
             output_dir = \"{}\"\nresume = true\n",
            out.display()
        ),
    )
    .unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cylinder"))
        .arg(&cfg_path)
        .output()
        .expect("run the cylinder binary");

    assert!(
        !output.status.success(),
        "resume = true with no checkpoint present must fail rather than silently start over"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("checkpoint") && stderr.contains("resume"),
        "the failure should name the checkpoint/resume problem, got stderr: {stderr}"
    );
}

/// The other motivating bug: `write_meta` used to run only after the loop,
/// so an interrupted run left no `meta.json` at all. Runs the real binary on
/// a config it cannot finish inside the sleep below (a huge `steps` and a
/// `stride` so large no frame is due yet either), kills it, and checks
/// `meta.json` is already there and honestly marked incomplete.
#[test]
fn meta_json_exists_and_is_readable_before_the_run_completes() {
    let dir = tempdir("meta-early");
    let out = dir.join("out");
    let cfg_path = dir.join("config.toml");
    let (steps, stride) = (50_000_000u64, 1_000_000u64);
    std::fs::write(
        &cfg_path,
        format!(
            "nx = 32\nny = 16\nlx_d = 4.0\nly_d = 2.0\nre = 100.0\nu_mean = 1.0\n\
             steps = {steps}\nstride = {stride}\nspin_up = 0\neta_p = 1e-3\nsigma_max = 5.0\n\
             output_dir = \"{}\"\n",
            out.display()
        ),
    )
    .unwrap();

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_cylinder"))
        .arg(&cfg_path)
        .stdout(std::process::Stdio::null())
        .spawn()
        .expect("spawn the cylinder binary");

    std::thread::sleep(std::time::Duration::from_millis(300));
    // A 32 x 16 grid cannot get anywhere near 50,000,000 IF-RK4 steps (each
    // several FFT calls) in 300ms, so the kill below always lands mid-run;
    // this is a fixed, bounded wait regardless of machine speed, not a race
    // against how far the loop gets.
    child
        .kill()
        .expect("kill the still-running cylinder process");
    let _ = child.wait();

    let meta_path = out.join("meta.json");
    let text = std::fs::read_to_string(&meta_path).unwrap_or_else(|e| {
        panic!("meta.json should already exist after the process was killed mid-run: {e}")
    });
    let meta: serde_json::Value =
        serde_json::from_str(&text).expect("meta.json should parse as JSON");
    assert_eq!(
        meta["complete"], false,
        "a killed-mid-run meta.json must not claim the run completed, got: {meta}"
    );
    let expected_planned = steps / stride + 1;
    assert_eq!(
        meta["frames"].as_u64(),
        Some(expected_planned),
        "the pre-loop write should carry the planned frame count"
    );
}
