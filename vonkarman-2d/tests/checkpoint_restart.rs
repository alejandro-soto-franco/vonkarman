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
///
/// Removes any pre-existing directory of the same name before recreating it.
/// The name is already unique per test and per process, so a stale directory
/// here only happens if an earlier run with the same pid left one behind
/// (pids do get reused), but tests such as
/// `write_checkpoint_leaves_no_temporary_file_behind` assert the directory's
/// *entire* contents, so a leftover file from that scenario would corrupt
/// the assertion silently rather than the test catching a real bug.
fn tempdir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("vk2d-checkpoint-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
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
        stride: 4,
        spin_up: 0,
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
    std::fs::write(&path, &full[..40]).unwrap(); // header is 124 bytes

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

/// A length-preserving corruption, a flipped byte deep inside the payload
/// with the total length untouched, passes every truncation check above:
/// `buf.len()` still equals `expected_len`. Before the payload checksum, a
/// checkpoint corrupted this way was read back as though nothing had
/// happened and the resumed run continued from silently wrong state.
#[test]
fn a_checkpoint_with_a_flipped_payload_byte_is_rejected() {
    let (spec, wh, gh, rh) = seed_state();
    let params = params_for(&spec, 1e-3);
    let dir = tempdir("flipped-byte");
    export::write_checkpoint(&dir, 1, 0, params, &wh, &gh, &rh).unwrap();

    let path = dir.join("checkpoint.bin");
    let mut full = std::fs::read(&path).unwrap();
    let corrupt_at = full.len() / 2; // well inside the array payload
    full[corrupt_at] ^= 0xff;
    std::fs::write(&path, &full).unwrap();

    let err = export::read_checkpoint(&dir)
        .expect_err("a checkpoint with a flipped payload byte must be rejected");
    assert!(
        err.to_string().contains("checksum"),
        "error should name the checksum as the problem, got: {err}"
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

/// `stride` and `spin_up` do not feed the trajectory, but a mismatch here
/// still corrupts what `meta.json` claims about the frame cadence: see the
/// note on `CheckpointParams`. Covers both fields in one test since they
/// are the same field class (output cadence), matching the grouping the
/// other refusal tests use for grid/`re`/`dt`/penalisation.
#[test]
fn resume_refuses_a_cadence_mismatch() {
    let (spec, wh, gh, rh) = seed_state();
    let params = params_for(&spec, 1e-3);
    let dir = tempdir("mismatch-cadence");
    export::write_checkpoint(&dir, 1, 0, params, &wh, &gh, &rh).unwrap();
    let checkpoint = export::read_checkpoint(&dir).unwrap();

    let mut stride_mismatch = params;
    stride_mismatch.stride += 3;
    let err = checkpoint
        .params
        .verify(stride_mismatch)
        .expect_err("a stride mismatch must be refused");
    assert!(
        err.contains("stride"),
        "error should name 'stride', got: {err}"
    );

    let mut spin_up_mismatch = params;
    spin_up_mismatch.spin_up += 3;
    let err = checkpoint
        .params
        .verify(spin_up_mismatch)
        .expect_err("a spin_up mismatch must be refused");
    assert!(
        err.contains("spin_up"),
        "error should name 'spin_up', got: {err}"
    );
}

/// `write_checkpoint`'s atomicity claim rests on `checkpoint.bin` only ever
/// being touched by `rename(2)`, never opened and written to directly. The
/// existing round-trip and no-leftover-tmp tests would both still pass a
/// `write_checkpoint` that dropped the temp-file-then-rename path and wrote
/// straight to `checkpoint.bin`, since neither observes *how* the file was
/// replaced, only its final content.
///
/// This test does observe how: it pre-creates `checkpoint.bin` as a
/// read-only file. A direct write opens that existing file and fails with a
/// permission error (file mode restricts the owner too, not just other
/// users). The correct atomic path creates a *new* temp file, unaffected by
/// the target's permissions, and only ever touches `checkpoint.bin` through
/// `rename`, which replaces a directory entry regardless of the old file's
/// mode and so succeeds.
#[test]
fn write_checkpoint_replaces_the_target_via_rename_not_a_direct_write() {
    use std::os::unix::fs::PermissionsExt;

    let (spec, wh, gh, rh) = seed_state();
    let params = params_for(&spec, 1e-3);
    let dir = tempdir("atomic-via-rename");
    let target = dir.join("checkpoint.bin");

    std::fs::write(&target, b"placeholder, not a real checkpoint").unwrap();
    std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o400)).unwrap();

    let result = export::write_checkpoint(&dir, 1, 0, params, &wh, &gh, &rh);

    // Restore write permission unconditionally before any assert below can
    // panic, so a failing run does not leave a read-only file for the OS
    // temp-dir cleanup (or a later test reusing this directory name) to
    // trip over.
    let _ = std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o644));

    result.expect(
        "write_checkpoint must replace checkpoint.bin by renaming a temp file onto it, \
         not by opening and writing to the existing file directly",
    );
    let back = export::read_checkpoint(&dir).unwrap();
    assert_eq!(
        back.wh, wh,
        "the checkpoint read back after the replacement should carry the new content"
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

/// Writes a config for the two end-to-end tests below. `checkpoint_every`
/// and `resume` are the only knobs the two tests vary; everything else is
/// fixed so a straight-through run and a split-and-resumed run are the same
/// simulation.
fn write_e2e_config(
    path: &std::path::Path,
    steps: u64,
    output_dir: &std::path::Path,
    checkpoint_every: u64,
    resume: bool,
) {
    std::fs::write(
        path,
        format!(
            "nx = 16\nny = 8\nlx_d = 4.0\nly_d = 2.0\nre = 100.0\nu_mean = 1.0\n\
             steps = {steps}\nstride = 4\nspin_up = 0\neta_p = 1e-3\nsigma_max = 5.0\n\
             output_dir = \"{}\"\ncheckpoint_every = {checkpoint_every}\nresume = {resume}\n",
            output_dir.display()
        ),
    )
    .unwrap();
}

/// The test the round-1 report said it ran by hand and did not commit: the
/// actual `cylinder` binary, run straight through and separately split
/// across a checkpoint and a resumed process, must leave byte-identical
/// `.npy` files. This is the resume path that actually runs at hour 90 of a
/// production solve, not the pure `export::write_checkpoint`/
/// `read_checkpoint` round trip the library-level tests above cover; those
/// prove the file format is right, not that `main` wires it up correctly.
///
/// With `stride = 4` and `spin_up = 0`, frames land at steps 0, 4, 8, 12.
/// Part A runs to step 6, `checkpoint_every = 6`, so the checkpoint used to
/// resume is written at step 6, strictly between the frame writes at steps
/// 4 and 8, not aligned with either. Part B resumes and runs the remaining
/// steps to the same total as the straight-through run.
///
/// A `write_checkpoint`/`read_checkpoint` pair that only round-trips state
/// exactly (proved above) is not enough by itself: this test additionally
/// exercises `main`'s own bookkeeping around them, `start_step` and `frame`
/// carried out of the checkpoint into the resumed loop, `sim.step()` never
/// re-run or skipped across the split, and the checkpoint read and verified
/// before the resumed process does anything else.
///
/// Byte-comparing final output alone cannot catch every wrong resume here:
/// since this driver's only initial condition is zero, a resume that
/// discards the checkpoint and restarts the loop at step 0 computes the
/// *same* deterministic trajectory from 0 to `steps` that a correct resume
/// does, so the two would coincidentally agree once part B is allowed to
/// run all the way to the same final `steps` — even though that resume
/// silently redid the work the checkpoint existed to save, exactly the
/// four-day loss this branch exists to prevent. To make that observable,
/// the frames part A already wrote (indices 0 and 1, steps 0 and 4) are
/// locked read-only before part B runs. A correct resume starts its loop at
/// step 6 and never re-evaluates the frame-write guard for steps 0 or 4, so
/// it never touches those files. A resume that replays from step 0, whether
/// because it discarded the checkpoint's state or only reset the loop's
/// step/frame counters, tries to rewrite them and is refused by the
/// filesystem, which turns "replayed steps already past the checkpoint"
/// into a failing exit status instead of an accident of matching output.
#[test]
fn a_resumed_run_reproduces_the_straight_through_run_end_to_end() {
    use std::os::unix::fs::PermissionsExt;

    let base = tempdir("e2e-fidelity");
    let straight_dir = base.join("straight");
    let split_dir = base.join("split");
    let total_steps = 12;

    let straight_cfg = base.join("straight.toml");
    write_e2e_config(&straight_cfg, total_steps, &straight_dir, 0, false);
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_cylinder"))
        .arg(&straight_cfg)
        .status()
        .expect("run the cylinder binary straight through");
    assert!(status.success(), "the straight-through run must succeed");

    let part_a_cfg = base.join("part-a.toml");
    write_e2e_config(&part_a_cfg, 6, &split_dir, 6, false);
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_cylinder"))
        .arg(&part_a_cfg)
        .status()
        .expect("run the cylinder binary, part A of the split");
    assert!(status.success(), "part A of the split run must succeed");

    // Frames 0 and 1 (steps 0 and 4) are already on disk from part A. Lock
    // them read-only: see the doc comment above for why.
    let early_frame_files: Vec<std::path::PathBuf> = ["psi", "omega", "speed"]
        .iter()
        .flat_map(|stem| (0..2u32).map(move |idx| format!("{stem}_{idx:05}.npy")))
        .map(|name| split_dir.join(name))
        .collect();
    for path in &early_frame_files {
        let mut perms = std::fs::metadata(path)
            .unwrap_or_else(|e| panic!("stat {path:?} written by part A: {e}"))
            .permissions();
        perms.set_mode(0o400);
        std::fs::set_permissions(path, perms).unwrap();
    }

    let part_b_cfg = base.join("part-b.toml");
    write_e2e_config(&part_b_cfg, total_steps, &split_dir, 6, true);
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cylinder"))
        .arg(&part_b_cfg)
        .output()
        .expect("run the cylinder binary, part B (resume) of the split");

    // Restore write permission unconditionally, before the assert below can
    // panic, so a failing run does not leave read-only files behind for a
    // later test or the OS temp-dir cleanup to trip over.
    for path in &early_frame_files {
        let mut perms = std::fs::metadata(path).unwrap().permissions();
        perms.set_mode(0o644);
        let _ = std::fs::set_permissions(path, perms);
    }

    assert!(
        output.status.success(),
        "part B (resume) of the split run must succeed without rewriting frames already \
         on disk from part A; a correct resume starts at step 6 and never touches them. \
         stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mut straight_files: Vec<String> = std::fs::read_dir(&straight_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".npy"))
        .collect();
    straight_files.sort();
    assert_eq!(
        straight_files.len(),
        13, // mask.npy + (psi, omega, speed) x 4 frames (steps 0, 4, 8, 12)
        "sanity check on the straight-through run's own output: {straight_files:?}"
    );

    for name in &straight_files {
        let a = std::fs::read(straight_dir.join(name))
            .unwrap_or_else(|e| panic!("read {name} from the straight-through run: {e}"));
        let b = std::fs::read(split_dir.join(name)).unwrap_or_else(|e| {
            panic!(
                "{name} exists in the straight-through output but not in the \
                 split/resumed output: {e}"
            )
        });
        assert_eq!(
            a, b,
            "{name} differs between the straight-through run and the checkpoint-resumed run"
        );
    }

    let split_npy_count = std::fs::read_dir(&split_dir)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".npy")
        })
        .count();
    assert_eq!(
        split_npy_count,
        straight_files.len(),
        "the resumed run should write exactly the frames the straight-through run does, no more"
    );
}

/// The refusal path a matching-params fidelity test cannot exercise: since
/// the trajectory test above always resumes into a config that agrees with
/// the checkpoint, a `main` that never calls
/// `CheckpointParams::verify` at all would still pass it, because skipping
/// the check changes nothing when there is nothing to disagree about. This
/// test resumes into a config with a different `re`, through the actual
/// binary, and requires both a non-zero exit and the specific
/// "disagrees"/`re` message `verify` produces, so a resume that pushes
/// mismatched state into a fresh `Sim` instead of refusing (silently
/// continuing a different simulation, the failure mode the brief calls the
/// worst available) fails this test rather than exiting 0 and printing
/// "done: N frames" like any successful run.
#[test]
fn resume_through_the_binary_refuses_a_mismatched_config() {
    let base = tempdir("e2e-mismatch-refusal");
    let dir = base.join("out");

    let write_cfg =
        |path: &std::path::Path, steps: u64, re: f64, checkpoint_every: u64, resume: bool| {
            std::fs::write(
                path,
                format!(
                    "nx = 16\nny = 8\nlx_d = 4.0\nly_d = 2.0\nre = {re}\nu_mean = 1.0\n\
                 steps = {steps}\nstride = 4\nspin_up = 0\neta_p = 1e-3\nsigma_max = 5.0\n\
                 output_dir = \"{}\"\ncheckpoint_every = {checkpoint_every}\nresume = {resume}\n",
                    dir.display()
                ),
            )
            .unwrap();
        };

    let write_cfg_path = base.join("write.toml");
    write_cfg(&write_cfg_path, 6, 100.0, 6, false);
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_cylinder"))
        .arg(&write_cfg_path)
        .status()
        .expect("run the cylinder binary to produce a checkpoint");
    assert!(
        status.success(),
        "the checkpoint-producing run must succeed"
    );

    let resume_cfg_path = base.join("resume.toml");
    write_cfg(&resume_cfg_path, 12, 250.0, 6, true); // re disagrees: 100.0 vs 250.0
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cylinder"))
        .arg(&resume_cfg_path)
        .output()
        .expect("run the cylinder binary attempting a mismatched resume");

    assert!(
        !output.status.success(),
        "a resume whose re disagrees with the checkpoint must fail, not run to completion"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("field 're' disagrees"),
        "the failure should name 're' as the disagreeing field, got stderr: {stderr}"
    );
}
