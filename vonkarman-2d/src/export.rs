//! The `.npy`/`meta.json` frame interchange consumed by the flowforms
//! renderer, and the checkpoint interchange a driver uses to resume an
//! interrupted run.
//!
//! Frames go out as `f32`: the renderer only draws contour levels, and
//! halving the frame size matters across a few thousand frames. Shapes are
//! `(nx, ny)` with axis 0 along `x`, matching the solver's own layout.
//!
//! A checkpoint is solver state, not a rendered artefact, and goes out as
//! `f64`: see [`write_checkpoint`].

use std::path::{Path, PathBuf};

use ndarray::Array2;
use ndarray_npy::write_npy;
use num_complex::Complex;
use serde::Serialize;

type C = Complex<f64>;

/// Run metadata, written before the time loop starts and again when it ends.
///
/// The first write happens before any frame exists, with `frames` set to the
/// *planned* count and `complete: false`, so an interrupted run still leaves
/// a readable `meta.json` rather than none at all. The final write, after
/// the loop exits normally, carries the *actual* count and `complete: true`.
///
/// A consumer that reads `meta.json` while `complete` is `false` is reading
/// a run in progress or one that was interrupted: `frames` there is an upper
/// bound, not a count, since the loop may have written fewer frames than
/// planned (or none) before stopping. Such a consumer should trust the
/// frame files actually present in the output directory over the `frames`
/// field, and use `complete` rather than comparing `frames` against a
/// directory listing to tell a finished run from an interrupted one.
///
/// `complete: true` means only that this process reached its own
/// configured `steps`, not that the physical run a user cares about is
/// finished. Part A of a checkpoint-and-resume split exits normally with
/// `complete: true` and a `frames` count short of the eventual total; part
/// B, resuming, writes its own `complete: true` once it reaches the same
/// `steps` its config names. A consumer only sees the two processes as one
/// continuous run if it already knows the run was split.
#[derive(Debug, Clone, Serialize)]
pub struct Meta {
    /// Grid points along `x`.
    pub nx: usize,
    /// Grid points along `y`.
    pub ny: usize,
    /// Box length along `x`.
    pub lx: f64,
    /// Box length along `y`.
    pub ly: f64,
    /// Uniform stream speed along `+x`.
    pub u_mean: f64,
    /// Body radius.
    pub radius: f64,
    /// Body centre along `x`.
    pub cx: f64,
    /// Body centre along `y`.
    pub cy: f64,
    /// Time step.
    pub dt: f64,
    /// Steps between written frames.
    pub stride: usize,
    /// Reynolds number based on the body diameter.
    pub re: f64,
    /// Number of frames written, or planned to be written while `complete`
    /// is `false`.
    pub frames: usize,
    /// Whether the run reached its final step. `false` from the pre-loop
    /// write until the loop exits normally.
    pub complete: bool,
}

/// Build the `.npy` path for a named field at a given frame index, formatted
/// as a five-digit zero-padded stem (`psi_00007.npy`) so frames sort in order.
fn npy_path(dir: &Path, stem: &str, index: usize) -> PathBuf {
    dir.join(format!("{stem}_{index:05}.npy"))
}

/// Write one frame: streamfunction, vorticity and speed.
///
/// Each field is cast to `f32` before writing, since the renderer only
/// contours the values and never needs `f64` precision. Creates `dir` if it
/// does not already exist.
pub fn write_frame(
    dir: &Path,
    index: usize,
    psi: &Array2<f64>,
    omega: &Array2<f64>,
    speed: &Array2<f64>,
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    for (stem, field) in [("psi", psi), ("omega", omega), ("speed", speed)] {
        write_npy(npy_path(dir, stem, index), &field.mapv(|v| v as f32))
            .map_err(std::io::Error::other)?;
    }
    Ok(())
}

/// Write the body mask, once per run.
pub fn write_mask(dir: &Path, chi: &Array2<f64>) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    write_npy(dir.join("mask.npy"), &chi.mapv(|v| v as f32)).map_err(std::io::Error::other)
}

/// Write the run metadata. Called once before the time loop (planned frame
/// count, `complete: false`) and once after it ends (actual count,
/// `complete: true`); see [`Meta`].
pub fn write_meta(dir: &Path, meta: &Meta) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(meta)?)
}

/// Run parameters recorded in a checkpoint alongside the solver state, so a
/// resume can refuse to continue a different simulation.
///
/// These are exactly the quantities [`crate::Sim::new`] and a driver's own
/// body construction depend on: grid shape and box size (`nx`, `ny`, `lx`,
/// `ly`), the physics (`re`, `u_mean`), the time step (`dt`), and the
/// penalisation constants (`eta_p`, `sigma_max`), plus the output cadence
/// (`stride`, `spin_up`). The cadence fields do not affect the trajectory,
/// but they do affect what `meta.json` claims about it: a resume that
/// changes `stride` mid-run produces frames at an uneven, undocumented
/// spacing while `meta.json` still reports the new, uniform `stride` for
/// the whole run. Anything else a driver derives from these (body centre,
/// radius, fringe extent) is a deterministic function of them, so agreement
/// here is enough to guarantee a rebuilt body matches too.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CheckpointParams {
    pub nx: usize,
    pub ny: usize,
    pub lx: f64,
    pub ly: f64,
    pub re: f64,
    pub u_mean: f64,
    pub dt: f64,
    pub eta_p: f64,
    pub sigma_max: f64,
    pub stride: usize,
    pub spin_up: usize,
}

impl CheckpointParams {
    /// Refuse a resume if `self` (the parameters a checkpoint was written
    /// under) disagrees with `current` (the parameters of the run about to
    /// continue from it).
    ///
    /// Checked in a fixed field order, returning on the first disagreement
    /// so the message names exactly one offending field together with both
    /// values, which is enough to diagnose without re-running anything.
    /// Continuing silently from a checkpoint that does not match is the
    /// worst failure mode available here: it produces output that looks
    /// like a completed run of the current config while actually being a
    /// continuation of a different one.
    pub fn verify(&self, current: CheckpointParams) -> Result<(), String> {
        if self.nx != current.nx {
            return Err(format!(
                "checkpoint field 'nx' disagrees: checkpoint has {}, config has {}",
                self.nx, current.nx
            ));
        }
        if self.ny != current.ny {
            return Err(format!(
                "checkpoint field 'ny' disagrees: checkpoint has {}, config has {}",
                self.ny, current.ny
            ));
        }
        if self.lx != current.lx {
            return Err(format!(
                "checkpoint field 'lx' disagrees: checkpoint has {}, config has {}",
                self.lx, current.lx
            ));
        }
        if self.ly != current.ly {
            return Err(format!(
                "checkpoint field 'ly' disagrees: checkpoint has {}, config has {}",
                self.ly, current.ly
            ));
        }
        if self.re != current.re {
            return Err(format!(
                "checkpoint field 're' disagrees: checkpoint has {}, config has {}",
                self.re, current.re
            ));
        }
        if self.u_mean != current.u_mean {
            return Err(format!(
                "checkpoint field 'u_mean' disagrees: checkpoint has {}, config has {}",
                self.u_mean, current.u_mean
            ));
        }
        if self.dt != current.dt {
            return Err(format!(
                "checkpoint field 'dt' disagrees: checkpoint has {}, config has {}",
                self.dt, current.dt
            ));
        }
        if self.eta_p != current.eta_p {
            return Err(format!(
                "checkpoint field 'eta_p' disagrees: checkpoint has {}, config has {}",
                self.eta_p, current.eta_p
            ));
        }
        if self.sigma_max != current.sigma_max {
            return Err(format!(
                "checkpoint field 'sigma_max' disagrees: checkpoint has {}, config has {}",
                self.sigma_max, current.sigma_max
            ));
        }
        if self.stride != current.stride {
            return Err(format!(
                "checkpoint field 'stride' disagrees: checkpoint has {}, config has {}",
                self.stride, current.stride
            ));
        }
        if self.spin_up != current.spin_up {
            return Err(format!(
                "checkpoint field 'spin_up' disagrees: checkpoint has {}, config has {}",
                self.spin_up, current.spin_up
            ));
        }
        Ok(())
    }
}

/// Magic bytes identifying a `checkpoint.bin` file.
const CHECKPOINT_MAGIC: [u8; 8] = *b"VK2DCKPT";
/// Checkpoint binary layout version. Bump on any layout change and reject
/// mismatches outright, rather than trying to read an old layout with the
/// new offsets. Version 2 added `stride`, `spin_up` and a checksum over the
/// payload alone to version 1's layout. Version 3 widens that checksum to
/// cover the header too (with the checksum field itself zeroed), so a
/// corrupted `step` or `frame` is caught the same way a corrupted payload
/// byte already was; a version-2 checksum, computed over the payload only,
/// left every header field, `step` and `frame` included, free to disagree
/// with what was actually written.
const CHECKPOINT_VERSION: u32 = 3;
/// Fixed header length in bytes: magic, version, `step`, `frame`, `nx`,
/// `ny`, `stride`, `spin_up` (six `u64`), then `lx, ly, re, u_mean, dt,
/// eta_p, sigma_max` (seven `f64`), then the checksum (`u64`).
const CHECKPOINT_HEADER_LEN: usize = 8 + 4 + 8 * 6 + 8 * 7 + 8;
/// Byte offset of the checksum field within the header. The checksum covers
/// `header[..CHECKPOINT_CHECKSUM_OFFSET]` followed by the payload, so the
/// field's own eight bytes are **excluded from the hashed sequence** rather
/// than hashed as zeros. The distinction matters to anyone reimplementing
/// the format: hashing 124 header bytes with eight zeros in place of the
/// checksum gives a different value from hashing the first 116 and then the
/// payload.
const CHECKPOINT_CHECKSUM_OFFSET: usize = CHECKPOINT_HEADER_LEN - 8;

/// FNV-1a, 64-bit, over the concatenation of `chunks` in order. Used as
/// [`write_checkpoint`]/[`read_checkpoint`]'s checksum, over the header
/// (with the checksum field zeroed) and the payload together.
///
/// Not cryptographic: it exists to catch a length-preserving corruption (a
/// flipped byte, a torn write landing on the expected total length by
/// coincidence), which the length check in `read_checkpoint` accepts
/// silently on its own. The production checkpoint lands on btrfs, which
/// checksums data at rest, but the format itself should not depend on the
/// filesystem underneath it for that guarantee.
fn fnv1a64(chunks: &[&[u8]]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for chunk in chunks {
        for &byte in *chunk {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(PRIME);
        }
    }
    hash
}

/// A checkpoint of solver state: the three spectral fields [`crate::Sim`]
/// carries (`wh`, `gh`, `rh`), the step and frame counters, and the run
/// parameters it was written under.
#[derive(Debug)]
pub struct Checkpoint {
    /// Number of completed [`crate::Sim::step`] calls when this checkpoint
    /// was written.
    pub step: usize,
    /// Number of frames written when this checkpoint was written.
    pub frame: usize,
    /// The run parameters this checkpoint was written under. Check against
    /// a resuming config with [`CheckpointParams::verify`] before using
    /// `wh`/`gh`/`rh` to seed a new [`crate::Sim`].
    pub params: CheckpointParams,
    /// Spectral vorticity.
    pub wh: Array2<C>,
    /// Spectral gold dye.
    pub gh: Array2<C>,
    /// Spectral rust dye.
    pub rh: Array2<C>,
}

/// Write a checkpoint of the solver's spectral state to `dir/checkpoint.bin`,
/// atomically and durably.
///
/// Layout: the [`CHECKPOINT_HEADER_LEN`]-byte header (magic, version, `step`,
/// `frame`, `nx`, `ny`, `stride`, `spin_up`, the seven `f64` physics/
/// penalisation fields, then a checksum covering the rest of the header and
/// the payload), followed by `wh`, `gh`, `rh` in that order, each as `nx *
/// (ny / 2 + 1)` complex values in the array's own row-major iteration
/// order, each complex value as two little-endian `f64` (real, then
/// imaginary). Every integer field is little-endian too.
///
/// **The checksum covers the header, not only the payload.** `step` and
/// `frame` carry the whole point of resuming; a checksum over the payload
/// alone would let either flip without detection, since the physics fields
/// are re-checked against the resuming config by [`CheckpointParams::verify`]
/// but `step` and `frame` have nothing else to check them. Computed over
/// the header up to but excluding the checksum field (the header's own
/// last eight bytes, which cannot cover themselves), followed by the
/// payload.
///
/// **`f64`, not `f32`.** Frame export casts to `f32` because the renderer
/// only contours the values; a checkpoint is what re-enters the
/// integrating-factor recursion on resume, and rounding it would mean the
/// resumed trajectory departs from the uninterrupted one at the very first
/// step back.
///
/// **Atomic and durable.** Serialises to a temporary file in `dir` (so the
/// rename below stays on one filesystem), `fsync`s that file, `rename`s it
/// onto `checkpoint.bin`, then `fsync`s the directory. POSIX `rename(2)`
/// within one filesystem replaces the destination in a single
/// directory-entry update: there is no instant at which `checkpoint.bin`
/// exists but holds a partial write, so a crash mid-write leaves the old
/// checkpoint (or none) rather than a truncated one. That ordering
/// guarantee alone is not durability: `rename(2)` orders the directory
/// entry, not the data blocks behind it, so without the two `fsync` calls a
/// power loss or kernel panic straight after a successful `write_checkpoint`
/// can leave `checkpoint.bin` naming an extent whose contents never reached
/// disk. This branch exists because exactly that class of event, an
/// unplanned reboot mid-run, interrupted the run it protects. A crash
/// mid-write does leave the temporary file behind; a later run's
/// `read_checkpoint` never looks at it, since it names the target, not the
/// temporary pattern.
///
/// Materialises the full payload in one `Vec` before writing: at the
/// production grid (`nx = 3072`, `ny = 1536`) that is `wh`, `gh` and `rh`
/// together, about 108 MiB. Fine at that size; a much larger grid would
/// want to stream the arrays out instead of buffering them.
pub fn write_checkpoint(
    dir: &Path,
    step: usize,
    frame: usize,
    params: CheckpointParams,
    wh: &Array2<C>,
    gh: &Array2<C>,
    rh: &Array2<C>,
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let nyh = params.ny / 2 + 1;
    for (name, arr) in [("wh", wh), ("gh", gh), ("rh", rh)] {
        assert_eq!(
            arr.shape(),
            [params.nx, nyh],
            "checkpoint: {name} has shape {:?}, expected [{}, {nyh}] from params (nx={}, ny={})",
            arr.shape(),
            params.nx,
            params.nx,
            params.ny
        );
    }

    let mut payload = Vec::with_capacity(3 * params.nx * nyh * 16);
    for arr in [wh, gh, rh] {
        for c in arr.iter() {
            payload.extend_from_slice(&c.re.to_le_bytes());
            payload.extend_from_slice(&c.im.to_le_bytes());
        }
    }

    // Build the header with the checksum field itself zeroed, so the
    // checksum below can cover the rest of the header (`step`, `frame` and
    // every params field) alongside the payload without covering itself.
    let mut header = Vec::with_capacity(CHECKPOINT_HEADER_LEN);
    header.extend_from_slice(&CHECKPOINT_MAGIC);
    header.extend_from_slice(&CHECKPOINT_VERSION.to_le_bytes());
    header.extend_from_slice(&(step as u64).to_le_bytes());
    header.extend_from_slice(&(frame as u64).to_le_bytes());
    header.extend_from_slice(&(params.nx as u64).to_le_bytes());
    header.extend_from_slice(&(params.ny as u64).to_le_bytes());
    header.extend_from_slice(&(params.stride as u64).to_le_bytes());
    header.extend_from_slice(&(params.spin_up as u64).to_le_bytes());
    for v in [
        params.lx,
        params.ly,
        params.re,
        params.u_mean,
        params.dt,
        params.eta_p,
        params.sigma_max,
    ] {
        header.extend_from_slice(&v.to_le_bytes());
    }
    header.extend_from_slice(&0u64.to_le_bytes()); // checksum placeholder
    debug_assert_eq!(header.len(), CHECKPOINT_HEADER_LEN);
    debug_assert_eq!(
        &header[CHECKPOINT_CHECKSUM_OFFSET..],
        0u64.to_le_bytes().as_slice()
    );

    let checksum = fnv1a64(&[&header[..CHECKPOINT_CHECKSUM_OFFSET], &payload]);

    let mut buf = Vec::with_capacity(CHECKPOINT_HEADER_LEN + payload.len());
    buf.extend_from_slice(&header[..CHECKPOINT_CHECKSUM_OFFSET]);
    buf.extend_from_slice(&checksum.to_le_bytes());
    buf.extend_from_slice(&payload);

    let tmp = dir.join(format!(".checkpoint.{}.tmp", std::process::id()));
    {
        use std::io::Write;
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(&buf)?;
        // Orders the data blocks themselves, which `rename(2)` below does
        // not: see the "Atomic and durable" note above.
        file.sync_all()?;
    }
    std::fs::rename(&tmp, dir.join("checkpoint.bin"))?;
    // The rename is itself a directory-entry write; fsync the directory so
    // that write is durable too, not only the file's data.
    std::fs::File::open(dir)?.sync_all()?;
    Ok(())
}

/// Read the checkpoint written by [`write_checkpoint`] from
/// `dir/checkpoint.bin`.
///
/// Rejects the file outright on a bad magic, an unknown version, a length
/// that disagrees with what the header's own `nx`/`ny` implies (which is
/// what a truncated write looks like), or a header-plus-payload checksum
/// that disagrees with the one stored in the header (a length-preserving
/// corruption, which the length check alone accepts, whether it lands in
/// the payload or in a header field such as `step`), rather than reading
/// past the end or silently zero-filling the gap. Does not check the
/// recorded parameters against a config to resume into: call
/// [`CheckpointParams::verify`] on the returned [`Checkpoint::params`] for
/// that, since this function has no config to compare against.
pub fn read_checkpoint(dir: &Path) -> std::io::Result<Checkpoint> {
    let path = dir.join("checkpoint.bin");
    let buf = std::fs::read(&path)?;

    if buf.len() < CHECKPOINT_HEADER_LEN {
        return Err(std::io::Error::other(format!(
            "checkpoint {path:?} is truncated: {} bytes, the header alone needs {CHECKPOINT_HEADER_LEN}",
            buf.len()
        )));
    }
    if &buf[0..8] != CHECKPOINT_MAGIC.as_slice() {
        return Err(std::io::Error::other(format!(
            "{path:?} is not a vonkarman-2d checkpoint (bad magic bytes)"
        )));
    }
    let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    if version != CHECKPOINT_VERSION {
        return Err(std::io::Error::other(format!(
            "{path:?} is checkpoint format version {version}, this build reads version {CHECKPOINT_VERSION}"
        )));
    }

    let step = u64::from_le_bytes(buf[12..20].try_into().unwrap()) as usize;
    let frame = u64::from_le_bytes(buf[20..28].try_into().unwrap()) as usize;
    let nx = u64::from_le_bytes(buf[28..36].try_into().unwrap()) as usize;
    let ny = u64::from_le_bytes(buf[36..44].try_into().unwrap()) as usize;
    let stride = u64::from_le_bytes(buf[44..52].try_into().unwrap()) as usize;
    let spin_up = u64::from_le_bytes(buf[52..60].try_into().unwrap()) as usize;
    let f64_at = |off: usize| f64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
    let lx = f64_at(60);
    let ly = f64_at(68);
    let re = f64_at(76);
    let u_mean = f64_at(84);
    let dt = f64_at(92);
    let eta_p = f64_at(100);
    let sigma_max = f64_at(108);
    let stored_checksum = u64::from_le_bytes(
        buf[CHECKPOINT_CHECKSUM_OFFSET..CHECKPOINT_CHECKSUM_OFFSET + 8]
            .try_into()
            .unwrap(),
    );

    let nyh = ny / 2 + 1;
    let array_bytes = 3usize
        .checked_mul(nx)
        .and_then(|n| n.checked_mul(nyh))
        .and_then(|n| n.checked_mul(16))
        .ok_or_else(|| {
            std::io::Error::other(format!(
                "checkpoint {path:?} claims a {nx}x{ny} grid; sizing a payload for it would \
                 overflow, so the header is almost certainly corrupt"
            ))
        })?;
    let expected_len = CHECKPOINT_HEADER_LEN
        .checked_add(array_bytes)
        .ok_or_else(|| {
            std::io::Error::other(format!(
                "checkpoint {path:?} claims a {nx}x{ny} grid; sizing a payload for it would \
                 overflow, so the header is almost certainly corrupt"
            ))
        })?;
    if buf.len() != expected_len {
        return Err(std::io::Error::other(format!(
            "checkpoint {path:?} is truncated or corrupt: a {nx}x{ny} grid needs \
             {expected_len} bytes total, found {}",
            buf.len()
        )));
    }
    // Recomputed exactly as write_checkpoint computed it: the header up to
    // but excluding the checksum field, then the payload, skipping over the
    // stored checksum's own bytes rather than reading them as data.
    let actual_checksum = fnv1a64(&[
        &buf[..CHECKPOINT_CHECKSUM_OFFSET],
        &buf[CHECKPOINT_CHECKSUM_OFFSET + 8..],
    ]);
    if actual_checksum != stored_checksum {
        return Err(std::io::Error::other(format!(
            "checkpoint {path:?} fails its checksum (header says {stored_checksum:#x}, \
             computed {actual_checksum:#x}): the length is right but the header or the \
             payload is corrupt"
        )));
    }

    let mut cursor = CHECKPOINT_HEADER_LEN;
    let read_array = |cursor: &mut usize| -> Array2<C> {
        let mut arr = Array2::<C>::zeros((nx, nyh));
        for c in arr.iter_mut() {
            let re_v = f64::from_le_bytes(buf[*cursor..*cursor + 8].try_into().unwrap());
            let im_v = f64::from_le_bytes(buf[*cursor + 8..*cursor + 16].try_into().unwrap());
            *c = C::new(re_v, im_v);
            *cursor += 16;
        }
        arr
    };
    let wh = read_array(&mut cursor);
    let gh = read_array(&mut cursor);
    let rh = read_array(&mut cursor);

    Ok(Checkpoint {
        step,
        frame,
        params: CheckpointParams {
            nx,
            ny,
            lx,
            ly,
            re,
            u_mean,
            dt,
            eta_p,
            sigma_max,
            stride,
            spin_up,
        },
        wh,
        gh,
        rh,
    })
}
