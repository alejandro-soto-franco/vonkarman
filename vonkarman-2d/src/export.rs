//! The `.npy` frame interchange consumed by the flowforms renderer.
//!
//! Fields go out as `f32`: the renderer only draws contour levels, and halving
//! the frame size matters across a few thousand frames. Shapes are `(nx, ny)`
//! with axis 0 along `x`, matching the solver's own layout.

use std::path::{Path, PathBuf};

use ndarray::Array2;
use ndarray_npy::write_npy;
use serde::Serialize;

/// Run metadata, written once as `meta.json`.
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
    /// Number of frames written.
    pub frames: usize,
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

/// Write the run metadata, once per run.
pub fn write_meta(dir: &Path, meta: &Meta) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(meta)?)
}
