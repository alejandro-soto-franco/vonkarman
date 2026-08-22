//! CSV writer for the frame / coherence / pressure diagnostics.
//!
//! Kept separate from the Parquet scalar-diagnostics path: the frame diagnostics are a
//! research instrument (Clifford-NS regularity programme), emitted only when enabled,
//! and CSV keeps them trivially readable for the kappa / closure-margin post-processing.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use vonkarman_diag::FrameDiagnostics;

/// Streams `FrameDiagnostics` rows to a CSV file.
pub struct FrameWriter {
    writer: BufWriter<File>,
    /// Whether the null-collocation warning has already been logged. The
    /// condition is a property of the datum and the mesh, so it holds for
    /// every frame of a run; logging it once says it without burying the
    /// rest of the run's output.
    warned_null_collocation: bool,
}

impl FrameWriter {
    /// Create the file and write the header row.
    pub fn new(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "{}", FrameDiagnostics::csv_header())?;
        Ok(Self {
            writer,
            warned_null_collocation: false,
        })
    }

    /// Append one diagnostics row.
    pub fn write_row(&mut self, d: &FrameDiagnostics) -> Result<(), Box<dyn std::error::Error>> {
        if !self.warned_null_collocation {
            if let Some(msg) = d.null_collocation_warning() {
                tracing::warn!("{msg}");
                self.warned_null_collocation = true;
            }
        }
        writeln!(self.writer, "{}", d.csv_row())?;
        Ok(())
    }

    /// Flush and close.
    pub fn finish(mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.writer.flush()?;
        Ok(())
    }
}
