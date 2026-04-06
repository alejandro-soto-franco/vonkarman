use arrow::array::Float64Array;
use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use vonkarman_diag::ScalarDiagnostics;

/// Writes scalar diagnostics to a Parquet file, one row per timestep.
pub struct DiagnosticsWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
    /// Buffer rows before flushing.
    buffer: Vec<ScalarDiagnostics>,
    /// Flush every N rows.
    flush_interval: usize,
}

impl DiagnosticsWriter {
    pub fn new(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("step", DataType::UInt64, false),
            Field::new("time", DataType::Float64, false),
            Field::new("dt", DataType::Float64, false),
            Field::new("energy", DataType::Float64, false),
            Field::new("enstrophy", DataType::Float64, false),
            Field::new("helicity", DataType::Float64, false),
            Field::new("superhelicity", DataType::Float64, false),
            Field::new("max_vorticity", DataType::Float64, false),
            Field::new("energy_dissipation_rate", DataType::Float64, false),
            Field::new("helicity_dissipation_rate", DataType::Float64, false),
            Field::new("cfl_number", DataType::Float64, false),
        ]));

        let file = File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .build();
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        Ok(Self {
            writer,
            schema,
            buffer: Vec::with_capacity(1000),
            flush_interval: 1000,
        })
    }

    pub fn write_row(
        &mut self,
        diag: &ScalarDiagnostics,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.buffer.push(diag.clone());
        if self.buffer.len() >= self.flush_interval {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(UInt64Array::from(
                    self.buffer.iter().map(|d| d.step).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    self.buffer.iter().map(|d| d.time).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    self.buffer.iter().map(|d| d.dt).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    self.buffer.iter().map(|d| d.energy).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    self.buffer.iter().map(|d| d.enstrophy).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    self.buffer.iter().map(|d| d.helicity).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    self.buffer
                        .iter()
                        .map(|d| d.superhelicity)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    self.buffer
                        .iter()
                        .map(|d| d.max_vorticity)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    self.buffer
                        .iter()
                        .map(|d| d.energy_dissipation_rate)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    self.buffer
                        .iter()
                        .map(|d| d.helicity_dissipation_rate)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    self.buffer.iter().map(|d| d.cfl_number).collect::<Vec<_>>(),
                )),
            ],
        )?;
        self.writer.write(&batch)?;
        self.buffer.clear();
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.flush()?;
        self.writer.close()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vonkarman_diag::ScalarDiagnostics;

    #[test]
    fn write_and_verify_parquet() {
        let dir = std::env::temp_dir().join("vonkarman_test_parquet");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("diagnostics.parquet");

        let mut writer = DiagnosticsWriter::new(&path).unwrap();

        for i in 0..3 {
            let d = ScalarDiagnostics {
                time: i as f64 * 0.1,
                step: i,
                dt: 0.01,
                energy: 1.0 - i as f64 * 0.05,
                enstrophy: 2.0 + i as f64 * 0.1,
                helicity: 0.0,
                superhelicity: 0.0,
                max_vorticity: 5.0,
                energy_dissipation_rate: -0.04,
                helicity_dissipation_rate: 0.0,
                cfl_number: 0.3,
            };
            writer.write_row(&d).unwrap();
        }
        writer.finish().unwrap();

        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
