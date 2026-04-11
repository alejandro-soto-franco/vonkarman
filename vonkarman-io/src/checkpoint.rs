//! Checkpoint I/O for exact solver restart.
//!
//! Writes the full spectral state (u_hat), temporal state (time, step, dt),
//! grid metadata, and physics parameters to HDF5. ETD coefficients are
//! deterministically recomputed on restart, so they are not stored.

use hdf5::File as H5File;
use ndarray::Array3;
use num_complex::Complex;
use std::path::Path;
use vonkarman_core::field::GridSpec;

/// All data needed to restart a solver from a checkpoint.
#[derive(Debug)]
pub struct CheckpointData {
    pub u_hat: [Array3<Complex<f64>>; 3],
    pub time: f64,
    pub step_count: u64,
    pub dt: f64,
    pub grid: GridSpec,
    pub nu: f64,
    pub config_toml: String,
}

/// Write a checkpoint to HDF5.
///
/// File layout:
/// - `/metadata`: time, step_count, dt, nx, ny, nz, lx, ly, lz, nu
/// - `/spectral/u_hat_re_0..2`, `/spectral/u_hat_im_0..2`: spectral coefficients
/// - `/config`: config_toml string as byte dataset
pub fn write_checkpoint(
    path: &Path,
    data: &CheckpointData,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = H5File::create(path)?;

    // Metadata
    let meta = file.create_group("metadata")?;
    meta.new_attr::<f64>()
        .create("time")?
        .write_scalar(&data.time)?;
    meta.new_attr::<u64>()
        .create("step_count")?
        .write_scalar(&data.step_count)?;
    meta.new_attr::<f64>()
        .create("dt")?
        .write_scalar(&data.dt)?;
    meta.new_attr::<u64>()
        .create("nx")?
        .write_scalar(&(data.grid.nx as u64))?;
    meta.new_attr::<u64>()
        .create("ny")?
        .write_scalar(&(data.grid.ny as u64))?;
    meta.new_attr::<u64>()
        .create("nz")?
        .write_scalar(&(data.grid.nz as u64))?;
    meta.new_attr::<f64>()
        .create("lx")?
        .write_scalar(&data.grid.lx)?;
    meta.new_attr::<f64>()
        .create("ly")?
        .write_scalar(&data.grid.ly)?;
    meta.new_attr::<f64>()
        .create("lz")?
        .write_scalar(&data.grid.lz)?;
    meta.new_attr::<f64>()
        .create("nu")?
        .write_scalar(&data.nu)?;

    // Spectral coefficients
    let (snx, sny, snz) = data.grid.spectral_shape();
    let spec_shape = [snx, sny, snz];
    let spec = file.create_group("spectral")?;
    for c in 0..3 {
        let re_data: Array3<f64> = data.u_hat[c].mapv(|z| z.re);
        let im_data: Array3<f64> = data.u_hat[c].mapv(|z| z.im);
        let re_name = format!("u_hat_re_{c}");
        let im_name = format!("u_hat_im_{c}");
        let ds_re = spec
            .new_dataset::<f64>()
            .shape(spec_shape)
            .create(re_name.as_str())?;
        ds_re.write_raw(re_data.as_slice().ok_or("re not contiguous")?)?;
        let ds_im = spec
            .new_dataset::<f64>()
            .shape(spec_shape)
            .create(im_name.as_str())?;
        ds_im.write_raw(im_data.as_slice().ok_or("im not contiguous")?)?;
    }

    // Config TOML string as byte dataset
    let cfg = file.create_group("config")?;
    let toml_bytes = data.config_toml.as_bytes();
    let ds = cfg
        .new_dataset::<u8>()
        .shape([toml_bytes.len()])
        .create("toml")?;
    let toml_arr = ndarray::Array1::from(toml_bytes.to_vec());
    ds.write_raw(toml_arr.view())?;

    Ok(())
}

/// Read a checkpoint from HDF5.
pub fn read_checkpoint(path: &Path) -> Result<CheckpointData, Box<dyn std::error::Error>> {
    let file = H5File::open(path)?;
    let meta = file.group("metadata")?;

    let nx = meta.attr("nx")?.read_scalar::<u64>()? as usize;
    let ny = meta.attr("ny")?.read_scalar::<u64>()? as usize;
    let nz = meta.attr("nz")?.read_scalar::<u64>()? as usize;
    let grid = GridSpec {
        nx,
        ny,
        nz,
        lx: meta.attr("lx")?.read_scalar()?,
        ly: meta.attr("ly")?.read_scalar()?,
        lz: meta.attr("lz")?.read_scalar()?,
    };

    let (snx, sny, snz) = grid.spectral_shape();
    let spec = file.group("spectral")?;
    let mut u_hat = [
        Array3::zeros((snx, sny, snz)),
        Array3::zeros((snx, sny, snz)),
        Array3::zeros((snx, sny, snz)),
    ];
    for c in 0..3 {
        let re_ds = spec.dataset(&format!("u_hat_re_{c}"))?;
        let im_ds = spec.dataset(&format!("u_hat_im_{c}"))?;
        let re_buf: Vec<f64> = re_ds.read_raw()?;
        let im_buf: Vec<f64> = im_ds.read_raw()?;
        for (i, val) in u_hat[c].iter_mut().enumerate() {
            *val = Complex {
                re: re_buf[i],
                im: im_buf[i],
            };
        }
    }

    // Read config TOML
    let cfg = file.group("config")?;
    let toml_ds = cfg.dataset("toml")?;
    let toml_buf: Vec<u8> = toml_ds.read_raw()?;
    let config_toml = String::from_utf8(toml_buf)?;

    Ok(CheckpointData {
        u_hat,
        time: meta.attr("time")?.read_scalar()?,
        step_count: meta.attr("step_count")?.read_scalar()?,
        dt: meta.attr("dt")?.read_scalar()?,
        grid,
        nu: meta.attr("nu")?.read_scalar()?,
        config_toml,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_roundtrip_bitwise() {
        let dir = std::env::temp_dir().join("vonkarman_checkpoint_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("checkpoint.h5");

        let grid = GridSpec::cubic(8, 2.0 * std::f64::consts::PI);
        let (snx, sny, snz) = grid.spectral_shape();

        let mut u_hat = [
            Array3::zeros((snx, sny, snz)),
            Array3::zeros((snx, sny, snz)),
            Array3::zeros((snx, sny, snz)),
        ];
        for c in 0..3 {
            for ((ix, iy, iz), val) in u_hat[c].indexed_iter_mut() {
                *val = Complex {
                    re: (ix + iy + iz + c) as f64 * 0.123,
                    im: (ix * iy + iz + c) as f64 * -0.456,
                };
            }
        }

        let data = CheckpointData {
            u_hat: u_hat.clone(),
            time: 3.14159,
            step_count: 12345,
            dt: 0.00789,
            grid,
            nu: 0.01,
            config_toml: "[run]\nname = \"test\"".to_string(),
        };

        write_checkpoint(&path, &data).unwrap();
        let loaded = read_checkpoint(&path).unwrap();

        assert_eq!(loaded.time, data.time);
        assert_eq!(loaded.step_count, data.step_count);
        assert_eq!(loaded.dt, data.dt);
        assert_eq!(loaded.grid.nx, data.grid.nx);
        assert_eq!(loaded.nu, data.nu);
        assert_eq!(loaded.config_toml, data.config_toml);

        for c in 0..3 {
            assert_eq!(loaded.u_hat[c], data.u_hat[c], "u_hat[{c}] mismatch");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
