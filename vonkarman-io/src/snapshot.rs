use hdf5::File as H5File;
use ndarray::Array3;
use std::path::Path;
use vonkarman_core::domain::Snapshot;

/// Metadata read from a snapshot file (without loading full arrays).
#[derive(Debug)]
pub struct SnapshotMetadata {
    pub time: f64,
    pub step: u64,
    pub dt: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub nu: f64,
    pub re: f64,
}

/// Write a full-state snapshot to HDF5.
///
/// File layout:
/// - `/metadata` group: time, step, dt, nx, ny, nz, lx, ly, lz, nu, re
/// - `/velocity/ux`, `/velocity/uy`, `/velocity/uz`: physical-space velocity (f64)
/// - `/vorticity/wx`, `/vorticity/wy`, `/vorticity/wz`: physical-space vorticity (f64)
/// - `/spectral/u_hat_re_0..2`, `/spectral/u_hat_im_0..2`: spectral coefficients
pub fn write_snapshot(
    path: &Path,
    snapshot: &Snapshot<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = H5File::create(path)?;

    // Metadata
    let meta = file.create_group("metadata")?;
    meta.new_attr::<f64>()
        .create("time")?
        .write_scalar(&snapshot.time)?;
    meta.new_attr::<u64>()
        .create("step")?
        .write_scalar(&snapshot.step)?;
    meta.new_attr::<f64>()
        .create("dt")?
        .write_scalar(&snapshot.dt)?;
    meta.new_attr::<u64>()
        .create("nx")?
        .write_scalar(&(snapshot.grid.nx as u64))?;
    meta.new_attr::<u64>()
        .create("ny")?
        .write_scalar(&(snapshot.grid.ny as u64))?;
    meta.new_attr::<u64>()
        .create("nz")?
        .write_scalar(&(snapshot.grid.nz as u64))?;
    meta.new_attr::<f64>()
        .create("lx")?
        .write_scalar(&snapshot.grid.lx)?;
    meta.new_attr::<f64>()
        .create("ly")?
        .write_scalar(&snapshot.grid.ly)?;
    meta.new_attr::<f64>()
        .create("lz")?
        .write_scalar(&snapshot.grid.lz)?;
    meta.new_attr::<f64>()
        .create("nu")?
        .write_scalar(&snapshot.params.nu)?;
    meta.new_attr::<f64>()
        .create("re")?
        .write_scalar(&snapshot.params.re)?;

    // Velocity (physical space)
    let vel = file.create_group("velocity")?;
    let shape = [snapshot.grid.nx, snapshot.grid.ny, snapshot.grid.nz];
    write_array3(&vel, "ux", &snapshot.velocity.data[0], &shape)?;
    write_array3(&vel, "uy", &snapshot.velocity.data[1], &shape)?;
    write_array3(&vel, "uz", &snapshot.velocity.data[2], &shape)?;

    // Vorticity (physical space)
    let vort = file.create_group("vorticity")?;
    write_array3(&vort, "wx", &snapshot.vorticity.data[0], &shape)?;
    write_array3(&vort, "wy", &snapshot.vorticity.data[1], &shape)?;
    write_array3(&vort, "wz", &snapshot.vorticity.data[2], &shape)?;

    // Spectral coefficients (for exact restart)
    let (snx, sny, snz) = snapshot.grid.spectral_shape();
    let spec_shape = [snx, sny, snz];
    let spec = file.create_group("spectral")?;
    for c in 0..3 {
        let re_data: Array3<f64> = snapshot.u_hat[c].mapv(|z| z.re);
        let im_data: Array3<f64> = snapshot.u_hat[c].mapv(|z| z.im);
        write_array3(&spec, &format!("u_hat_re_{c}"), &re_data, &spec_shape)?;
        write_array3(&spec, &format!("u_hat_im_{c}"), &im_data, &spec_shape)?;
    }

    Ok(())
}

/// Read snapshot metadata without loading full arrays.
pub fn read_snapshot_metadata(path: &Path) -> Result<SnapshotMetadata, Box<dyn std::error::Error>> {
    let file = H5File::open(path)?;
    let meta = file.group("metadata")?;

    Ok(SnapshotMetadata {
        time: meta.attr("time")?.read_scalar()?,
        step: meta.attr("step")?.read_scalar()?,
        dt: meta.attr("dt")?.read_scalar()?,
        nx: meta.attr("nx")?.read_scalar::<u64>()? as usize,
        ny: meta.attr("ny")?.read_scalar::<u64>()? as usize,
        nz: meta.attr("nz")?.read_scalar::<u64>()? as usize,
        nu: meta.attr("nu")?.read_scalar()?,
        re: meta.attr("re")?.read_scalar()?,
    })
}

fn write_array3(
    group: &hdf5::Group,
    name: &str,
    data: &Array3<f64>,
    shape: &[usize; 3],
) -> Result<(), Box<dyn std::error::Error>> {
    let ds = group.new_dataset::<f64>().shape(*shape).create(name)?;
    // hdf5-metno uses ndarray 0.15 internally, so write via raw slice
    let slice = data.as_slice().ok_or("array not contiguous")?;
    ds.write_raw(slice)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use vonkarman_core::domain::{DomainType, PhysicsParams};
    use vonkarman_core::field::{GridSpec, VectorField};

    #[test]
    fn write_and_read_snapshot() {
        let dir = std::env::temp_dir().join("vonkarman_hdf5_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("snapshot_test.h5");

        let grid = GridSpec::cubic(8, 2.0 * std::f64::consts::PI);
        let (snx, sny, snz) = grid.spectral_shape();
        let zero = Complex { re: 0.0, im: 0.0 };

        let snapshot = Snapshot {
            time: 1.234,
            step: 100,
            dt: 0.01,
            velocity: VectorField::zeros(grid),
            vorticity: VectorField::zeros(grid),
            u_hat: [
                Array3::from_elem((snx, sny, snz), zero),
                Array3::from_elem((snx, sny, snz), zero),
                Array3::from_elem((snx, sny, snz), zero),
            ],
            grid,
            params: PhysicsParams {
                nu: 0.01,
                re: 100.0,
                domain: DomainType::Periodic3D,
            },
        };

        write_snapshot(&path, &snapshot).unwrap();

        // Read back metadata
        let meta = read_snapshot_metadata(&path).unwrap();
        assert!((meta.time - 1.234).abs() < 1e-10);
        assert_eq!(meta.step, 100);
        assert_eq!(meta.nx, 8);
        assert!((meta.nu - 0.01).abs() < 1e-10);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
