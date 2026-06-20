//! VTK XML writers (ImageData .vti and PVD time-series) for visualization.
use base64::Engine;
use ndarray::Array3;
use std::io::Write;
use std::path::Path;
use vonkarman_core::domain::Snapshot;

/// Reorder a C-order (z-fastest) ndarray into VTK point order (x-fastest).
pub fn reorder_zfast_to_xfast(field: &Array3<f64>) -> Vec<f64> {
    let (nx, ny, nz) = field.dim();
    let mut out = vec![0.0; nx * ny * nz];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let p = ix + nx * (iy + ny * iz);
                out[p] = field[[ix, iy, iz]];
            }
        }
    }
    out
}

/// VTK inline-binary encoding: base64 of [u64 LE byte-length][raw LE f64 bytes].
pub fn encode_f64_le(data: &[f64]) -> String {
    let nbytes = (data.len() * 8) as u64;
    let mut buf = Vec::with_capacity(8 + data.len() * 8);
    buf.extend_from_slice(&nbytes.to_le_bytes());
    for &x in data {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(&buf)
}

/// Interleave three scalar buffers (already x-fastest) into VTK vector layout.
fn interleave3(x: &[f64], y: &[f64], z: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n * 3];
    for i in 0..n {
        out[i * 3] = x[i];
        out[i * 3 + 1] = y[i];
        out[i * 3 + 2] = z[i];
    }
    out
}

/// Write a snapshot as a VTK ImageData (.vti) file with velocity, vorticity,
/// and |omega| as point data on the periodic grid.
pub fn write_vti(path: &Path, snapshot: &Snapshot<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let g = &snapshot.grid;
    let (nx, ny, nz) = (g.nx, g.ny, g.nz);
    let (dx, dy, dz) = (g.dx(), g.dy(), g.dz());

    let ux = reorder_zfast_to_xfast(&snapshot.velocity.data[0]);
    let uy = reorder_zfast_to_xfast(&snapshot.velocity.data[1]);
    let uz = reorder_zfast_to_xfast(&snapshot.velocity.data[2]);
    let wx = reorder_zfast_to_xfast(&snapshot.vorticity.data[0]);
    let wy = reorder_zfast_to_xfast(&snapshot.vorticity.data[1]);
    let wz = reorder_zfast_to_xfast(&snapshot.vorticity.data[2]);
    let velocity = interleave3(&ux, &uy, &uz);
    let vorticity = interleave3(&wx, &wy, &wz);
    let omega_mag: Vec<f64> = (0..wx.len())
        .map(|i| (wx[i] * wx[i] + wy[i] * wy[i] + wz[i] * wz[i]).sqrt())
        .collect();

    // WholeExtent counts points: 0..nx etc. is nx+1 samples; we have nx points
    // per axis on the periodic grid, so extent is 0 (nx-1) 0 (ny-1) 0 (nz-1).
    let ex = format!("0 {} 0 {} 0 {}", nx - 1, ny - 1, nz - 1);
    let mut f = std::fs::File::create(path)?;
    write!(
        f,
        r#"<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">
  <ImageData WholeExtent="{ex}" Origin="0 0 0" Spacing="{dx} {dy} {dz}">
    <Piece Extent="{ex}">
      <PointData Vectors="velocity">
"#
    )?;
    writeln!(
        f,
        r#"        <DataArray type="Float64" Name="velocity" NumberOfComponents="3" format="binary">{}</DataArray>"#,
        encode_f64_le(&velocity)
    )?;
    writeln!(
        f,
        r#"        <DataArray type="Float64" Name="vorticity" NumberOfComponents="3" format="binary">{}</DataArray>"#,
        encode_f64_le(&vorticity)
    )?;
    writeln!(
        f,
        r#"        <DataArray type="Float64" Name="omega_mag" NumberOfComponents="1" format="binary">{}</DataArray>"#,
        encode_f64_le(&omega_mag)
    )?;
    write!(
        f,
        r#"      </PointData>
    </Piece>
  </ImageData>
</VTKFile>
"#
    )?;
    Ok(())
}

/// Write a PVD (ParaView Data) time-series collection file that lists
/// a sequence of timesteps and their corresponding relative file paths.
pub fn write_pvd(path: &Path, entries: &[(f64, String)]) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, r#"<?xml version="1.0"?>"#)?;
    writeln!(
        f,
        r#"<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">"#
    )?;
    writeln!(f, r#"  <Collection>"#)?;
    for (t, file) in entries {
        writeln!(
            f,
            r#"    <DataSet timestep="{t}" group="" part="0" file="{file}"/>"#
        )?;
    }
    writeln!(f, r#"  </Collection>"#)?;
    writeln!(f, r#"</VTKFile>"#)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn reorder_puts_x_fastest() {
        // 2x2x2 array where value encodes (ix,iy,iz) as ix*100+iy*10+iz.
        let mut a = Array3::<f64>::zeros((2, 2, 2));
        for ix in 0..2 {
            for iy in 0..2 {
                for iz in 0..2 {
                    a[[ix, iy, iz]] = (ix * 100 + iy * 10 + iz) as f64;
                }
            }
        }
        let flat = reorder_zfast_to_xfast(&a);
        // VTK index p = ix + nx*(iy + ny*iz); x fastest.
        // p=0 -> (0,0,0)=0 ; p=1 -> (1,0,0)=100 ; p=2 -> (0,1,0)=10
        assert_eq!(flat[0], 0.0);
        assert_eq!(flat[1], 100.0);
        assert_eq!(flat[2], 10.0);
        assert_eq!(flat[7], 111.0);
    }

    #[test]
    fn write_vti_produces_parseable_file() {
        use num_complex::Complex;
        use vonkarman_core::domain::{DomainType, PhysicsParams, Snapshot};
        use vonkarman_core::field::{GridSpec, VectorField};
        let grid = GridSpec::cubic(4, std::f64::consts::TAU);
        let zero = || VectorField {
            data: [
                Array3::zeros((4, 4, 4)),
                Array3::zeros((4, 4, 4)),
                Array3::zeros((4, 4, 4)),
            ],
            grid,
        };
        let snap = Snapshot {
            time: 1.5_f64,
            step: 3,
            dt: 0.01,
            velocity: zero(),
            vorticity: zero(),
            u_hat: [
                Array3::<Complex<f64>>::zeros((4, 4, 3)),
                Array3::<Complex<f64>>::zeros((4, 4, 3)),
                Array3::<Complex<f64>>::zeros((4, 4, 3)),
            ],
            grid,
            params: PhysicsParams {
                nu: 1e-3,
                re: 1000.0,
                domain: DomainType::Periodic3D,
            },
        };
        let tmp = std::env::temp_dir().join("ff_test.vti");
        write_vti(&tmp, &snap).unwrap();
        let body = std::fs::read_to_string(&tmp).unwrap();
        assert!(body.contains(r#"WholeExtent="0 3 0 3 0 3""#));
        assert!(body.contains(r#"Name="velocity""#));
    }

    #[test]
    fn pvd_lists_timesteps() {
        let tmp = std::env::temp_dir().join("ff_seq.pvd");
        write_pvd(
            &tmp,
            &[(0.0, "f_0000.vti".into()), (0.5, "f_0001.vti".into())],
        )
        .unwrap();
        let body = std::fs::read_to_string(&tmp).unwrap();
        assert!(body.contains(r#"timestep="0""#));
        assert!(body.contains(r#"file="f_0001.vti""#));
        assert!(body.contains("Collection"));
    }
}
