use vonkarman_core::field::{GridSpec, VectorField};
use vonkarman_core::float::Float;

/// Kida-Pelz high-symmetry initial condition.
///
/// A velocity field with octahedral symmetry group, designed to study
/// vorticity depletion and potential blowup in a maximally symmetric setting.
///
/// u = ( sin(x)(cos(3*y)*cos(z) - cos(y)*cos(3*z)),
///       sin(y)(cos(3*z)*cos(x) - cos(z)*cos(3*x)),
///       sin(z)(cos(3*x)*cos(y) - cos(x)*cos(3*y)) )
///
/// This is divergence-free and has zero helicity by the octahedral symmetry.
/// Reference: Kida (1985), Pelz (2001).
pub fn kida_pelz<F: Float>(grid: &GridSpec) -> VectorField<F> {
    let mut v = VectorField::zeros(*grid);
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let lx = F::from_f64(grid.lx);
    let ly = F::from_f64(grid.ly);
    let lz = F::from_f64(grid.lz);
    let fnx = F::from_f64(nx as f64);
    let fny = F::from_f64(ny as f64);
    let fnz = F::from_f64(nz as f64);
    let three = F::from_f64(3.0);

    for i in 0..nx {
        let x = F::from_f64(i as f64) * lx / fnx;
        let (sx, cx) = x.sin_cos();
        let (_s3x, c3x) = (three * x).sin_cos();
        for j in 0..ny {
            let y = F::from_f64(j as f64) * ly / fny;
            let (sy, cy) = y.sin_cos();
            let (_s3y, c3y) = (three * y).sin_cos();
            for k in 0..nz {
                let z = F::from_f64(k as f64) * lz / fnz;
                let (sz, cz) = z.sin_cos();
                let (_s3z, c3z) = (three * z).sin_cos();

                v.data[0][[i, j, k]] = sx * (c3y * cz - cy * c3z);
                v.data[1][[i, j, k]] = sy * (c3z * cx - cz * c3x);
                v.data[2][[i, j, k]] = sz * (c3x * cy - cx * c3y);
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use vonkarman_core::field::GridSpec;

    #[test]
    fn kida_pelz_finite_and_nonzero() {
        let grid = GridSpec::cubic(32, 2.0 * std::f64::consts::PI);
        let v = kida_pelz::<f64>(&grid);
        let mut max_val = 0.0_f64;
        for c in 0..3 {
            for val in v.data[c].iter() {
                assert!(val.is_finite());
                max_val = max_val.max(val.abs());
            }
        }
        assert!(max_val > 0.1, "field should be nonzero");
    }

    #[test]
    fn kida_pelz_energy() {
        let grid = GridSpec::cubic(32, 2.0 * std::f64::consts::PI);
        let v = kida_pelz::<f64>(&grid);
        let dv = grid.dv();
        let vol = grid.lx * grid.ly * grid.lz;
        let mut energy = 0.0_f64;
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let u2 = v.x()[[i, j, k]].powi(2)
                        + v.y()[[i, j, k]].powi(2)
                        + v.z()[[i, j, k]].powi(2);
                    energy += u2 * dv;
                }
            }
        }
        energy *= 0.5 / vol;
        assert!(energy > 0.0, "energy should be positive, got {energy}");
        // Each component has form sin(a)*(cos(3b)*cos(c) - cos(b)*cos(3c))
        // The cross terms integrate to zero, so we get
        // integral sin^2(x) * (cos^2(3y)*cos^2(z) + cos^2(y)*cos^2(3z)) dx dy dz
        // = pi * (pi * pi + pi * pi) = 2*pi^3
        // Times 3 components: 6*pi^3
        // E = 0.5 * 6*pi^3 / (2*pi)^3 = 0.5 * 6*pi^3 / (8*pi^3) = 3/8 = 0.375
        let expected = 0.375;
        assert!(
            (energy - expected).abs() < 1e-4,
            "energy = {energy}, expected ~{expected}"
        );
    }
}
