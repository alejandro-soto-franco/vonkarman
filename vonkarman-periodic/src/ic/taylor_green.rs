use vonkarman_core::field::{GridSpec, VectorField};
use vonkarman_core::float::Float;

/// Taylor-Green vortex initial condition:
///   u = ( sin(x) cos(y) cos(z), -cos(x) sin(y) cos(z), 0 )
///
/// Standard benchmark for pseudospectral NS solvers. At Re = 1600,
/// enstrophy peaks at t ~ 8-9. Energy decays as E(t) = E(0) * exp(-2*nu*t)
/// for short times (before nonlinear effects dominate).
pub fn taylor_green<F: Float>(grid: &GridSpec) -> VectorField<F> {
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

    for i in 0..nx {
        let x = F::from_f64(i as f64) * lx / fnx;
        let (sx, cx) = x.sin_cos();
        for j in 0..ny {
            let y = F::from_f64(j as f64) * ly / fny;
            let (sy, cy) = y.sin_cos();
            for k in 0..nz {
                let z = F::from_f64(k as f64) * lz / fnz;
                let cz = z.cos();
                v.data[0][[i, j, k]] = sx * cy * cz;
                v.data[1][[i, j, k]] = -(cx * sy * cz);
                v.data[2][[i, j, k]] = F::ZERO;
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
    fn taylor_green_divergence_free() {
        let grid = GridSpec::cubic(16, 2.0 * std::f64::consts::PI);
        let v = taylor_green::<f64>(&grid);
        for val in v.z().iter() {
            assert!(val.abs() < 1e-14, "w should be zero, got {val}");
        }
    }

    #[test]
    fn taylor_green_energy() {
        let grid = GridSpec::cubic(32, 2.0 * std::f64::consts::PI);
        let v = taylor_green::<f64>(&grid);
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
        // Analytical: E = 1/8 for u = (sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0) on [0,2pi]^3
        let expected = 0.125;
        assert!(
            (energy - expected).abs() < 1e-6,
            "energy = {energy}, expected {expected}"
        );
    }
}
