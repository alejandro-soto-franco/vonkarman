use vonkarman_core::field::{GridSpec, VectorField};
use vonkarman_core::float::Float;

/// Anti-parallel vortex tubes initial condition.
///
/// Two straight vortex tubes aligned along the x-axis, centered at
/// (y0, z0) and (y0, -z0+Lz), with opposite circulation. A sinusoidal
/// perturbation in x drives reconnection.
///
/// This is the Kerr (1993) / Hou-Li (2006) setup for studying
/// vortex reconnection and potential finite-time singularity.
///
/// Parameters:
/// - `circulation`: tube circulation Gamma (total vorticity flux)
/// - `core_radius`: Gaussian core radius sigma
/// - `separation`: distance between tube centers (2*z0)
/// - `perturbation`: amplitude of x-direction sinusoidal perturbation
pub fn anti_parallel_tubes<F: Float>(
    grid: &GridSpec,
    circulation: F,
    core_radius: F,
    separation: F,
    perturbation: F,
) -> VectorField<F> {
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
    let half_ly = ly / F::TWO;
    let half_lz = lz / F::TWO;
    let pi = F::PI;

    let sigma2 = core_radius * core_radius;
    let half_sep = separation / F::TWO;
    // Vorticity amplitude: Gamma / (pi * sigma^2)
    // Build vorticity field (omega_x only for straight tubes along x)
    // then compute velocity via Biot-Savart (stream function approach).
    // For simplicity, we construct velocity directly from the
    // Lamb-Oseen vortex profile for each tube.
    //
    // For a single tube along x at (y_c, z_c) with circulation Gamma:
    //   omega_x = (Gamma / (pi*sigma^2)) * exp(-((y-y_c)^2 + (z-z_c)^2) / sigma^2)
    //   u_y = -(z - z_c) * (Gamma / (2*pi*r^2)) * (1 - exp(-r^2/sigma^2))
    //   u_z =  (y - y_c) * (Gamma / (2*pi*r^2)) * (1 - exp(-r^2/sigma^2))
    //
    // Two tubes: tube 1 at (Ly/2, Lz/2 - d/2) with +Gamma
    //            tube 2 at (Ly/2, Lz/2 + d/2) with -Gamma
    // Plus sinusoidal perturbation in the y-position.

    let two_pi = F::TWO * pi;
    let y_c = half_ly;
    let z_c1 = half_lz - half_sep;
    let z_c2 = half_lz + half_sep;

    for i in 0..nx {
        let x = F::from_f64(i as f64) * lx / fnx;
        // Sinusoidal perturbation of tube centerline in y
        let dy_pert = perturbation * (two_pi * x / lx).sin();

        for j in 0..ny {
            let y_raw = F::from_f64(j as f64) * ly / fny;

            for k in 0..nz {
                let z_raw = F::from_f64(k as f64) * lz / fnz;

                // Tube 1: +Gamma at (y_c + dy_pert, z_c1)
                let dy1 = y_raw - (y_c + dy_pert);
                let dz1 = z_raw - z_c1;
                let r2_1 = dy1 * dy1 + dz1 * dz1;

                // Tube 2: -Gamma at (y_c - dy_pert, z_c2)
                let dy2 = y_raw - (y_c - dy_pert);
                let dz2 = z_raw - z_c2;
                let r2_2 = dy2 * dy2 + dz2 * dz2;

                // Lamb-Oseen velocity: (Gamma / (2*pi*r^2)) * (1 - exp(-r^2/sigma^2))
                let eps = F::from_f64(1e-30);
                let factor1 = if r2_1.to_f64() > 1e-20 {
                    circulation / (two_pi * (r2_1 + eps)) * (F::ONE - (-r2_1 / sigma2).exp())
                } else {
                    F::ZERO
                };
                let factor2 = if r2_2.to_f64() > 1e-20 {
                    circulation / (two_pi * (r2_2 + eps)) * (F::ONE - (-r2_2 / sigma2).exp())
                } else {
                    F::ZERO
                };

                // Tube 1 (+Gamma): u_y = -dz1 * factor1, u_z = +dy1 * factor1
                // Tube 2 (-Gamma): u_y = +dz2 * factor2, u_z = -dy2 * factor2
                v.data[1][[i, j, k]] = -(dz1 * factor1) + dz2 * factor2;
                v.data[2][[i, j, k]] = dy1 * factor1 - dy2 * factor2;
                // u_x = 0 for straight tubes (perturbation is in centerline, not velocity)
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
    fn anti_parallel_finite() {
        let grid = GridSpec::cubic(32, 2.0 * std::f64::consts::PI);
        let v = anti_parallel_tubes::<f64>(
            &grid, 1.0, 0.3, 1.0, 0.1,
        );
        for c in 0..3 {
            for val in v.data[c].iter() {
                assert!(val.is_finite(), "non-finite value in component {c}");
            }
        }
    }

    #[test]
    fn anti_parallel_nonzero_energy() {
        let grid = GridSpec::cubic(32, 2.0 * std::f64::consts::PI);
        let v = anti_parallel_tubes::<f64>(
            &grid, 1.0, 0.3, 1.0, 0.1,
        );
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
    }
}
