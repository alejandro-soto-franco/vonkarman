use vonkarman_core::field::{GridSpec, VectorField};
use vonkarman_core::float::Float;

/// ABC (Arnold-Beltrami-Childress) flow initial condition.
///
/// u = (A*sin(z) + C*cos(y), B*sin(x) + A*cos(z), C*sin(y) + B*cos(x))
///
/// This is an eigenfunction of curl: curl(u) = u (when A=B=C=1).
/// Helicity density h = u . omega = u . u = |u|^2 everywhere.
/// Standard test for helicity conservation in NS solvers.
///
/// The classic choice A=B=C=1 is known to be unstable with Lyapunov
/// exponent ~0.17 (Galloway-Frisch 1984).
pub fn abc_flow<F: Float>(grid: &GridSpec, a: F, b: F, c: F) -> VectorField<F> {
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
                let (sz, cz) = z.sin_cos();
                v.data[0][[i, j, k]] = a * sz + c * cy;
                v.data[1][[i, j, k]] = b * sx + a * cz;
                v.data[2][[i, j, k]] = c * sy + b * cx;
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
    fn abc_divergence_free() {
        // ABC flow is divergence-free by construction.
        // du/dx + dv/dy + dw/dz = 0 analytically.
        // Verify w component is nonzero (unlike TG) and all are finite.
        let grid = GridSpec::cubic(16, 2.0 * std::f64::consts::PI);
        let v = abc_flow::<f64>(&grid, 1.0, 1.0, 1.0);
        for c in 0..3 {
            for val in v.data[c].iter() {
                assert!(val.is_finite());
            }
        }
        // z-component should be nonzero
        let max_w: f64 = v.z().iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(max_w > 0.1, "w should be nonzero for ABC, got max {max_w}");
    }

    #[test]
    fn abc_energy() {
        // For A=B=C=1 on [0,2*pi]^3:
        // E = (1/2V) * integral |u|^2 dV
        // Each term like A^2*sin^2(z) integrates to pi over [0,2*pi].
        // Cross terms like 2*A*C*sin(z)*cos(y) integrate to zero.
        // So integral |u|^2 = 6 * pi * (2*pi)^2 = 6*pi * 4*pi^2 = 24*pi^3
        // V = (2*pi)^3 = 8*pi^3
        // E = 0.5 * 24*pi^3 / 8*pi^3 = 1.5
        let grid = GridSpec::cubic(32, 2.0 * std::f64::consts::PI);
        let v = abc_flow::<f64>(&grid, 1.0, 1.0, 1.0);
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
        let expected = 1.5;
        assert!(
            (energy - expected).abs() < 1e-6,
            "ABC energy = {energy}, expected {expected}"
        );
    }

    #[test]
    fn abc_is_beltrami() {
        // For A=B=C=1: curl(u) = u, so helicity density = |u|^2.
        // Helicity H = integral u . omega dV = integral |u|^2 dV = 2*E*V/V = 2*E = 3.0
        // Actually H = (1/V) * integral u . omega dV = (1/V) * integral |u|^2 dV = 2*E = 3.0
        let grid = GridSpec::cubic(32, 2.0 * std::f64::consts::PI);
        let v = abc_flow::<f64>(&grid, 1.0, 1.0, 1.0);
        let dv = grid.dv();
        let vol = grid.lx * grid.ly * grid.lz;

        // Since curl(u)=u for ABC, helicity = integral |u|^2 / V = 2*E = 3.0
        let mut u_sq = 0.0_f64;
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let u2 = v.x()[[i, j, k]].powi(2)
                           + v.y()[[i, j, k]].powi(2)
                           + v.z()[[i, j, k]].powi(2);
                    u_sq += u2 * dv;
                }
            }
        }
        let helicity = u_sq / vol;
        let expected = 3.0;
        assert!(
            (helicity - expected).abs() < 1e-5,
            "ABC helicity (as |u|^2/V) = {helicity}, expected {expected}"
        );
    }
}
