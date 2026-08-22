use vonkarman_core::field::{GridSpec, VectorField};
use vonkarman_core::float::Float;

/// Taylor-Green vortex initial condition:
///   u = ( sin(x) cos(y) cos(z), -cos(x) sin(y) cos(z), 0 )
///
/// Standard benchmark for pseudospectral NS solvers. At Re = 1600,
/// enstrophy peaks at t ~ 8-9. Energy decays as E(t) = E(0) * exp(-2*nu*t)
/// for short times (before nonlinear effects dominate).
///
/// # Vorticity nulls sit on grid nodes
///
/// The vorticity of this datum is
///   omega = ( -cos(x) sin(y) sin(z), -sin(x) cos(y) sin(z), 2 sin(x) sin(y) cos(z) ),
/// which vanishes on the planes `z = 0` and `z = pi` wherever `sin(x)` or
/// `sin(y)` also vanishes, so on the whole lines `x in {0, pi}` and
/// `y in {0, pi}` of those planes. On a uniform periodic mesh with an even
/// `n` those are all mesh points, so a null is evaluated AT the singularity
/// of every direction-field quantity rather than near it.
///
/// What comes back is a plausible number that is entirely quadrature
/// artefact: at 128^3 the geodesic curvature of a vortex line read of order
/// `1e16` at the node-collocated nulls and swamped every quantity weighted
/// by a low power of `|omega|`. A functional's near-null share returned
/// exactly `1.0000`, the diagnostic reading itself.
///
/// [`taylor_green_shifted`] evaluates the same field off the mesh points
/// and removes the collocation. `vonkarman_diag::FrameDiagnostics` reports
/// `min_vorticity`, `null_cell_margin` and `null_fraction` on every frame,
/// so the condition is visible in the output rather than in the reader.
///
/// The hazard generalises to any symmetric datum whose critical points land
/// on mesh points; Taylor-Green is the common one.
pub fn taylor_green<F: Float>(grid: &GridSpec) -> VectorField<F> {
    taylor_green_shifted(grid, 0.0)
}

/// [`taylor_green`] sampled at `x_i = (i + shift) h`, in cells.
///
/// `shift = 0` reproduces the canonical datum exactly, nulls on nodes and
/// all. Any `shift` outside the integers moves every mesh point off the
/// null planes: the field is unchanged, only where it is read moves, so the
/// flow this evolves is the same Taylor-Green vortex translated by a
/// fraction of a cell. A half cell, `shift = 0.5`, puts the mesh as far
/// from the null planes as it goes.
///
/// The energy and the divergence are invariant under the translation, so
/// the analytic checks on the canonical datum hold for any `shift`.
pub fn taylor_green_shifted<F: Float>(grid: &GridSpec, shift: f64) -> VectorField<F> {
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
        let x = F::from_f64(i as f64 + shift) * lx / fnx;
        let (sx, cx) = x.sin_cos();
        for j in 0..ny {
            let y = F::from_f64(j as f64 + shift) * ly / fny;
            let (sy, cy) = y.sin_cos();
            for k in 0..nz {
                let z = F::from_f64(k as f64 + shift) * lz / fnz;
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

    /// The analytic vorticity of the datum, evaluated where the mesh reads
    /// it. Independent of the solver, so this states the collocation as a
    /// property of the initial condition rather than of a transform.
    fn min_vorticity_magnitude(n: usize, shift: f64) -> f64 {
        let h = 2.0 * std::f64::consts::PI / n as f64;
        let mut min = f64::INFINITY;
        for i in 0..n {
            let x = (i as f64 + shift) * h;
            for j in 0..n {
                let y = (j as f64 + shift) * h;
                for k in 0..n {
                    let z = (k as f64 + shift) * h;
                    let wx = -x.cos() * y.sin() * z.sin();
                    let wy = -x.sin() * y.cos() * z.sin();
                    let wz = 2.0 * x.sin() * y.sin() * z.cos();
                    min = min.min((wx * wx + wy * wy + wz * wz).sqrt());
                }
            }
        }
        min
    }

    /// The reported defect. Every direction-field quantity divides by
    /// `|omega|`, so a null AT a mesh point is evaluated at the singularity
    /// and returns quadrature artefact.
    #[test]
    fn the_canonical_datum_puts_a_null_exactly_on_a_node() {
        for n in [16, 32, 64, 128] {
            let min = min_vorticity_magnitude(n, 0.0);
            assert!(
                min < 1e-14,
                "n={n}: expected a node-collocated null, min |omega| = {min:e}"
            );
        }
    }

    /// A half-cell shift moves every mesh point off the null planes, so the
    /// smallest sampled `|omega|` is bounded away from zero by a margin
    /// that shrinks with the mesh rather than vanishing on it.
    #[test]
    fn a_shifted_datum_samples_no_null() {
        for n in [16, 32, 64, 128] {
            let min = min_vorticity_magnitude(n, 0.5);
            assert!(
                min > 1e-6,
                "n={n}: a half-cell shift must clear the nulls, min |omega| = {min:e}"
            );
        }
    }

    /// The shift translates where the field is read and leaves the field
    /// alone, so the analytic energy is unchanged.
    #[test]
    fn a_shift_leaves_the_energy_alone() {
        let grid = GridSpec::cubic(32, 2.0 * std::f64::consts::PI);
        let energy = |shift: f64| {
            let v = taylor_green_shifted::<f64>(&grid, shift);
            let dv = grid.dv();
            let vol = grid.lx * grid.ly * grid.lz;
            let mut e = 0.0_f64;
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        e += (v.x()[[i, j, k]].powi(2)
                            + v.y()[[i, j, k]].powi(2)
                            + v.z()[[i, j, k]].powi(2))
                            * dv;
                    }
                }
            }
            e * 0.5 / vol
        };
        for shift in [0.25, 0.5, 0.75] {
            assert!(
                (energy(shift) - 0.125).abs() < 1e-6,
                "shift {shift}: energy {} != 1/8",
                energy(shift)
            );
        }
    }

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
