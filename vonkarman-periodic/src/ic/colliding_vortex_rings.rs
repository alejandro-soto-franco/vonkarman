use ndarray::Array3;
use num_complex::Complex;
use vonkarman_core::field::{GridSpec, VectorField};
use vonkarman_core::spectral_ops::SpectralOps;
use vonkarman_fft::FftBackend;

/// Two vortex rings placed head-on along `axis`, set to approach and collide
/// near the box center.
///
/// A vortex ring is a torus of azimuthal vorticity. The velocity of a single
/// ring is recovered from its vorticity by inverting the curl in spectral
/// space (`u_hat = i k x omega_hat / |k|^2`), which is divergence-free by
/// construction. Two rings are summed in vorticity before the inversion.
///
/// Parameters:
/// - `ring_radius` R: radius of each ring centerline circle.
/// - `core_radius` sigma: Gaussian core thickness of the vortex tube.
/// - `circulation` Gamma: circulation of each ring.
/// - `separation`: initial distance between the two ring centers along `axis`.
/// - `axis`: propagation axis (0 = x, 1 = y, 2 = z).
///
/// The two rings carry opposite-signed circulation so both self-propel toward
/// the box center (a head-on collision). If a smoke run shows the rings flying
/// apart instead of colliding, swap the sign of `s1`/`s2` below.
pub fn colliding_vortex_rings(
    grid: &GridSpec,
    ring_radius: f64,
    core_radius: f64,
    circulation: f64,
    separation: f64,
    axis: usize,
    fft: &dyn FftBackend<f64>,
) -> VectorField<f64> {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let l = [grid.lx, grid.ly, grid.lz];
    let center = [l[0] / 2.0, l[1] / 2.0, l[2] / 2.0];

    // Unit axis vector e_a.
    let mut e_a = [0.0_f64; 3];
    e_a[axis] = 1.0;

    // Two ring centers, offset along the axis by +/- separation/2.
    let half_sep = separation / 2.0;
    let mut c1 = center;
    let mut c2 = center;
    c1[axis] = center[axis] - half_sep;
    c2[axis] = center[axis] + half_sep;

    // Circulation signs chosen so both rings propagate toward the center.
    // Ring 1 (below center) propagates +axis; ring 2 (above) propagates -axis.
    let s1 = 1.0_f64;
    let s2 = -1.0_f64;

    let sigma2 = core_radius * core_radius;
    let amp = circulation / (std::f64::consts::PI * sigma2);

    // Build physical-space vorticity (3 components).
    let mut omega: [Array3<f64>; 3] = [
        Array3::zeros((nx, ny, nz)),
        Array3::zeros((nx, ny, nz)),
        Array3::zeros((nx, ny, nz)),
    ];

    // Azimuthal-vorticity contribution of one ring at center `c` with sign `s`.
    // Returns the 3 vorticity components at grid point `p`.
    let ring_omega = |p: [f64; 3], c: [f64; 3], s: f64| -> [f64; 3] {
        let r = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        // Axial coordinate z = r . e_a.
        let z = r[0] * e_a[0] + r[1] * e_a[1] + r[2] * e_a[2];
        // Radial vector rho_vec = r - z e_a.
        let rho_vec = [r[0] - z * e_a[0], r[1] - z * e_a[1], r[2] - z * e_a[2]];
        let rho =
            (rho_vec[0] * rho_vec[0] + rho_vec[1] * rho_vec[1] + rho_vec[2] * rho_vec[2]).sqrt();
        if rho < 1e-12 {
            return [0.0, 0.0, 0.0];
        }
        let rho_hat = [rho_vec[0] / rho, rho_vec[1] / rho, rho_vec[2] / rho];
        // Distance from grid point to the core circle.
        let d2 = (rho - ring_radius) * (rho - ring_radius) + z * z;
        let mag = s * amp * (-d2 / sigma2).exp();
        // Vorticity direction: azimuthal about the axis, e_phi = e_a x rho_hat.
        let e_phi = [
            e_a[1] * rho_hat[2] - e_a[2] * rho_hat[1],
            e_a[2] * rho_hat[0] - e_a[0] * rho_hat[2],
            e_a[0] * rho_hat[1] - e_a[1] * rho_hat[0],
        ];
        [mag * e_phi[0], mag * e_phi[1], mag * e_phi[2]]
    };

    for i in 0..nx {
        let x = i as f64 * grid.dx();
        for j in 0..ny {
            let y = j as f64 * grid.dy();
            for k in 0..nz {
                let zc = k as f64 * grid.dz();
                let p = [x, y, zc];
                let w1 = ring_omega(p, c1, s1);
                let w2 = ring_omega(p, c2, s2);
                for c in 0..3 {
                    omega[c][[i, j, k]] = w1[c] + w2[c];
                }
            }
        }
    }

    // Forward FFT vorticity to spectral space.
    let (snx, sny, snz) = grid.spectral_shape();
    let shape = (snx, sny, snz);
    let mut omega_hat: [Array3<Complex<f64>>; 3] = [
        Array3::zeros(shape),
        Array3::zeros(shape),
        Array3::zeros(shape),
    ];
    for c in 0..3 {
        fft.r2c_3d(&omega[c], &mut omega_hat[c]);
    }

    // Invert the curl: u_hat = i k x omega_hat / |k|^2.
    // Multiplying a complex value z by i: (re, im) -> (-im, re).
    let ops = SpectralOps::<f64>::new(grid);
    let mut u_hat: [Array3<Complex<f64>>; 3] = [
        Array3::zeros(shape),
        Array3::zeros(shape),
        Array3::zeros(shape),
    ];
    for ix in 0..snx {
        let kx = ops.kx[ix];
        for iy in 0..sny {
            let ky = ops.ky[iy];
            for iz in 0..snz {
                let kz = ops.kz[iz];
                let k2 = ops.k_mag_sq[[ix, iy, iz]];
                if k2 < 1e-30 {
                    continue; // zero-mean flow at k = 0
                }
                let ox = omega_hat[0][[ix, iy, iz]];
                let oy = omega_hat[1][[ix, iy, iz]];
                let oz = omega_hat[2][[ix, iy, iz]];
                // cross = k x omega (complex components, real k)
                let cx = Complex::new(ky * oz.re - kz * oy.re, ky * oz.im - kz * oy.im);
                let cy = Complex::new(kz * ox.re - kx * oz.re, kz * ox.im - kx * oz.im);
                let cz = Complex::new(kx * oy.re - ky * ox.re, kx * oy.im - ky * ox.im);
                // multiply by i and divide by k^2: i*c = (-c.im, c.re)
                u_hat[0][[ix, iy, iz]] = Complex::new(-cx.im / k2, cx.re / k2);
                u_hat[1][[ix, iy, iz]] = Complex::new(-cy.im / k2, cy.re / k2);
                u_hat[2][[ix, iy, iz]] = Complex::new(-cz.im / k2, cz.re / k2);
            }
        }
    }

    // Guard divergence-free against round-off.
    ops.leray_project(&mut u_hat);

    // Inverse FFT to physical velocity.
    let mut v = VectorField::zeros(*grid);
    for c in 0..3 {
        fft.c2r_3d(&u_hat[c], &mut v.data[c]);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use vonkarman_core::field::GridSpec;
    use vonkarman_core::spectral_ops::SpectralOps;
    use vonkarman_fft::{BackendMode, create_backend};

    fn make(grid: &GridSpec) -> VectorField<f64> {
        let fft = create_backend(grid.nx, grid.ny, grid.nz, BackendMode::Cpu);
        colliding_vortex_rings(grid, 1.0, 0.35, 1.0, 2.0, 2, fft.as_ref())
    }

    #[test]
    fn rings_finite() {
        let grid = GridSpec::cubic(32, 2.0 * std::f64::consts::PI);
        let v = make(&grid);
        for c in 0..3 {
            for val in v.data[c].iter() {
                assert!(val.is_finite(), "non-finite in component {c}");
            }
        }
    }

    #[test]
    fn rings_nonzero_energy() {
        let grid = GridSpec::cubic(32, 2.0 * std::f64::consts::PI);
        let v = make(&grid);
        let mut e = 0.0;
        for c in 0..3 {
            for val in v.data[c].iter() {
                e += val * val;
            }
        }
        assert!(e > 0.0, "expected nonzero kinetic energy, got {e}");
    }

    #[test]
    fn rings_divergence_free() {
        // The defining property: spectral divergence k . u_hat ~ 0.
        let grid = GridSpec::cubic(32, 2.0 * std::f64::consts::PI);
        let fft = create_backend(grid.nx, grid.ny, grid.nz, BackendMode::Cpu);
        let v = colliding_vortex_rings(&grid, 1.0, 0.35, 1.0, 2.0, 2, fft.as_ref());
        let (snx, sny, snz) = grid.spectral_shape();
        let mut u_hat: [Array3<Complex<f64>>; 3] = [
            Array3::zeros((snx, sny, snz)),
            Array3::zeros((snx, sny, snz)),
            Array3::zeros((snx, sny, snz)),
        ];
        for c in 0..3 {
            fft.r2c_3d(&v.data[c], &mut u_hat[c]);
        }
        let ops = SpectralOps::<f64>::new(&grid);
        // Max |k . u_hat| relative to max |u_hat|.
        let mut max_div = 0.0_f64;
        let mut max_u = 1e-30_f64;
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    let kx = ops.kx[ix];
                    let ky = ops.ky[iy];
                    let kz = ops.kz[iz];
                    let ux = u_hat[0][[ix, iy, iz]];
                    let uy = u_hat[1][[ix, iy, iz]];
                    let uz = u_hat[2][[ix, iy, iz]];
                    let div_re = kx * ux.re + ky * uy.re + kz * uz.re;
                    let div_im = kx * ux.im + ky * uy.im + kz * uz.im;
                    max_div = max_div.max((div_re * div_re + div_im * div_im).sqrt());
                    max_u = max_u.max((ux.norm_sqr() + uy.norm_sqr() + uz.norm_sqr()).sqrt());
                }
            }
        }
        assert!(
            max_div / max_u < 1e-8,
            "not divergence-free: {}",
            max_div / max_u
        );
    }

    #[test]
    fn rings_two_cores() {
        // Vorticity must concentrate in TWO rings separated along the z-axis.
        // Test: reconstruct |omega| on the grid via spectral curl, then compute
        // the |omega|-weighted mean of |z - z_center|. This must be close to
        // separation/2 = 1.0 (within 35%), proving two cores offset along the
        // axis rather than a single central blob.
        let separation = 2.0_f64;
        let n = 48;
        let l = 2.0 * std::f64::consts::PI;
        let grid = GridSpec::cubic(n, l);
        let fft = create_backend(grid.nx, grid.ny, grid.nz, BackendMode::Cpu);
        let v = colliding_vortex_rings(&grid, 1.0, 0.35, 1.0, separation, 2, fft.as_ref());

        // Forward FFT velocity to spectral space.
        let (snx, sny, snz) = grid.spectral_shape();
        let mut u_hat: [Array3<Complex<f64>>; 3] = [
            Array3::zeros((snx, sny, snz)),
            Array3::zeros((snx, sny, snz)),
            Array3::zeros((snx, sny, snz)),
        ];
        for c in 0..3 {
            fft.r2c_3d(&v.data[c], &mut u_hat[c]);
        }

        // Compute curl in spectral space to get omega_hat.
        let ops = SpectralOps::<f64>::new(&grid);
        let mut omega_hat: [Array3<Complex<f64>>; 3] = [
            Array3::zeros((snx, sny, snz)),
            Array3::zeros((snx, sny, snz)),
            Array3::zeros((snx, sny, snz)),
        ];
        ops.curl(&u_hat, &mut omega_hat);

        // Inverse FFT omega_hat to get physical-space vorticity.
        let mut omega_phys: [ndarray::Array3<f64>; 3] = [
            ndarray::Array3::zeros((n, n, n)),
            ndarray::Array3::zeros((n, n, n)),
            ndarray::Array3::zeros((n, n, n)),
        ];
        for c in 0..3 {
            fft.c2r_3d(&omega_hat[c], &mut omega_phys[c]);
        }

        // Compute |omega|-weighted mean of |z - z_center| along the propagation axis (z).
        // axis=2 so z is the third index (k), with z_k = k * dz, z_center = l/2.
        let dz = grid.dz();
        let z_center = l / 2.0;
        let mut weight_sum = 0.0_f64;
        let mut weighted_dist_sum = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let wx = omega_phys[0][[i, j, k]];
                    let wy = omega_phys[1][[i, j, k]];
                    let wz = omega_phys[2][[i, j, k]];
                    let mag = (wx * wx + wy * wy + wz * wz).sqrt();
                    let z_k = k as f64 * dz;
                    let dist = (z_k - z_center).abs();
                    weight_sum += mag;
                    weighted_dist_sum += mag * dist;
                }
            }
        }

        assert!(weight_sum > 0.0, "total |omega| weight is zero");
        let mean_axial_dist = weighted_dist_sum / weight_sum;
        let expected = separation / 2.0;
        let tol = 0.35 * expected;
        assert!(
            (mean_axial_dist - expected).abs() < tol,
            "|omega|-weighted mean axial distance = {mean_axial_dist:.4}, expected {expected:.4} +/- {tol:.4}; \
             vorticity is not concentrated in two cores offset by separation/2 along the axis"
        );
    }
}
