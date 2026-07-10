use crate::etd::EtdCoeffs;
use crate::ic::{self, IcType};
use crate::nonlinear::compute_nonlinear;
use ndarray::Array3;
use num_complex::Complex;
use vonkarman_core::domain::{Domain, DomainType, PhysicsParams, Snapshot};
use vonkarman_core::field::{GridSpec, VectorField};
use vonkarman_core::spectral_ops::SpectralOps;
use vonkarman_diag::FrameDiagnostics;
use vonkarman_fft::{BackendMode, FftBackend, create_backend};

/// 3D pseudospectral Navier-Stokes solver on the periodic torus T^3.
pub struct Periodic3D {
    /// Spectral velocity coefficients (primary state).
    u_hat: [Array3<Complex<f64>>; 3],
    /// Precomputed spectral operators.
    ops: SpectralOps<f64>,
    /// ETD coefficients per wavenumber (flattened, indexed by linear index into k_mag_sq).
    etd_coeffs: Vec<EtdCoeffs>,
    /// Physics parameters.
    params: PhysicsParams,
    /// Grid metadata.
    grid: GridSpec,
    /// Current time.
    time: f64,
    /// Current timestep.
    dt: f64,
    /// Step counter.
    step_count: u64,
    /// CFL safety factor.
    cfl_safety: f64,
    /// FFT backend (original grid).
    fft: Box<dyn FftBackend<f64>>,
    /// FFT backend (3/2-padded grid).
    fft_padded: Box<dyn FftBackend<f64>>,
}

impl Periodic3D {
    pub fn new(grid: GridSpec, nu: f64, ic: IcType, backend_mode: BackendMode) -> Self {
        let ops = SpectralOps::<f64>::new(&grid);
        let (snx, sny, snz) = grid.spectral_shape();
        let shape = (snx, sny, snz);

        // FFT backends (needed before IC generation for random_isotropic)
        let fft = create_backend(grid.nx, grid.ny, grid.nz, backend_mode);

        // Generate physical-space IC and transform to spectral
        let velocity = match ic {
            IcType::TaylorGreen => ic::taylor_green::<f64>(&grid),
            IcType::Abc { a, b, c } => ic::abc_flow::<f64>(&grid, a, b, c),
            IcType::AntiParallelTubes {
                circulation,
                core_radius,
                separation,
                perturbation,
            } => ic::anti_parallel_tubes::<f64>(
                &grid,
                circulation,
                core_radius,
                separation,
                perturbation,
            ),
            IcType::KidaPelz => ic::kida_pelz::<f64>(&grid),
            IcType::RandomIsotropic {
                k_peak,
                energy,
                seed,
            } => ic::random_isotropic(&grid, k_peak, energy, seed, fft.as_ref()),
            IcType::CollidingVortexRings {
                ring_radius,
                core_radius,
                circulation,
                separation,
                axis,
            } => ic::colliding_vortex_rings(
                &grid,
                ring_radius,
                core_radius,
                circulation,
                separation,
                axis,
                fft.as_ref(),
            ),
        };
        let mut u_hat: [Array3<Complex<f64>>; 3] = [
            Array3::zeros(shape),
            Array3::zeros(shape),
            Array3::zeros(shape),
        ];
        for c in 0..3 {
            fft.r2c_3d(&velocity.data[c], &mut u_hat[c]);
        }

        let pg = grid.padded_3half();
        let fft_padded = create_backend(pg.nx, pg.ny, pg.nz, backend_mode);

        let re = if nu > 0.0 { 1.0 / nu } else { f64::INFINITY };
        let params = PhysicsParams {
            nu,
            re,
            domain: DomainType::Periodic3D,
        };

        // Initial dt from CFL
        let cfl_safety = 0.5;
        let dt = Self::compute_cfl_dt_static(&u_hat, fft.as_ref(), &grid, cfl_safety, nu);

        // Precompute ETD coefficients
        let etd_coeffs = Self::compute_etd_coeffs(&ops, nu, dt);

        Self {
            u_hat,
            ops,
            etd_coeffs,
            params,
            grid,
            time: 0.0,
            dt,
            step_count: 0,
            cfl_safety,
            fft,
            fft_padded,
        }
    }

    /// Extract checkpoint data for serialisation.
    pub fn checkpoint_data(&self) -> vonkarman_io::CheckpointData {
        vonkarman_io::CheckpointData {
            u_hat: self.u_hat.clone(),
            time: self.time,
            step_count: self.step_count,
            dt: self.dt,
            grid: self.grid,
            nu: self.params.nu,
            config_toml: String::new(), // caller fills this in
        }
    }

    /// Reconstruct a solver from checkpoint data.
    ///
    /// Recomputes ETD coefficients, SpectralOps, and FFT backends
    /// from the stored grid, nu, and dt.
    pub fn from_checkpoint(data: vonkarman_io::CheckpointData, backend_mode: BackendMode) -> Self {
        let grid = data.grid;
        let nu = data.nu;
        let ops = SpectralOps::<f64>::new(&grid);

        let fft = create_backend(grid.nx, grid.ny, grid.nz, backend_mode);
        let pg = grid.padded_3half();
        let fft_padded = create_backend(pg.nx, pg.ny, pg.nz, backend_mode);

        let re = if nu > 0.0 { 1.0 / nu } else { f64::INFINITY };
        let params = PhysicsParams {
            nu,
            re,
            domain: DomainType::Periodic3D,
        };

        let cfl_safety = 0.5;
        let etd_coeffs = Self::compute_etd_coeffs(&ops, nu, data.dt);

        Self {
            u_hat: data.u_hat,
            ops,
            etd_coeffs,
            params,
            grid,
            time: data.time,
            dt: data.dt,
            step_count: data.step_count,
            cfl_safety,
            fft,
            fft_padded,
        }
    }

    fn compute_etd_coeffs(ops: &SpectralOps<f64>, nu: f64, dt: f64) -> Vec<EtdCoeffs> {
        let (snx, sny, snz) = ops.grid.spectral_shape();
        let mut coeffs = Vec::with_capacity(snx * sny * snz);
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    let lambda = -nu * ops.k_mag_sq[[ix, iy, iz]];
                    coeffs.push(EtdCoeffs::new(lambda * dt));
                }
            }
        }
        coeffs
    }

    fn recompute_etd(&mut self) {
        self.etd_coeffs = Self::compute_etd_coeffs(&self.ops, self.params.nu, self.dt);
    }

    /// Compute CFL-based timestep from current velocity field.
    fn compute_cfl_dt_static(
        u_hat: &[Array3<Complex<f64>>; 3],
        fft: &dyn FftBackend<f64>,
        grid: &GridSpec,
        safety: f64,
        nu: f64,
    ) -> f64 {
        // Transform to physical space to get ||u||_inf
        let mut u_phys = [
            Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz)),
            Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz)),
            Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz)),
        ];
        for c in 0..3 {
            fft.c2r_3d(&u_hat[c], &mut u_phys[c]);
        }

        let mut u_max = 0.0_f64;
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let speed = (u_phys[0][[i, j, k]].powi(2)
                        + u_phys[1][[i, j, k]].powi(2)
                        + u_phys[2][[i, j, k]].powi(2))
                    .sqrt();
                    u_max = u_max.max(speed);
                }
            }
        }

        let dx = grid.dx();
        let advective = if u_max > 1e-30 { dx / u_max } else { 1.0 };
        let viscous = if nu > 1e-30 {
            dx * dx / nu
        } else {
            f64::INFINITY
        };
        safety * advective.min(viscous).min(0.1) // cap at 0.1
    }

    /// ETD-RK4 step (Cox-Matthews / Kassam-Trefethen).
    fn etd_rk4_step(&mut self) {
        let (snx, sny, snz) = self.grid.spectral_shape();
        let shape = (snx, sny, snz);
        let zero = Complex { re: 0.0, im: 0.0 };
        let dt = self.dt;

        // Allocate RK stage nonlinear terms
        let mut n1 = [
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
        ];
        let mut n2 = [
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
        ];
        let mut n3 = [
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
        ];
        let mut n4 = [
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
        ];
        let mut temp = [
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
        ];

        // Stage 1: N1 = nonlinear(u_hat)
        compute_nonlinear(
            &self.ops,
            self.fft.as_ref(),
            self.fft_padded.as_ref(),
            &self.grid,
            &self.u_hat,
            &mut n1,
        );

        // Stage 2: temp = exp_half * u_hat + dt * a21 * N1
        for c in 0..3 {
            let mut idx = 0;
            for ix in 0..snx {
                for iy in 0..sny {
                    for iz in 0..snz {
                        let ec = &self.etd_coeffs[idx];
                        let u = self.u_hat[c][[ix, iy, iz]];
                        let n = n1[c][[ix, iy, iz]];
                        temp[c][[ix, iy, iz]] = Complex {
                            re: ec.exp_half * u.re + dt * ec.a21 * n.re,
                            im: ec.exp_half * u.im + dt * ec.a21 * n.im,
                        };
                        idx += 1;
                    }
                }
            }
        }
        compute_nonlinear(
            &self.ops,
            self.fft.as_ref(),
            self.fft_padded.as_ref(),
            &self.grid,
            &temp,
            &mut n2,
        );

        // Stage 3: temp = exp_half * u_hat + dt * a31 * N2
        for c in 0..3 {
            let mut idx = 0;
            for ix in 0..snx {
                for iy in 0..sny {
                    for iz in 0..snz {
                        let ec = &self.etd_coeffs[idx];
                        let u = self.u_hat[c][[ix, iy, iz]];
                        let n = n2[c][[ix, iy, iz]];
                        temp[c][[ix, iy, iz]] = Complex {
                            re: ec.exp_half * u.re + dt * ec.a31 * n.re,
                            im: ec.exp_half * u.im + dt * ec.a31 * n.im,
                        };
                        idx += 1;
                    }
                }
            }
        }
        compute_nonlinear(
            &self.ops,
            self.fft.as_ref(),
            self.fft_padded.as_ref(),
            &self.grid,
            &temp,
            &mut n3,
        );

        // Stage 4: temp = exp_full * u_hat + dt * a41 * (2*N3 - N1)
        // Note: stage 4 uses exp_half * (exp_half * u_hat) for the linear part,
        // but for ETD-RK4, the intermediate state is:
        //   temp = exp_half * a + dt * phi1_half * (2*N3 - N1)
        // where a = exp_half * u_hat from stage 2.
        // Equivalently: temp = exp_full * u_hat + dt * a41 * (2*N3 - N1)
        for c in 0..3 {
            let mut idx = 0;
            for ix in 0..snx {
                for iy in 0..sny {
                    for iz in 0..snz {
                        let ec = &self.etd_coeffs[idx];
                        let u = self.u_hat[c][[ix, iy, iz]];
                        let dn = Complex {
                            re: 2.0 * n3[c][[ix, iy, iz]].re - n1[c][[ix, iy, iz]].re,
                            im: 2.0 * n3[c][[ix, iy, iz]].im - n1[c][[ix, iy, iz]].im,
                        };
                        // Use exp_half on the stage-2 intermediate: exp_half * (exp_half * u) = exp_full * u
                        temp[c][[ix, iy, iz]] = Complex {
                            re: ec.exp_full * u.re + dt * ec.a41 * dn.re,
                            im: ec.exp_full * u.im + dt * ec.a41 * dn.im,
                        };
                        idx += 1;
                    }
                }
            }
        }
        compute_nonlinear(
            &self.ops,
            self.fft.as_ref(),
            self.fft_padded.as_ref(),
            &self.grid,
            &temp,
            &mut n4,
        );

        // Final update: u_hat = exp_full * u_hat + dt * (b1*N1 + b23*(N2+N3) + b4*N4)
        for c in 0..3 {
            let mut idx = 0;
            for ix in 0..snx {
                for iy in 0..sny {
                    for iz in 0..snz {
                        let ec = &self.etd_coeffs[idx];
                        let u = self.u_hat[c][[ix, iy, iz]];
                        let rhs = Complex {
                            re: ec.b1 * n1[c][[ix, iy, iz]].re
                                + ec.b23 * (n2[c][[ix, iy, iz]].re + n3[c][[ix, iy, iz]].re)
                                + ec.b4 * n4[c][[ix, iy, iz]].re,
                            im: ec.b1 * n1[c][[ix, iy, iz]].im
                                + ec.b23 * (n2[c][[ix, iy, iz]].im + n3[c][[ix, iy, iz]].im)
                                + ec.b4 * n4[c][[ix, iy, iz]].im,
                        };
                        self.u_hat[c][[ix, iy, iz]] = Complex {
                            re: ec.exp_full * u.re + dt * rhs.re,
                            im: ec.exp_full * u.im + dt * rhs.im,
                        };
                        idx += 1;
                    }
                }
            }
        }
    }
}

impl Domain<f64> for Periodic3D {
    fn step(&mut self) {
        // Adaptive dt
        let new_dt = Self::compute_cfl_dt_static(
            &self.u_hat,
            self.fft.as_ref(),
            &self.grid,
            self.cfl_safety,
            self.params.nu,
        );
        if (new_dt - self.dt).abs() / self.dt.max(1e-30) > 0.01 {
            self.dt = new_dt;
            self.recompute_etd();
        }

        self.etd_rk4_step();
        self.time += self.dt;
        self.step_count += 1;
    }

    fn time(&self) -> f64 {
        self.time
    }
    fn step_count(&self) -> u64 {
        self.step_count
    }
    fn dt(&self) -> f64 {
        self.dt
    }

    fn energy(&self) -> f64 {
        // E = (1/2) * (1/N^3) * sum |u_hat|^2 (with R2C weighting)
        let (snx, sny, snz) = self.grid.spectral_shape();
        let ntot = (self.grid.nx * self.grid.ny * self.grid.nz) as f64;
        let mut e = 0.0_f64;
        for c in 0..3 {
            for ix in 0..snx {
                for iy in 0..sny {
                    for iz in 0..snz {
                        let mag2 = self.u_hat[c][[ix, iy, iz]].re.powi(2)
                            + self.u_hat[c][[ix, iy, iz]].im.powi(2);
                        let weight = if iz == 0 || iz == self.grid.nz / 2 {
                            1.0
                        } else {
                            2.0
                        };
                        e += weight * mag2;
                    }
                }
            }
        }
        0.5 * e / (ntot * ntot)
    }

    fn enstrophy(&self) -> f64 {
        let (snx, sny, snz) = self.grid.spectral_shape();
        let ntot = (self.grid.nx * self.grid.ny * self.grid.nz) as f64;
        let shape = (snx, sny, snz);
        let zero = Complex { re: 0.0, im: 0.0 };
        let mut omega_hat = [
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
        ];
        self.ops.curl(&self.u_hat, &mut omega_hat);
        let mut ens = 0.0_f64;
        for c in 0..3 {
            for ix in 0..snx {
                for iy in 0..sny {
                    for iz in 0..snz {
                        let mag2 = omega_hat[c][[ix, iy, iz]].re.powi(2)
                            + omega_hat[c][[ix, iy, iz]].im.powi(2);
                        let weight = if iz == 0 || iz == self.grid.nz / 2 {
                            1.0
                        } else {
                            2.0
                        };
                        ens += weight * mag2;
                    }
                }
            }
        }
        ens / (ntot * ntot)
    }

    fn helicity(&self) -> f64 {
        let (snx, sny, snz) = self.grid.spectral_shape();
        let ntot = (self.grid.nx * self.grid.ny * self.grid.nz) as f64;
        let shape = (snx, sny, snz);
        let zero = Complex { re: 0.0, im: 0.0 };
        let mut omega_hat = [
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
        ];
        self.ops.curl(&self.u_hat, &mut omega_hat);
        let mut h = 0.0_f64;
        for c in 0..3 {
            for ix in 0..snx {
                for iy in 0..sny {
                    for iz in 0..snz {
                        let u = self.u_hat[c][[ix, iy, iz]];
                        let o = omega_hat[c][[ix, iy, iz]];
                        let dot = u.re * o.re + u.im * o.im;
                        let weight = if iz == 0 || iz == self.grid.nz / 2 {
                            1.0
                        } else {
                            2.0
                        };
                        h += weight * dot;
                    }
                }
            }
        }
        h / (ntot * ntot)
    }

    fn superhelicity(&self) -> f64 {
        let (snx, sny, snz) = self.grid.spectral_shape();
        let ntot = (self.grid.nx * self.grid.ny * self.grid.nz) as f64;
        let shape = (snx, sny, snz);
        let zero = Complex { re: 0.0, im: 0.0 };
        let mut omega_hat = [
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
        ];
        self.ops.curl(&self.u_hat, &mut omega_hat);
        let mut curl_omega_hat = [
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
        ];
        self.ops.curl(&omega_hat, &mut curl_omega_hat);
        let mut h2 = 0.0_f64;
        for c in 0..3 {
            for ix in 0..snx {
                for iy in 0..sny {
                    for iz in 0..snz {
                        let o = omega_hat[c][[ix, iy, iz]];
                        let co = curl_omega_hat[c][[ix, iy, iz]];
                        let dot = o.re * co.re + o.im * co.im;
                        let weight = if iz == 0 || iz == self.grid.nz / 2 {
                            1.0
                        } else {
                            2.0
                        };
                        h2 += weight * dot;
                    }
                }
            }
        }
        h2 / (ntot * ntot)
    }

    fn max_vorticity(&self) -> f64 {
        let (snx, sny, snz) = self.grid.spectral_shape();
        let shape = (snx, sny, snz);
        let zero = Complex { re: 0.0, im: 0.0 };
        let mut omega_hat = [
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
        ];
        self.ops.curl(&self.u_hat, &mut omega_hat);
        let mut omega_phys = [
            Array3::<f64>::zeros((self.grid.nx, self.grid.ny, self.grid.nz)),
            Array3::<f64>::zeros((self.grid.nx, self.grid.ny, self.grid.nz)),
            Array3::<f64>::zeros((self.grid.nx, self.grid.ny, self.grid.nz)),
        ];
        for c in 0..3 {
            self.fft.c2r_3d(&omega_hat[c], &mut omega_phys[c]);
        }
        let mut max_w = 0.0_f64;
        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let w2 = omega_phys[0][[i, j, k]].powi(2)
                        + omega_phys[1][[i, j, k]].powi(2)
                        + omega_phys[2][[i, j, k]].powi(2);
                    max_w = max_w.max(w2.sqrt());
                }
            }
        }
        max_w
    }

    fn cfl_dt(&self) -> f64 {
        Self::compute_cfl_dt_static(
            &self.u_hat,
            self.fft.as_ref(),
            &self.grid,
            self.cfl_safety,
            self.params.nu,
        )
    }

    fn u_hat(&self) -> &[Array3<Complex<f64>>; 3] {
        &self.u_hat
    }
    fn grid(&self) -> &GridSpec {
        &self.grid
    }
    fn params(&self) -> &PhysicsParams {
        &self.params
    }

    fn snapshot(&self) -> Snapshot<f64> {
        let mut velocity = VectorField::zeros(self.grid);
        for c in 0..3 {
            self.fft.c2r_3d(&self.u_hat[c], &mut velocity.data[c]);
        }
        let (snx, sny, snz) = self.grid.spectral_shape();
        let shape = (snx, sny, snz);
        let zero = Complex { re: 0.0, im: 0.0 };
        let mut omega_hat = [
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
            Array3::from_elem(shape, zero),
        ];
        self.ops.curl(&self.u_hat, &mut omega_hat);
        let mut vorticity = VectorField::zeros(self.grid);
        for c in 0..3 {
            self.fft.c2r_3d(&omega_hat[c], &mut vorticity.data[c]);
        }
        Snapshot {
            time: self.time,
            step: self.step_count,
            dt: self.dt,
            velocity,
            vorticity,
            u_hat: self.u_hat.clone(),
            grid: self.grid,
            params: self.params,
        }
    }
}

impl Periodic3D {
    /// Frame / coherence / pressure diagnostics for the Clifford-NS regularity
    /// programme: the frame-projected pressure occupancy rho = ||alpha_p|| / ||f||
    /// (alpha_p = -xi_i xi_j R_i R_j f, f = |S|^2 - 1/2 |omega|^2), the vector vs
    /// nematic coherence energies (defect-honesty of Xi = xi (x) xi), and the
    /// omega_Xi_crit density <|omega|^{1/2} |grad Xi|^2>. Heavy (order 50 FFTs), so it
    /// is meant to run at the diagnostics cadence, not every step. Physics in
    /// `vonkarman_diag::frame`.
    pub fn frame_diagnostics(&self) -> FrameDiagnostics {
        let (snx, sny, snz) = self.grid.spectral_shape();
        let pshape = (self.grid.nx, self.grid.ny, self.grid.nz);
        let sshape = (snx, sny, snz);
        let n3 = (self.grid.nx * self.grid.ny * self.grid.nz) as f64;
        let zero = Complex { re: 0.0, im: 0.0 };

        // d/dx_j of a spectral field, returned in physical space (i k_j, then inverse).
        let deriv = |a_hat: &Array3<Complex<f64>>, j: usize| -> Array3<f64> {
            let mut d_hat = Array3::from_elem(sshape, zero);
            for ix in 0..snx {
                for iy in 0..sny {
                    for iz in 0..snz {
                        let k = match j {
                            0 => self.ops.kx[ix],
                            1 => self.ops.ky[iy],
                            _ => self.ops.kz[iz],
                        };
                        let a = a_hat[[ix, iy, iz]];
                        d_hat[[ix, iy, iz]] = Complex {
                            re: -k * a.im,
                            im: k * a.re,
                        };
                    }
                }
            }
            let mut out = Array3::zeros(pshape);
            self.fft.c2r_3d(&d_hat, &mut out);
            out
        };
        let fwd = |field: &Array3<f64>| -> Array3<Complex<f64>> {
            let mut h = Array3::from_elem(sshape, zero);
            self.fft.r2c_3d(field, &mut h);
            h
        };

        // vorticity (physical), magnitude and direction xi = omega / |omega|
        let mut omega_hat: [Array3<Complex<f64>>; 3] =
            std::array::from_fn(|_| Array3::from_elem(sshape, zero));
        self.ops.curl(&self.u_hat, &mut omega_hat);
        let omega: [Array3<f64>; 3] = std::array::from_fn(|c| {
            let mut o = Array3::zeros(pshape);
            self.fft.c2r_3d(&omega_hat[c], &mut o);
            o
        });
        let mut w2 = Array3::<f64>::zeros(pshape);
        for c in 0..3 {
            w2 += &omega[c].mapv(|x| x * x);
        }
        let wmag = w2.mapv(f64::sqrt);
        let wmax = wmag.iter().cloned().fold(0.0_f64, f64::max);
        let eps = 1e-6 * wmax.max(1e-30);
        let xi: [Array3<f64>; 3] = std::array::from_fn(|c| {
            let denom = wmag.mapv(|w| w + eps);
            &omega[c] / &denom
        });

        // strain magnitude and the CLMS null form f = |S|^2 - 1/2 |omega|^2
        let gradu: [[Array3<f64>; 3]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| deriv(&self.u_hat[i], j)));
        let mut s2 = Array3::<f64>::zeros(pshape);
        for i in 0..3 {
            for j in 0..3 {
                let s = (&gradu[i][j] + &gradu[j][i]).mapv(|x| 0.5 * x);
                s2 += &s.mapv(|x| x * x);
            }
        }
        let f = &s2 - &w2.mapv(|x| 0.5 * x);

        // frame-projected pressure alpha_p = - xi_i xi_j R_i R_j f
        let f_hat = fwd(&f);
        let mut alpha_p = Array3::<f64>::zeros(pshape);
        for i in 0..3 {
            for j in 0..3 {
                let mut rrf_hat = Array3::from_elem(sshape, zero);
                for ix in 0..snx {
                    for iy in 0..sny {
                        for iz in 0..snz {
                            let k2 = self.ops.k_mag_sq[[ix, iy, iz]];
                            if k2 < 1e-30 {
                                continue;
                            }
                            let ki = match i {
                                0 => self.ops.kx[ix],
                                1 => self.ops.ky[iy],
                                _ => self.ops.kz[iz],
                            };
                            let kj = match j {
                                0 => self.ops.kx[ix],
                                1 => self.ops.ky[iy],
                                _ => self.ops.kz[iz],
                            };
                            let m = ki * kj / k2;
                            let fh = f_hat[[ix, iy, iz]];
                            rrf_hat[[ix, iy, iz]] = Complex {
                                re: m * fh.re,
                                im: m * fh.im,
                            };
                        }
                    }
                }
                let mut rrf = Array3::zeros(pshape);
                self.fft.c2r_3d(&rrf_hat, &mut rrf);
                let term = (&xi[i] * &xi[j]) * &rrf; // xi_i xi_j (R_i R_j f)
                alpha_p -= &term;
            }
        }

        // coherence energies: vector <|grad xi|^2>, nematic <|grad Xi|^2>
        let mut gxi2 = Array3::<f64>::zeros(pshape);
        for c in 0..3 {
            let xh = fwd(&xi[c]);
            for j in 0..3 {
                let d = deriv(&xh, j);
                gxi2 += &d.mapv(|x| x * x);
            }
        }
        let mut gnem2 = Array3::<f64>::zeros(pshape);
        for a in 0..3 {
            for b in a..3 {
                let xab = &xi[a] * &xi[b];
                let xh = fwd(&xab);
                let wt = if a == b { 1.0 } else { 2.0 }; // off-diagonal counted twice
                for j in 0..3 {
                    let d = deriv(&xh, j);
                    gnem2 += &d.mapv(|x| wt * x * x);
                }
            }
        }

        // reductions (means over the physical grid; _hi over the high-|omega| region)
        let mean = |arr: &Array3<f64>| arr.sum() / n3;
        let rms = |arr: &Array3<f64>| (arr.mapv(|x| x * x).sum() / n3).sqrt();
        let thresh = 0.3 * wmax;
        let mask: Array3<f64> = wmag.mapv(|w| if w > thresh { 1.0_f64 } else { 0.0 });
        let mcount = mask.sum().max(1.0);
        let masked_rms = |arr: &Array3<f64>| ((&arr.mapv(|x| x * x) * &mask).sum() / mcount).sqrt();
        let masked_mean = |arr: &Array3<f64>| (arr * &mask).sum() / mcount;

        let f_rms = rms(&f);
        let alpha_p_rms = rms(&alpha_p);
        FrameDiagnostics {
            time: self.time,
            step: self.step_count,
            enstrophy: mean(&w2),
            max_vorticity: wmax,
            f_rms,
            alpha_p_rms,
            rho_all: alpha_p_rms / (f_rms + 1e-30),
            rho_hi: masked_rms(&alpha_p) / (masked_rms(&f) + 1e-30),
            xi_energy: mean(&gxi2),
            nem_energy: mean(&gnem2),
            xi_energy_hi: masked_mean(&gxi2),
            nem_energy_hi: masked_mean(&gnem2),
            coherence_w: mean(&(&wmag.mapv(f64::sqrt) * &gnem2)),
            hi_fraction: mask.sum() / n3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vonkarman_core::domain::Domain;

    #[test]
    fn taylor_green_energy_decays() {
        let n = 16;
        let nu = 0.01;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let mut solver = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);

        let e0 = solver.energy();
        assert!(e0 > 0.0, "initial energy should be positive");

        let mut prev_e = e0;
        for _ in 0..50 {
            solver.step();
            let e = solver.energy();
            assert!(
                e <= prev_e + 1e-14 * prev_e.abs().max(1e-30),
                "energy increased: {prev_e} -> {e} at t={}",
                solver.time()
            );
            prev_e = e;
        }
        assert!(
            prev_e < 0.99 * e0,
            "energy didn't decay enough: {e0} -> {prev_e}"
        );
    }

    #[test]
    fn taylor_green_short_time_exponential_decay() {
        let n = 32;
        let nu = 0.01;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let mut solver = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);

        let e0 = solver.energy();
        while solver.time() < 0.5 {
            solver.step();
        }
        let e = solver.energy();
        let t = solver.time();
        let expected = e0 * (-2.0 * nu * t).exp();
        let rel_err = (e - expected).abs() / expected;
        assert!(
            rel_err < 0.1,
            "energy at t={t}: got {e}, expected ~{expected} (rel_err={rel_err})"
        );
    }

    #[test]
    fn from_checkpoint_matches_continuous() {
        let n = 16;
        let nu = 0.01;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);

        // Run 100 steps continuously
        let mut continuous = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
        for _ in 0..100 {
            continuous.step();
        }

        // Run 50 steps, checkpoint, restart, run 50 more
        let mut first_half = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
        for _ in 0..50 {
            first_half.step();
        }
        let checkpoint = first_half.checkpoint_data();
        let mut restarted = Periodic3D::from_checkpoint(checkpoint, BackendMode::Cpu);
        for _ in 0..50 {
            restarted.step();
        }

        // Bitwise comparison
        for c in 0..3 {
            assert_eq!(
                continuous.u_hat()[c],
                restarted.u_hat()[c],
                "u_hat[{c}] diverged after restart"
            );
        }
        assert_eq!(continuous.time(), restarted.time());
        assert_eq!(continuous.step_count(), restarted.step_count());
    }

    #[test]
    fn colliding_rings_solver_constructs() {
        let grid = GridSpec::cubic(16, 2.0 * std::f64::consts::PI);
        let ic = IcType::CollidingVortexRings {
            ring_radius: 1.0,
            core_radius: 0.35,
            circulation: 1.0,
            separation: 2.0,
            axis: 2,
        };
        let _solver = Periodic3D::new(grid, 1e-3, ic, BackendMode::Cpu);
    }
}
