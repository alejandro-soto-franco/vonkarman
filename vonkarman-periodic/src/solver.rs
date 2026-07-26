use crate::etd::EtdCoeffs;
use crate::ic::{self, IcType};
use crate::nonlinear::compute_nonlinear;
use ndarray::{Array3, Zip};
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
    /// Frame / coherence / pressure diagnostics (see `frame_diagnostics_uhat`).
    pub fn frame_diagnostics(&self) -> FrameDiagnostics {
        frame_diagnostics_uhat(
            &self.u_hat,
            &self.ops,
            &self.grid,
            self.params.nu,
            self.time,
            self.step_count,
        )
    }
}

/// Frame / coherence / pressure diagnostics from a raw spectral velocity state
/// `u_hat` (Clifford-NS programme): the frame-projected pressure occupancy
/// rho = ||alpha_p|| / ||f|| (alpha_p = -xi_i xi_j R_i R_j f, f = |S|^2 - 1/2 |omega|^2),
/// the vector vs nematic coherence energies (Xi = xi (x) xi), and the omega_Xi_crit
/// density <|omega|^{1/2} |grad Xi|^2>. Shared by the CPU `Periodic3D` path and the
/// GPU-resident path. Heavy (order 50 FFTs); run at the diagnostics cadence. Physics
/// in `vonkarman_diag::frame`.
pub fn frame_diagnostics_uhat(
    u_hat: &[Array3<Complex<f64>>; 3],
    ops: &SpectralOps<f64>,
    grid: &GridSpec,
    nu: f64,
    time: f64,
    step: u64,
) -> FrameDiagnostics {
    use rayon::prelude::*;
    use vonkarman_fft::NdrustfftBackend;

    let (snx, sny, snz) = grid.spectral_shape();
    let pshape = (grid.nx, grid.ny, grid.nz);
    let sshape = (snx, sny, snz);
    let (nx, ny, nz) = pshape;
    let n3 = (grid.nx * grid.ny * grid.nz) as f64;
    let zero = Complex { re: 0.0, im: 0.0 };

    // Parallel inverse FFTs: this diagnostic needs ~22 independent transforms; run
    // them across cores (each on its own cheap backend, which has no shared mutable
    // state) instead of serially -- the serial CPU FFTs were the diagnostic
    // bottleneck at n=256, while the GPU-resident solver does the stepping.
    let par_c2r = |inputs: Vec<Array3<Complex<f64>>>| -> Vec<Array3<f64>> {
        inputs
            .into_par_iter()
            .map(|inp| {
                let local = NdrustfftBackend::new(nx, ny, nz);
                let mut out = Array3::<f64>::zeros(pshape);
                local.c2r_3d(&inp, &mut out);
                out
            })
            .collect()
    };
    // i k_j * a_hat (spectral derivative; inverse-transformed later in a batch).
    let deriv_hat = |a_hat: &Array3<Complex<f64>>, j: usize| -> Array3<Complex<f64>> {
        let mut d = Array3::from_elem(sshape, zero);
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    let k = match j {
                        0 => ops.kx[ix],
                        1 => ops.ky[iy],
                        _ => ops.kz[iz],
                    };
                    let a = a_hat[[ix, iy, iz]];
                    d[[ix, iy, iz]] = Complex {
                        re: -k * a.im,
                        im: k * a.re,
                    };
                }
            }
        }
        d
    };

    // Batch 1: vorticity (3) + velocity gradients (9), inverse-transformed together.
    let mut omega_hat: [Array3<Complex<f64>>; 3] =
        std::array::from_fn(|_| Array3::from_elem(sshape, zero));
    ops.curl(u_hat, &mut omega_hat);
    let mut batch1: Vec<Array3<Complex<f64>>> = Vec::with_capacity(12);
    for c in 0..3 {
        batch1.push(omega_hat[c].clone());
    }
    for i in 0..3 {
        for j in 0..3 {
            batch1.push(deriv_hat(&u_hat[i], j));
        }
    }
    // Vorticity gradients d_j omega_i, needed for the band-limited director-gradient
    // energy below. omega is band-limited, so these spectral derivatives are exact.
    for i in 0..3 {
        for j in 0..3 {
            batch1.push(deriv_hat(&omega_hat[i], j));
        }
    }
    // Vorticity Laplacians, for the second-order structure scale l3 = Sqrt(rho/|Lap omega|).
    // Symbol -|k|^2, so three transforms and no derivative composition.
    for i in 0..3 {
        let mut lh = Array3::from_elem(sshape, zero);
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    let k2 = ops.k_mag_sq[[ix, iy, iz]];
                    let a = omega_hat[i][[ix, iy, iz]];
                    lh[[ix, iy, iz]] = Complex { re: -k2 * a.re, im: -k2 * a.im };
                }
            }
        }
        batch1.push(lh);
    }
    let phys1 = par_c2r(batch1);
    let omega: [&Array3<f64>; 3] = [&phys1[0], &phys1[1], &phys1[2]];
    let gradu = |i: usize, j: usize| -> &Array3<f64> { &phys1[3 + i * 3 + j] };
    let gradw = |i: usize, j: usize| -> &Array3<f64> { &phys1[12 + i * 3 + j] };
    let lapw = |i: usize| -> &Array3<f64> { &phys1[21 + i] };

    let mut w2 = Array3::<f64>::zeros(pshape);
    for c in 0..3 {
        w2 += &omega[c].mapv(|x| x * x);
    }
    let wmag = w2.mapv(f64::sqrt);
    let wmax = wmag.iter().cloned().fold(0.0_f64, f64::max);
    let eps = 1e-6 * wmax.max(1e-30);
    let xi: [Array3<f64>; 3] = std::array::from_fn(|c| {
        let denom = wmag.mapv(|w| w + eps);
        omega[c] / &denom
    });

    // strain magnitude, the CLMS null form f = |S|^2 - 1/2 |omega|^2, and the enstrophy
    // production density omega . S omega = rho^2 (xi . S xi), the left side of (PAYOFF).
    let mut s2 = Array3::<f64>::zeros(pshape);
    let mut prod = Array3::<f64>::zeros(pshape);
    for i in 0..3 {
        for j in 0..3 {
            let s = (gradu(i, j) + gradu(j, i)).mapv(|x| 0.5 * x);
            s2 += &s.mapv(|x| x * x);
            prod += &(&s * omega[i] * omega[j]);
        }
    }
    let f = &s2 - &w2.mapv(|x| 0.5 * x);

    // frame-projected pressure alpha_p = - xi_i xi_j R_i R_j f
    let f_hat = {
        let local = NdrustfftBackend::new(nx, ny, nz);
        let mut h = Array3::from_elem(sshape, zero);
        local.r2c_3d(&f, &mut h);
        h
    };
    // Batch 2: the 9 Riesz-projected pressure components R_i R_j f = ifft(k_i k_j/|k|^2 f_hat).
    let mut batch2: Vec<Array3<Complex<f64>>> = Vec::with_capacity(9);
    for i in 0..3 {
        for j in 0..3 {
            let mut rrf_hat = Array3::from_elem(sshape, zero);
            for ix in 0..snx {
                for iy in 0..sny {
                    for iz in 0..snz {
                        let k2 = ops.k_mag_sq[[ix, iy, iz]];
                        if k2 < 1e-30 {
                            continue;
                        }
                        let ki = match i {
                            0 => ops.kx[ix],
                            1 => ops.ky[iy],
                            _ => ops.kz[iz],
                        };
                        let kj = match j {
                            0 => ops.kx[ix],
                            1 => ops.ky[iy],
                            _ => ops.kz[iz],
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
            batch2.push(rrf_hat);
        }
    }
    let rrf = par_c2r(batch2);
    let mut alpha_p = Array3::<f64>::zeros(pshape);
    for i in 0..3 {
        for j in 0..3 {
            let term = (&xi[i] * &xi[j]) * &rrf[i * 3 + j]; // xi_i xi_j (R_i R_j f)
            alpha_p -= &term;
        }
    }

    // Coherence energies via PERIODIC FINITE DIFFERENCES (physical space): the
    // direction xi = omega/|omega| is NOT band-limited, so a spectral derivative of
    // xi (or Xi = xi (x) xi) aliases -- the aliasing grows with the small-scale
    // content and makes <|grad Xi|^2> unmeasurable (it appears to blow up with
    // resolution). Second-order central differences are the non-aliased,
    // resolution-convergent measure of the direction-gradient energy, and they also
    // remove ~36 FFTs from this diagnostic. Vector <|grad xi|^2> and nematic
    // <|grad Xi|^2>, Xi_ab = xi_a xi_b (off-diagonal counted twice).
    let dx = grid.dx();
    let mut gxi2 = Array3::<f64>::zeros(pshape);
    for c in 0..3 {
        gxi2 += &grad_sq_periodic(&xi[c], dx);
    }
    let mut gnem2 = Array3::<f64>::zeros(pshape);
    for a in 0..3 {
        for b in a..3 {
            let xab = &xi[a] * &xi[b];
            let wt = if a == b { 1.0 } else { 2.0 };
            gnem2 += &grad_sq_periodic(&xab, dx).mapv(|v| wt * v);
        }
    }

    // BAND-LIMITED director-gradient energy, replacing the finite-difference one for
    // every (PAYOFF) quantity. Pointwise omega_i = rho xi_i gives
    //     d_j omega_i = (d_j rho) xi_i + rho d_j xi_i,
    // and sum_i xi_i d_j xi_i = (1/2) d_j |xi|^2 = 0 kills the cross term, so
    //     |grad omega|^2 = |grad rho|^2 + rho^2 |grad xi|^2      EXACTLY,
    // with d_j rho = (omega . d_j omega)/rho. Every ingredient is a spectral derivative
    // of the BAND-LIMITED omega, so this neither aliases (as a spectral derivative of
    // the non-band-limited xi would) nor damps (as the finite-difference derivative of
    // xi does: it recovers only ~0.36 of the true dissipation at n=64 and ~0.59 at
    // n=128 in a stressed Taylor-Green, see .wolf/buglog.json). Note rho^2 |grad xi|^2
    // needs no division and stays finite where the vorticity vanishes.
    let mut gw2 = Array3::<f64>::zeros(pshape);
    for i in 0..3 {
        for j in 0..3 {
            gw2 += &gradw(i, j).mapv(|x| x * x);
        }
    }
    let w2_safe = w2.mapv(|v| v.max(1e-300));
    let mut grho2 = Array3::<f64>::zeros(pshape);
    for j in 0..3 {
        let mut num = Array3::<f64>::zeros(pshape);
        for i in 0..3 {
            num += &(omega[i] * gradw(i, j));
        }
        grho2 += &(num.mapv(|x| x * x) / &w2_safe);
    }
    // rho^2 |grad xi|^2 and |grad xi|^2, both band-limited. The clamp is round-off only:
    // the identity makes the difference nonnegative exactly.
    let transverse_spec = (&gw2 - &grho2).mapv(|v| v.max(0.0));
    let gxi2_spec = &transverse_spec / &w2_safe;

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

    // (PAYOFF) instrumentation. The inequality under test is
    //     int rho^2 (xi . S xi)  <=  nu int rho^2 |grad xi|^2  +  subcritical,
    // whose two sides are exactly the production <omega . S omega> and nu times the
    // transverse dissipation <|omega|^2 |grad xi|^2>. The ratio of the two is the
    // measurement the specification calls for: the depletion saturates at rate 1/rho
    // precisely when this ratio stays bounded as the flow stresses.
    //
    // |grad xi|^2 comes from the finite-difference `gxi2`, not a spectral derivative,
    // because xi is not band-limited (see `grad_sq_periodic`). The full dissipation
    // <|grad omega|^2> is taken by Parseval in spectral space instead, where omega IS
    // band-limited, so it is exact and costs no transform.
    // The band-limited transverse dissipation is the one (PAYOFF) is tested with. The
    // finite-difference form is retained alongside it purely so the damping stays
    // visible in the output rather than being silently corrected away.
    let transverse = transverse_spec.clone();
    let transverse_fd = &w2 * &gxi2;
    let full_dissipation = {
        // Parseval as a RATIO, which is independent of the backend's FFT normalisation:
        //     <|grad omega|^2> / <|omega|^2>  =  sum k^2 |omega_hat|^2 / sum |omega_hat|^2
        // exactly. Multiplying by the physical-space enstrophy then fixes the scale
        // without assuming any transform convention. The r2c layout halves the LAST
        // axis (spectral_shape = (nx, ny, nz/2 + 1)), so iz = 0 and, for even nz, the
        // Nyquist iz = nz/2 are self-conjugate and count once; all other iz count twice.
        let nyq = grid.nz / 2;
        let even_nz = grid.nz.is_multiple_of(2);
        let mut num = 0.0;
        let mut den = 0.0;
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    let weight = if iz == 0 || (even_nz && iz == nyq) {
                        1.0
                    } else {
                        2.0
                    };
                    let k2 = ops.k_mag_sq[[ix, iy, iz]];
                    for c in 0..3 {
                        let w = omega_hat[c][[ix, iy, iz]];
                        let p = weight * (w.re * w.re + w.im * w.im);
                        num += k2 * p;
                        den += p;
                    }
                }
            }
        }
        if den > 0.0 {
            mean(&w2) * num / den
        } else {
            0.0
        }
    };
    // The transverse part uses a finite-difference |grad xi|^2, so the fraction it
    // carries must be measured against a finite-difference |grad omega|^2 as well.
    // Comparing an FD numerator to the spectral denominator would understate the
    // fraction, because second-order differences damp exactly the high-k content that
    // dominates the dissipation. omega is band-limited so FD on it is merely less
    // accurate, not aliased, and the like-for-like ratio is the meaningful one.
    let full_dissipation_fd = {
        let mut acc = Array3::<f64>::zeros(pshape);
        for c in 0..3 {
            acc += &grad_sq_periodic(omega[c], dx);
        }
        mean(&acc)
    };
    // THE CONDITIONAL TEST, binned on |omega|/max|omega|. The specification requires
    // `alpha <~ nu Phi` with `alpha = xi . S xi = (omega . S omega)/|omega|^2`, and that
    // is a claim about HIGH vorticity, where a singularity would form. The
    // volume-integrated ratio is dominated by the bulk and cannot see it. Counts cancel
    // in each conditional ratio, so it is a clean per-bin average:
    //     <alpha | rho> / (nu <Phi | rho>)  =  sum(alpha) / (nu sum(Phi))  over the bin.
    // Bins the BUDGET DENSITIES themselves, rho^2 alpha = omega . S omega and
    // rho^2 Phi = |grad omega|^2 - |grad rho|^2, rather than alpha and Phi separately.
    // Two reasons. First, (PAYOFF) is an inequality between those two integrals, so a
    // bin's ratio is literally that bin's contribution to it, and the whole-domain
    // payoff_ratio is the count-weighted combination of them. Second, and decisively,
    // neither density needs a division by rho^2, so the low-vorticity void where the
    // director spins arbitrarily fast (Phi unbounded as rho -> 0) stops contaminating
    // the fit. Logarithmic bins, since |omega| spans orders of magnitude.
    const NBIN: usize = 12;
    const BIN_FLOOR: f64 = 1e-3; // in units of max|omega|
    let wmax_safe = wmax.max(1e-30);
    let ln_floor = BIN_FLOOR.ln();
    let mut bin_prod = [0.0_f64; NBIN];
    let mut bin_trans = [0.0_f64; NBIN];
    let mut bin_full = [0.0_f64; NBIN];
    let mut bin_rho3 = [0.0_f64; NBIN];
    let mut bin_rho_sum = [0.0_f64; NBIN];
    let mut bin_n = [0.0_f64; NBIN];
    Zip::from(&wmag)
        .and(&prod)
        .and(&transverse_spec)
        .and(&gw2)
        .for_each(|&w, &p, &tr, &fu| {
            let frac = w / wmax_safe;
            if frac <= BIN_FLOOR {
                return;
            }
            let u = (frac.ln() - ln_floor) / (-ln_floor); // 0 at the floor, 1 at max
            let b = ((u * NBIN as f64) as usize).min(NBIN - 1);
            bin_prod[b] += p;
            bin_trans[b] += tr;
            bin_full[b] += fu;
            bin_rho3[b] += w * w * w;
            bin_rho_sum[b] += w;
            bin_n[b] += 1.0;
        });
    let bin_ratio: [f64; NBIN] = std::array::from_fn(|b| {
        if bin_trans[b] > 0.0 {
            bin_prod[b] / (nu * bin_trans[b])
        } else {
            f64::NAN
        }
    });
    let bin_rho: [f64; NBIN] = std::array::from_fn(|b| {
        if bin_n[b] > 0.0 {
            bin_rho_sum[b] / bin_n[b]
        } else {
            0.0
        }
    });

    // Count-weighted log-log fit of the per-bin ratio against the bin's mean |omega|,
    // with a standard error and an R^2 so a noisy fit can be recognised as noisy rather
    // than read as a trend. Bins carrying fewer than MIN_FRAC of the grid are dropped:
    // the top bins are thin and would otherwise dominate a slope while carrying almost
    // no budget. THE VERDICT: slope <= 0 means the (PAYOFF) violation does not worsen
    // with vorticity, so the depletion saturates where it matters; slope > 0 means it
    // worsens exactly where a singularity would form, refuting the route.
    const MIN_FRAC: f64 = 1e-4;
    let min_count = MIN_FRAC * n3;
    // Shared count-weighted log-log fit, run twice: once against the TRANSVERSE
    // dissipation, which is the (PAYOFF) statement, and once against the FULL
    // dissipation, which is the actual enstrophy budget. If the transverse slope is
    // positive while the full slope is not, then the longitudinal dissipation
    // <|grad rho|^2> is what protects the intense regions, and (PAYOFF) fails because it
    // discards exactly the term that does the work.
    let fit = |ratio: &[f64; NBIN]| -> (f64, f64, f64, f64) {
        let pts: Vec<(f64, f64, f64)> = (0..NBIN)
            .filter(|&b| bin_n[b] >= min_count && bin_rho[b] > 0.0 && ratio[b] > 0.0)
            .map(|b| (bin_rho[b].ln(), ratio[b].ln(), bin_n[b]))
            .collect();
        let n = pts.len() as f64;
        if pts.len() < 3 {
            return (f64::NAN, f64::NAN, f64::NAN, n);
        }
        let sw: f64 = pts.iter().map(|p| p.2).sum();
        let sx: f64 = pts.iter().map(|p| p.2 * p.0).sum();
        let sy: f64 = pts.iter().map(|p| p.2 * p.1).sum();
        let sxx: f64 = pts.iter().map(|p| p.2 * p.0 * p.0).sum();
        let sxy: f64 = pts.iter().map(|p| p.2 * p.0 * p.1).sum();
        let den = sw * sxx - sx * sx;
        if den.abs() < 1e-30 {
            return (f64::NAN, f64::NAN, f64::NAN, n);
        }
        let slope = (sw * sxy - sx * sy) / den;
        let intercept = (sxx * sy - sx * sxy) / den;
        let ybar = sy / sw;
        let ss_res: f64 = pts
            .iter()
            .map(|p| p.2 * (p.1 - intercept - slope * p.0).powi(2))
            .sum();
        let ss_tot: f64 = pts.iter().map(|p| p.2 * (p.1 - ybar).powi(2)).sum();
        let dof = (pts.len() - 2) as f64;
        let se = (ss_res / dof * sw / den).sqrt();
        let r2 = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            f64::NAN
        };
        (slope, se, r2, n)
    };
    let bin_ratio_full: [f64; NBIN] = std::array::from_fn(|b| {
        if bin_full[b] > 0.0 {
            bin_prod[b] / (nu * bin_full[b])
        } else {
            f64::NAN
        }
    });
    let (cond_slope_full, _, cond_r2_full, _) = fit(&bin_ratio_full);
    // Ghat = alpha/rho, the purely geometric (amplitude degree 0) factor in the exact
    // decomposition ratio = Ghat (l/l_nu)^2 of FA`ViscousLength. Taken budget-consistently
    // as sum(rho^2 alpha)/sum(rho^3) over the bin. Its slope tests the assumption that
    // Ghat is amplitude-flat; the length separation l/l_nu ~ rho^e then has
    // e = (slope - Ghat slope)/2, which is the viscous-length statement itself.
    // THREE INDEPENDENT STRUCTURE SCALES, to test whether e is physical or definitional.
    //   l1 = rho/|grad omega|        first order, mixes modulus and director
    //   l2 = rho/|grad rho|          first order, MODULUS ONLY, independent of the director
    //   l3 = Sqrt(rho/|Lap omega|)   SECOND order, independent of both first-order forms
    // Each is compared to l_nu = Sqrt(nu/rho) pointwise and averaged per bin, so e_i is the
    // exponent in <l_i/l_nu> ~ rho^(e_i). Only l1 satisfies the exact decomposition
    // ratio = Ghat (l1/l_nu)^2; l2 and l3 are genuinely separate probes of the same question.
    // Each length is defined from RATIOS OF SUMMED SQUARES within the bin, never as a
    // mean of a reciprocal: <rho/|grad omega|> is dominated by cells of near-zero
    // gradient and its arithmetic mean has a heavy tail (a first attempt gave spreads of
    // +-3.6 and disagreed with the decomposition on the SAME length). The budget-
    // consistent forms are well conditioned and are what makes
    // ratio = Ghat (l1/l_nu)^2 hold:
    //     l1 = Sqrt(sum rho^2 / sum |grad omega|^2)      first order, mixed
    //     l2 = Sqrt(sum rho^2 / sum |grad rho|^2)        first order, MODULUS ONLY
    //     l3 = (sum rho^2 / sum |Lap omega|^2)^(1/4)     SECOND order
    // and l_nu is taken at the bin's mean vorticity. e_i is the exponent in
    // l_i/l_nu ~ rho^(e_i).
    let mut bin_r2 = [0.0_f64; NBIN];
    let mut bin_gw2 = [0.0_f64; NBIN];
    let mut bin_gr2 = [0.0_f64; NBIN];
    let mut bin_lap2 = [0.0_f64; NBIN];
    {
        let mut lap2f = Array3::<f64>::zeros(pshape);
        for i in 0..3 {
            lap2f += &lapw(i).mapv(|x| x * x);
        }
        Zip::from(&wmag)
            .and(&w2)
            .and(&gw2)
            .and(&grho2)
            .and(&lap2f)
            .for_each(|&w, &w2v, &gw, &gr, &lp| {
                let frac = w / wmax_safe;
                if frac <= BIN_FLOOR {
                    return;
                }
                let u = (frac.ln() - ln_floor) / (-ln_floor);
                let b = ((u * NBIN as f64) as usize).min(NBIN - 1);
                bin_r2[b] += w2v;
                bin_gw2[b] += gw;
                bin_gr2[b] += gr;
                bin_lap2[b] += lp;
            });
    }
    let scale_exponent = |den: &[f64; NBIN], quartic: bool| -> f64 {
        let r: [f64; NBIN] = std::array::from_fn(|b| {
            if den[b] > 0.0 && bin_r2[b] > 0.0 && bin_rho[b] > 0.0 && nu > 0.0 {
                let q = bin_r2[b] / den[b];
                let l = if quartic { q.powf(0.25) } else { q.sqrt() };
                l / (nu / bin_rho[b]).sqrt()
            } else {
                f64::NAN
            }
        });
        fit(&r).0
    };

    let bin_ghat: [f64; NBIN] = std::array::from_fn(|b| {
        if bin_rho3[b] > 0.0 {
            bin_prod[b] / bin_rho3[b]
        } else {
            f64::NAN
        }
    });
    let (cond_ghat_slope, _, _, _) = fit(&bin_ghat);
    let pts: Vec<(f64, f64, f64)> = (0..NBIN)
        .filter(|&b| bin_n[b] >= min_count && bin_rho[b] > 0.0 && bin_ratio[b] > 0.0)
        .map(|b| (bin_rho[b].ln(), bin_ratio[b].ln(), bin_n[b]))
        .collect();
    let (cond_slope, cond_slope_stderr, cond_r2) = if pts.len() < 3 {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        let sw: f64 = pts.iter().map(|p| p.2).sum();
        let sx: f64 = pts.iter().map(|p| p.2 * p.0).sum();
        let sy: f64 = pts.iter().map(|p| p.2 * p.1).sum();
        let sxx: f64 = pts.iter().map(|p| p.2 * p.0 * p.0).sum();
        let sxy: f64 = pts.iter().map(|p| p.2 * p.0 * p.1).sum();
        let den = sw * sxx - sx * sx;
        if den.abs() < 1e-30 {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            let slope = (sw * sxy - sx * sy) / den;
            let intercept = (sxx * sy - sx * sxy) / den;
            let ybar = sy / sw;
            let ss_res: f64 = pts
                .iter()
                .map(|p| p.2 * (p.1 - intercept - slope * p.0).powi(2))
                .sum();
            let ss_tot: f64 = pts.iter().map(|p| p.2 * (p.1 - ybar).powi(2)).sum();
            let dof = (pts.len() - 2) as f64;
            let se = if dof > 0.0 {
                (ss_res / dof * sw / den).sqrt()
            } else {
                f64::NAN
            };
            let r2 = if ss_tot > 0.0 {
                1.0 - ss_res / ss_tot
            } else {
                f64::NAN
            };
            (slope, se, r2)
        }
    };
    let cond_nbins = pts.len() as f64;
    // The viscous-length statement itself: ratio = Ghat (l/l_nu)^2 gives
    // l/l_nu ~ rho^e with e = (slope - Ghat slope)/2.
    let cond_lratio_slope = (cond_slope - cond_ghat_slope) / 2.0;
    // Four coarse quartile summaries of the same per-bin ratios, for eyeballing.
    let quart = |lo: usize, hi: usize| -> f64 {
        let (mut p, mut t) = (0.0, 0.0);
        for b in lo..hi {
            p += bin_prod[b];
            t += bin_trans[b];
        }
        if t > 0.0 { p / (nu * t) } else { f64::NAN }
    };
    let cond_ratio = [quart(0, 3), quart(3, 6), quart(6, 9), quart(9, NBIN)];
    let cond_rho_top = *bin_rho.last().unwrap_or(&0.0);

    // <|grad omega|^2> from the physical-space gradients. This MUST agree with the
    // Parseval value above: they are the same quantity computed through independent
    // paths, so their agreement validates the spectral-derivative batch, the Hermitian
    // weighting and the transform normalisation at once. It is the check that the
    // finite-difference form silently failed.
    let full_dissipation_grad = mean(&gw2);
    let production = mean(&prod);
    let transverse_dissipation = mean(&transverse);
    let transverse_dissipation_fd = mean(&transverse_fd);
    let production_hi = masked_mean(&prod);
    let transverse_dissipation_hi = masked_mean(&transverse);
    let ratio = |p: f64, d: f64| p / (nu * d + 1e-300);

    FrameDiagnostics {
        time,
        step,
        enstrophy: mean(&w2),
        max_vorticity: wmax,
        f_rms,
        alpha_p_rms,
        rho_all: alpha_p_rms / (f_rms + 1e-30),
        rho_hi: masked_rms(&alpha_p) / (masked_rms(&f) + 1e-30),
        xi_energy: mean(&gxi2_spec),
        xi_energy_fd: mean(&gxi2),
        full_dissipation_grad,
        transverse_dissipation_fd,
        fd_recovery: full_dissipation_fd / (full_dissipation + 1e-300),
        parseval_residual: (full_dissipation_grad - full_dissipation).abs()
            / (full_dissipation + 1e-300),
        nem_energy: mean(&gnem2),
        xi_energy_hi: masked_mean(&gxi2),
        nem_energy_hi: masked_mean(&gnem2),
        coherence_w: mean(&(&wmag.mapv(f64::sqrt) * &gnem2)),
        hi_fraction: mask.sum() / n3,
        nu,
        production,
        transverse_dissipation,
        full_dissipation,
        payoff_ratio: ratio(production, transverse_dissipation),
        production_hi,
        transverse_dissipation_hi,
        payoff_ratio_hi: ratio(production_hi, transverse_dissipation_hi),
        transverse_fraction: transverse_dissipation / (full_dissipation + 1e-300),
        cond_ratio_q1: cond_ratio[0],
        cond_ratio_q2: cond_ratio[1],
        cond_ratio_q3: cond_ratio[2],
        cond_ratio_q4: cond_ratio[3],
        cond_rho_q4: cond_rho_top,
        cond_slope,
        cond_slope_stderr,
        cond_r2,
        cond_nbins,
        cond_slope_full,
        cond_r2_full,
        cond_ghat_slope,
        cond_lratio_slope,
        e_grad_omega: scale_exponent(&bin_gw2, false),
        e_grad_rho: scale_exponent(&bin_gr2, false),
        e_laplacian: scale_exponent(&bin_lap2, true),
    }
}

/// `|grad field|^2` by second-order periodic central differences (physical space),
/// parallelised across the grid. Used for the coherence energies: the direction
/// field is not band-limited, so a spectral derivative would alias; finite
/// differences give the non-aliased, resolution-convergent gradient energy.
fn grad_sq_periodic(field: &Array3<f64>, dx: f64) -> Array3<f64> {
    let (nx, ny, nz) = field.dim();
    let inv = 1.0 / (2.0 * dx);
    let mut out = Array3::<f64>::zeros((nx, ny, nz));
    Zip::indexed(&mut out).par_for_each(|(i, j, k), o| {
        let ip = if i + 1 == nx { 0 } else { i + 1 };
        let im = if i == 0 { nx - 1 } else { i - 1 };
        let jp = if j + 1 == ny { 0 } else { j + 1 };
        let jm = if j == 0 { ny - 1 } else { j - 1 };
        let kp = if k + 1 == nz { 0 } else { k + 1 };
        let km = if k == 0 { nz - 1 } else { k - 1 };
        // SAFETY: ip/im/jp/jm/kp/km are all in [0, n) for their axis by construction.
        unsafe {
            let dfx = (*field.uget([ip, j, k]) - *field.uget([im, j, k])) * inv;
            let dfy = (*field.uget([i, jp, k]) - *field.uget([i, jm, k])) * inv;
            let dfz = (*field.uget([i, j, kp]) - *field.uget([i, j, km])) * inv;
            *o = dfx * dfx + dfy * dfy + dfz * dfz;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use vonkarman_core::domain::Domain;

    /// The (PAYOFF) instrumentation must respect the decomposition
    /// `|grad omega|^2 = |grad |omega||^2 + |omega|^2 |grad xi|^2`, so the full
    /// dissipation dominates its own transverse part. This is the check that catches a
    /// wrong Parseval normalisation or a Hermitian weight applied to the wrong axis:
    /// either error moves `full_dissipation` by a large factor while the physical-space
    /// `transverse_dissipation` is unaffected.
    #[test]
    fn payoff_instrumentation_is_consistent() {
        let n = 32;
        let nu = 0.01;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let solver = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
        let fd = solver.frame_diagnostics();

        assert!(fd.enstrophy > 0.0, "enstrophy should be positive");
        assert!(
            fd.transverse_dissipation > 0.0,
            "transverse dissipation should be positive"
        );
        assert!(
            fd.full_dissipation >= fd.transverse_dissipation * (1.0 - 1e-9),
            "full dissipation {} must dominate its transverse part {}",
            fd.full_dissipation,
            fd.transverse_dissipation
        );
        assert!(
            fd.transverse_fraction > 0.0 && fd.transverse_fraction <= 1.0 + 1e-9,
            "transverse fraction {} must lie in (0, 1]",
            fd.transverse_fraction
        );
        assert!(fd.production.is_finite(), "production must be finite");
        assert!(fd.payoff_ratio.is_finite(), "payoff ratio must be finite");
        assert_eq!(fd.nu, nu, "nu must be carried through");

        // The conditional bins must be populated and ordered in amplitude: the top bin
        // is conditioned on the largest |omega|, so its mean vorticity must exceed the
        // whole-field RMS. A binning error (wrong index, wrong normalisation) breaks
        // this immediately.
        assert!(
            fd.cond_rho_q4 > fd.enstrophy.sqrt(),
            "top-bin mean |omega| {} should exceed the RMS {}",
            fd.cond_rho_q4,
            fd.enstrophy.sqrt()
        );
        assert!(
            fd.cond_rho_q4 <= fd.max_vorticity * (1.0 + 1e-9),
            "top-bin mean |omega| cannot exceed max |omega|"
        );

        // THE CROSS-CHECK. <|grad omega|^2> is computed twice by independent paths:
        // Parseval in spectral space, and the sum of squared physical-space spectral
        // derivatives. They must agree. A wrong Hermitian weight, a wrong transform
        // normalisation or a mis-indexed derivative batch breaks this immediately, and
        // it is exactly the check the finite-difference estimator never had.
        // Tight at t = 0, where the Taylor-Green field is smooth and the grid quadrature
        // of the squared gradients is essentially exact. Along a stressed trajectory the
        // residual grows to ~1e-2, because (d omega)^2 carries content to 2 k_max and the
        // grid sum of it aliases; that is a quadrature effect on the cross-check only,
        // not on the Parseval value, and it is two orders below the damping it replaced.
        assert!(
            fd.parseval_residual < 1e-10,
            "Parseval and gradient paths to <|grad omega|^2> disagree by {} ({} vs {})",
            fd.parseval_residual,
            fd.full_dissipation,
            fd.full_dissipation_grad
        );
        // The band-limited identity is exact, so the transverse part cannot exceed the
        // whole, and the finite-difference form must sit below the band-limited one.
        assert!(
            fd.transverse_dissipation <= fd.full_dissipation * (1.0 + 1e-9),
            "transverse {} cannot exceed full {}",
            fd.transverse_dissipation,
            fd.full_dissipation
        );
        assert!(
            fd.fd_recovery > 0.0 && fd.fd_recovery <= 1.0 + 1e-9,
            "finite-difference recovery {} must lie in (0, 1]",
            fd.fd_recovery
        );
    }

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
