//! vonkarman-2d: 2D pseudo-spectral incompressible Navier-Stokes on the torus
//! T^2, vorticity-streamfunction form, with passive scalar (dye) transport.
//!
//! Sibling of the 3D `vonkarman` solver, sharing its pseudo-spectral
//! conventions: state lives in spectral space, the nonlinear term is formed in
//! physical space and dealiased (2/3 rule), and viscosity is integrated exactly
//! via an integrating factor (here IF-RK4). Because 2D is cheap, it runs at
//! 1024^2+ where the fine filaments are genuinely RESOLVED, so passive dye
//! advecting through it looks like the "incredible" graphics-fluid visuals
//! without any noise-upres or vorticity-confinement cheats.
//!
//! Two NON-cancelling dyes (gold density and rust density, each >= 0) are
//! advected, so turbulent mixing reads as interleaved filaments and an orange
//! blend rather than fading to black (which a single signed dye would do).
//!
//! Vorticity transport:  d(omega)/dt + u . grad(omega) = nu lap(omega)
//! Dye transport:        d(c)/dt     + u . grad(c)     = kappa lap(c)
//! with u = (d(psi)/dy, -d(psi)/dx) and lap(psi) = -omega.

use image::{ImageBuffer, Rgb};
use ndarray::{Array2, Zip};
use ndrustfft::{FftHandler, R2cFftHandler, ndfft, ndfft_r2c, ndifft, ndifft_r2c};
use num_complex::Complex;

mod body;
pub use body::Penalisation;

pub mod export;

type C = Complex<f64>;

/// Spectral operators on an `nx x ny` periodic grid over `[0, lx) x [0, ly)`.
///
/// Axis 0 is `x` and carries a full complex FFT; axis 1 is `y` and carries a
/// real-to-complex FFT, so spectral arrays have shape `(nx, ny/2 + 1)` and
/// physical arrays have shape `(nx, ny)`.
pub struct Spectral2D {
    nx: usize,
    ny: usize,
    nyh: usize, // ny/2 + 1 (rFFT half-spectrum length)
    lx: f64,
    ly: f64,
    hx: FftHandler<f64>,    // axis 0: full complex FFT (kx)
    hy: R2cFftHandler<f64>, // axis 1: real-to-complex FFT (ky)
    kx: Array2<f64>,
    ky: Array2<f64>,
    k2: Array2<f64>,
    k2inv: Array2<f64>,   // 1/|k|^2, zero at the mean mode
    dealias: Array2<f64>, // 2/3-rule mask (1 keep, 0 drop)
}

impl Spectral2D {
    /// Build operators for an `nx x ny` grid over a box of size `lx x ly`.
    ///
    /// Wavenumbers are physical: `kx = 2 pi m / lx` with `m` the signed mode
    /// index, `ky = 2 pi j / ly`. The 2/3 dealias mask is applied in mode-index
    /// space, so it is independent of the box size.
    pub fn new(nx: usize, ny: usize, lx: f64, ly: f64) -> Self {
        assert!(nx >= 4 && ny >= 4, "grid must be at least 4 x 4");
        assert!(lx > 0.0 && ly > 0.0, "box lengths must be positive");
        let nyh = ny / 2 + 1;
        let mut kx = Array2::<f64>::zeros((nx, nyh));
        let mut ky = Array2::<f64>::zeros((nx, nyh));
        let mut k2 = Array2::<f64>::zeros((nx, nyh));
        let mut k2inv = Array2::<f64>::zeros((nx, nyh));
        let mut dealias = Array2::<f64>::zeros((nx, nyh));
        let (cutx, cuty) = (nx as f64 / 3.0, ny as f64 / 3.0);
        let (fx, fy) = (std::f64::consts::TAU / lx, std::f64::consts::TAU / ly);
        for i in 0..nx {
            let m = if i <= nx / 2 {
                i as f64
            } else {
                i as f64 - nx as f64
            };
            for j in 0..nyh {
                let (kxi, kyj) = (fx * m, fy * j as f64);
                kx[[i, j]] = kxi;
                ky[[i, j]] = kyj;
                let kk = kxi * kxi + kyj * kyj;
                k2[[i, j]] = kk;
                k2inv[[i, j]] = if kk > 0.0 { 1.0 / kk } else { 0.0 };
                dealias[[i, j]] = if m.abs() <= cutx && (j as f64) <= cuty {
                    1.0
                } else {
                    0.0
                };
            }
        }
        Self {
            nx,
            ny,
            nyh,
            lx,
            ly,
            hx: FftHandler::new(nx),
            hy: R2cFftHandler::new(ny),
            kx,
            ky,
            k2,
            k2inv,
            dealias,
        }
    }

    /// The legacy square box `n x n` over `[0, 2 pi)^2`.
    pub fn new_square(n: usize) -> Self {
        Self::new(n, n, std::f64::consts::TAU, std::f64::consts::TAU)
    }

    /// Grid points along `x`.
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Grid points along `y`.
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Box length along `x`.
    pub fn lx(&self) -> f64 {
        self.lx
    }

    /// Box length along `y`.
    pub fn ly(&self) -> f64 {
        self.ly
    }

    /// Grid spacing `(dx, dy)`.
    pub fn spacing(&self) -> (f64, f64) {
        (self.lx / self.nx as f64, self.ly / self.ny as f64)
    }

    /// Squared wavenumber magnitude over the half-spectrum.
    pub fn k2(&self) -> &Array2<f64> {
        &self.k2
    }

    /// Real field -> spectral (rFFT along axis 1, complex FFT along axis 0).
    pub fn forward(&self, real: &Array2<f64>) -> Array2<C> {
        let mut tmp = Array2::<C>::zeros((self.nx, self.nyh));
        ndfft_r2c(real, &mut tmp, &self.hy, 1);
        let mut out = Array2::<C>::zeros((self.nx, self.nyh));
        ndfft(&tmp, &mut out, &self.hx, 0);
        out
    }

    /// Spectral -> real field (inverse of [`Self::forward`], normalised).
    pub fn inverse(&self, spec: &Array2<C>) -> Array2<f64> {
        let mut tmp = Array2::<C>::zeros((self.nx, self.nyh));
        ndifft(spec, &mut tmp, &self.hx, 0);
        let mut out = Array2::<f64>::zeros((self.nx, self.ny));
        ndifft_r2c(&tmp, &mut out, &self.hy, 1);
        out
    }

    /// Spectral streamfunction from spectral vorticity: `psi_hat = omega_hat / |k|^2`.
    ///
    /// The mean mode is left at zero, so this is the fluctuating part only.
    pub fn streamfunction_hat(&self, what: &Array2<C>) -> Array2<C> {
        let mut ph = Array2::<C>::zeros((self.nx, self.nyh));
        Zip::from(&mut ph)
            .and(what)
            .and(&self.k2inv)
            .for_each(|p, &w, &k2i| *p = w * k2i);
        ph
    }

    /// Physical velocity `(u, v)` from spectral vorticity:
    /// `u = +d(psi)/dy`, `v = -d(psi)/dx`, `psi_hat = omega_hat / |k|^2`.
    pub fn velocity(&self, what: &Array2<C>) -> (Array2<f64>, Array2<f64>) {
        let mut uh = Array2::<C>::zeros((self.nx, self.nyh));
        let mut vh = Array2::<C>::zeros((self.nx, self.nyh));
        let i = C::new(0.0, 1.0);
        Zip::from(&mut uh)
            .and(&mut vh)
            .and(what)
            .and(&self.kx)
            .and(&self.ky)
            .and(&self.k2inv)
            .for_each(|u, v, &w, &kx, &ky, &k2i| {
                let psi = w * k2i;
                *u = i * ky * psi; //  i ky psi  =  d/dy
                *v = -i * kx * psi; // -i kx psi  = -d/dx
            });
        (self.inverse(&uh), self.inverse(&vh))
    }

    /// Spectral vorticity from a physical velocity field:
    /// `omega = d(v)/dx - d(u)/dy`.
    ///
    /// Taking the curl discards any divergent part, so this doubles as the
    /// Helmholtz projection after a penalisation substep.
    pub fn curl(&self, u: &Array2<f64>, v: &Array2<f64>) -> Array2<C> {
        let (uh, vh) = (self.forward(u), self.forward(v));
        let mut wh = Array2::<C>::zeros((self.nx, self.nyh));
        let i = C::new(0.0, 1.0);
        Zip::from(&mut wh)
            .and(&uh)
            .and(&vh)
            .and(&self.kx)
            .and(&self.ky)
            .for_each(|w, &u, &v, &kx, &ky| *w = i * kx * v - i * ky * u);
        wh
    }

    /// Dealiased advective nonlinear term `-(u . grad s)` in spectral space for
    /// a scalar `s` (vorticity or dye) given the physical velocity.
    fn advect(&self, shat: &Array2<C>, u: &Array2<f64>, v: &Array2<f64>) -> Array2<C> {
        let i = C::new(0.0, 1.0);
        let mut sxh = Array2::<C>::zeros((self.nx, self.nyh));
        let mut syh = Array2::<C>::zeros((self.nx, self.nyh));
        Zip::from(&mut sxh)
            .and(&mut syh)
            .and(shat)
            .and(&self.kx)
            .and(&self.ky)
            .for_each(|sx, sy, &s, &kx, &ky| {
                *sx = i * kx * s;
                *sy = i * ky * s;
            });
        let sx = self.inverse(&sxh);
        let sy = self.inverse(&syh);
        let mut adv = Array2::<f64>::zeros((self.nx, self.ny));
        Zip::from(&mut adv)
            .and(u)
            .and(v)
            .and(&sx)
            .and(&sy)
            .for_each(|a, &u, &v, &sx, &sy| *a = -(u * sx + v * sy));
        let mut nh = self.forward(&adv);
        Zip::from(&mut nh)
            .and(&self.dealias)
            .for_each(|c, &m| *c *= m);
        nh
    }
}

/// Real integrating-factor `exp(-coeff * |k|^2 * dt)` over the half-spectrum.
fn integ_factor(s: &Spectral2D, coeff: f64, dt: f64) -> Array2<f64> {
    s.k2().mapv(|k2| (-coeff * k2 * dt).exp())
}

#[inline]
fn scale(a: &Array2<C>, r: &Array2<f64>) -> Array2<C> {
    let mut o = a.clone();
    Zip::from(&mut o).and(r).for_each(|c, &x| *c *= x);
    o
}

/// A running 2D simulation: vorticity + two dye fields and the fixed-dt IF-RK4
/// maps. Gold and rust dye are advected by the same flow and never cancel.
pub struct Sim {
    pub spec: Spectral2D,
    pub wh: Array2<C>, // vorticity (spectral)
    pub gh: Array2<C>, // gold dye (spectral)
    pub rh: Array2<C>, // rust dye (spectral)
    dt: f64,
    ew: Array2<f64>,
    ew2: Array2<f64>,
    ec: Array2<f64>,
    ec2: Array2<f64>,
    /// Steady prescribed downward jet velocity (physical, `nx x ny`, the `v`
    /// component) added to the advecting velocity. A divergence-free curtain
    /// `v = v(x)` in a vertical strip that cuts through the dye top-to-bottom.
    v_jet: Array2<f64>,
    /// Uniform stream speed along `+x`, added to the advecting velocity and to
    /// any reported total velocity. It carries no vorticity, so it is held
    /// outside the state.
    u_mean: f64,
    /// Optional penalised body and downstream fringe.
    body: Option<Penalisation>,
    /// Cached `exp(-chi * dt / (2 eta_p))` for the Strang half substep.
    vel_decay: Option<Array2<f64>>,
    /// Cached `exp(-sigma * dt / 2)` for the Strang half substep.
    vort_decay: Option<Array2<f64>>,
    /// Momentum removed by the penalisation substeps so far in the step in
    /// progress, `(dpx, dpy)`. Zeroed at the start of a stepped `step()` and
    /// accumulated once per half substep; divided by `dt` at the end of the
    /// step to give [`Self::body_force`].
    force_acc: (f64, f64),
    /// Force per unit span on the body over the last completed step, from
    /// [`Self::body_force`].
    last_body_force: (f64, f64),
}

impl Sim {
    pub fn new(
        spec: Spectral2D,
        wh: Array2<C>,
        gh: Array2<C>,
        rh: Array2<C>,
        dt: f64,
        nu: f64,
        kappa: f64,
    ) -> Self {
        let ew = integ_factor(&spec, nu, dt);
        let ew2 = integ_factor(&spec, nu, dt * 0.5);
        let ec = integ_factor(&spec, kappa, dt);
        let ec2 = integ_factor(&spec, kappa, dt * 0.5);
        let v_jet = Array2::<f64>::zeros((spec.nx(), spec.ny()));
        Self {
            spec,
            wh,
            gh,
            rh,
            dt,
            ew,
            ew2,
            ec,
            ec2,
            v_jet,
            u_mean: 0.0,
            body: None,
            vel_decay: None,
            vort_decay: None,
            force_acc: (0.0, 0.0),
            last_body_force: (0.0, 0.0),
        }
    }

    /// Set a steady downward jet: a smooth top-hat vertical strip of width
    /// `width` centred at `centre` (in `x`), full height, flowing down at
    /// `speed`. Added to the advecting velocity, so it drags vorticity and dye
    /// downward where it passes, cutting a swath through the swirl. `v = v(x)`
    /// is divergence-free, so it needs no pressure correction.
    pub fn set_jet(&mut self, centre: f64, width: f64, speed: f64) {
        let (nx, ny) = (self.spec.nx(), self.spec.ny());
        let (dx, _dy) = self.spec.spacing();
        let edge = 0.12; // smoothing width of the strip walls
        let (lo, hi) = (centre - 0.5 * width, centre + 0.5 * width);
        for i in 0..nx {
            let x = i as f64 * dx;
            // smooth top-hat in x: 1 inside the strip, 0 outside
            let th = 0.5 * (((x - lo) / edge).tanh() - ((x - hi) / edge).tanh());
            for j in 0..ny {
                self.v_jet[[i, j]] = -speed * th; // downward (-y)
            }
        }
    }

    /// Set the uniform stream speed along `+x`.
    pub fn set_mean_flow(&mut self, u_mean: f64) {
        self.u_mean = u_mean;
    }

    /// The uniform stream speed along `+x`.
    pub fn mean_flow(&self) -> f64 {
        self.u_mean
    }

    /// Physical vorticity.
    pub fn vorticity(&self) -> Array2<f64> {
        self.spec.inverse(&self.wh)
    }

    /// Physical total velocity: the vortical part plus the uniform stream.
    pub fn total_velocity(&self) -> (Array2<f64>, Array2<f64>) {
        let (mut u, v) = self.spec.velocity(&self.wh);
        u += self.u_mean;
        (u, v)
    }

    /// Attach a penalised body. The Strang half-step decay factors are cached
    /// here, so this must be called after the time step is fixed.
    pub fn set_body(&mut self, body: Penalisation) {
        let h = 0.5 * self.dt;
        self.vel_decay = Some(body.velocity_decay(h));
        self.vort_decay = Some(body.vorticity_decay(h));
        self.body = Some(body);
    }

    /// The attached body, if any.
    pub fn body(&self) -> Option<&Penalisation> {
        self.body.as_ref()
    }

    /// Force per unit span on the body over the last completed step, `(fx, fy)`.
    ///
    /// Measured as the momentum the penalisation substeps actually removed
    /// from the flow, divided by `dt`: `dp = integral(u_before * (1 -
    /// vel_decay))` over each half substep, summed over both halves of the
    /// step, then `F = dp / dt`. This is exact for the exponential substep as
    /// implemented, unlike the Angot estimator `(1 / eta_p) * integral(chi *
    /// u)`, which assumes an explicit penalisation term and diverges under an
    /// exponential one (its `1 / eta_p` prefactor grows faster than the
    /// velocity surviving across the smoothed mask edge shrinks).
    ///
    /// Returns `(0.0, 0.0)` when no body is attached.
    pub fn body_force(&self) -> (f64, f64) {
        if self.body.is_some() {
            self.last_body_force
        } else {
            (0.0, 0.0)
        }
    }

    /// One Strang half substep: relax the total velocity toward rest inside the
    /// body, reform vorticity by a spectral curl (which projects out the
    /// divergent part the multiplication introduced), then relax vorticity in
    /// the fringe. Both factors are exact solutions of their term.
    ///
    /// `u_mean` is folded into the velocity before the decay multiply, not
    /// after: the mask gradient crossed with the oncoming stream is what
    /// generates vorticity at the body, so decaying only the vortical part
    /// would leave `curl` at zero wherever the vortical velocity started at
    /// zero, and a body that never sheds.
    ///
    /// Also accumulates the momentum this half substep removes from the flow
    /// into `force_acc`, for [`Self::body_force`]: per point, the velocity
    /// removed is `u_before * (1 - vel_decay)`, and `u_before - u_after =
    /// u_before * (1 - vel_decay)` exactly, so the removed velocity is read
    /// off during the same pass that applies the decay, before it is
    /// overwritten, then weighted by the cell area to turn the grid sum into
    /// an integral.
    fn penalisation_half_step(&mut self) {
        let (Some(vd), Some(wd)) = (self.vel_decay.as_ref(), self.vort_decay.as_ref()) else {
            return;
        };
        let (mut u, mut v) = self.spec.velocity(&self.wh);
        u += self.u_mean;
        let mut dpx = 0.0_f64;
        let mut dpy = 0.0_f64;
        Zip::from(&mut u).and(&mut v).and(vd).for_each(|u, v, &f| {
            let (ub, vb) = (*u, *v);
            *u = ub * f;
            *v = vb * f;
            dpx += ub * (1.0 - f);
            dpy += vb * (1.0 - f);
        });
        let area = self
            .body
            .as_ref()
            .expect("vel_decay implies body")
            .cell_area();
        self.force_acc.0 += dpx * area;
        self.force_acc.1 += dpy * area;
        let mut wh = self.spec.curl(&u, &v);
        let w = self.spec.inverse(&wh);
        let mut damped = w;
        Zip::from(&mut damped).and(wd).for_each(|w, &f| *w *= f);
        wh = self.spec.forward(&damped);
        self.wh = wh;
    }

    /// One full step: a Strang-split penalisation half substep either side of
    /// the IF-RK4 step when a body is attached, otherwise the IF-RK4 step alone.
    ///
    /// When a body is attached, also zeroes the force accumulator before the
    /// first half substep and, after the second, converts the accumulated
    /// momentum into a force by dividing by `dt`, ready for
    /// [`Self::body_force`].
    pub fn step(&mut self) {
        if self.body.is_some() {
            self.force_acc = (0.0, 0.0);
            self.penalisation_half_step();
            self.step_ifrk4();
            self.penalisation_half_step();
            self.last_body_force = (self.force_acc.0 / self.dt, self.force_acc.1 / self.dt);
        } else {
            self.step_ifrk4();
        }
    }

    /// One IF-RK4 step for vorticity + the two dyes, sharing one velocity per
    /// stage. Standard integrating-factor RK4 (reduces to RK4 with zero
    /// viscosity, exact with zero nonlinear term).
    fn step_ifrk4(&mut self) {
        let dt = self.dt;
        let s = &self.spec;
        let vjet = &self.v_jet;
        // Nonlinear RHS from one velocity per stage. The steady jet is added to
        // the DYE advection only (not the vorticity), so the vortices keep their
        // true orbit-and-fray dynamics while the jet drags the colours downward
        // in its strip, cutting a vertical streak through the swirl.
        let u_mean = self.u_mean;
        let rhs = |w: &Array2<C>, g: &Array2<C>, r: &Array2<C>| {
            let (mut u, v) = s.velocity(w);
            u += u_mean;
            let mut vd = v.clone();
            Zip::from(&mut vd).and(vjet).for_each(|vd, &vj| *vd += vj);
            (
                s.advect(w, &u, &v),
                s.advect(g, &u, &vd),
                s.advect(r, &u, &vd),
            )
        };
        // per-field stage builders (E = field's integrating factor)
        let st2 = |e2: &Array2<f64>, base: &Array2<C>, k: &Array2<C>| {
            scale(&(base + &k.mapv(|c| c * (dt * 0.5))), e2)
        };
        let st3 = |e2: &Array2<f64>, base: &Array2<C>, k: &Array2<C>| {
            &scale(base, e2) + &k.mapv(|c| c * (dt * 0.5))
        };
        let st4 = |e: &Array2<f64>, e2: &Array2<f64>, base: &Array2<C>, k: &Array2<C>| {
            &scale(base, e) + &scale(k, e2).mapv(|c| c * dt)
        };
        let fin = |e: &Array2<f64>,
                   e2: &Array2<f64>,
                   base: &Array2<C>,
                   k1: &Array2<C>,
                   k2: &Array2<C>,
                   k3: &Array2<C>,
                   k4: &Array2<C>| {
            let term = scale(k1, e) + scale(&(k2 + k3), e2).mapv(|c| c * 2.0) + k4;
            let mut out = scale(base, e);
            Zip::from(&mut out)
                .and(&term)
                .for_each(|o, &t| *o += t * (dt / 6.0));
            out
        };

        let (ew, ew2, ec, ec2) = (&self.ew, &self.ew2, &self.ec, &self.ec2);
        let (k1w, k1g, k1r) = rhs(&self.wh, &self.gh, &self.rh);
        let (a2w, a2g, a2r) = (
            st2(ew2, &self.wh, &k1w),
            st2(ec2, &self.gh, &k1g),
            st2(ec2, &self.rh, &k1r),
        );
        let (k2w, k2g, k2r) = rhs(&a2w, &a2g, &a2r);
        let (a3w, a3g, a3r) = (
            st3(ew2, &self.wh, &k2w),
            st3(ec2, &self.gh, &k2g),
            st3(ec2, &self.rh, &k2r),
        );
        let (k3w, k3g, k3r) = rhs(&a3w, &a3g, &a3r);
        let (a4w, a4g, a4r) = (
            st4(ew, ew2, &self.wh, &k3w),
            st4(ec, ec2, &self.gh, &k3g),
            st4(ec, ec2, &self.rh, &k3r),
        );
        let (k4w, k4g, k4r) = rhs(&a4w, &a4g, &a4r);

        let nw = fin(ew, ew2, &self.wh, &k1w, &k2w, &k3w, &k4w);
        let ng = fin(ec, ec2, &self.gh, &k1g, &k2g, &k3g, &k4g);
        let nr = fin(ec, ec2, &self.rh, &k1r, &k2r, &k3r, &k4r);
        self.wh = nw;
        self.gh = ng;
        self.rh = nr;
    }

    /// Physical gold and rust dye fields (real space), for rendering.
    pub fn dyes(&self) -> (Array2<f64>, Array2<f64>) {
        (self.spec.inverse(&self.gh), self.spec.inverse(&self.rh))
    }

    /// Total physical streamfunction: the vortical part from the Poisson solve
    /// plus the mean-flow part `u_mean * y`.
    ///
    /// The mean part is non-periodic in `y` and is added analytically, which is
    /// why a uniform stream draws as straight parallel streamlines.
    pub fn streamfunction(&self) -> Array2<f64> {
        let mut psi = self.spec.inverse(&self.spec.streamfunction_hat(&self.wh));
        let (_dx, dy) = self.spec.spacing();
        for j in 0..self.spec.ny() {
            let y = j as f64 * dy;
            for i in 0..self.spec.nx() {
                psi[[i, j]] += self.u_mean * y;
            }
        }
        psi
    }
}

/// Two co-rotating Lamb-Oseen vortices: gold dye in one, rust-red in the other.
/// They orbit cleanly, then (with the band-limited noise seeding a shear
/// instability at high Re) shed and fray the dyes into interleaving filaments.
///
/// Returns spectral `(omega_hat, gold_hat, rust_hat)`; the dyes are independent
/// non-negative densities. A slight strength asymmetry plus the noise breaks the
/// symmetry so an instability actually grows.
pub fn co_rotating_vortices(
    spec: &Spectral2D,
    sep: f64,
    core: f64,
    circ: f64,
    noise: f64,
    seed: u64,
) -> (Array2<C>, Array2<C>, Array2<C>) {
    let (nx, ny) = (spec.nx(), spec.ny());
    let (dx, dy) = spec.spacing();
    let (cx, cy) = (0.5 * spec.lx(), 0.5 * spec.ly());
    let amp = circ / (std::f64::consts::PI * core * core);
    let mut omega = Array2::<f64>::zeros((nx, ny));
    let mut gold = Array2::<f64>::zeros((nx, ny));
    let mut rust = Array2::<f64>::zeros((nx, ny));
    let centres = [
        (cx - 0.5 * sep, cy, 1.0_f64),
        (cx + 0.5 * sep, cy, 0.97_f64),
    ];
    let rd = 1.1 * core;
    for i in 0..nx {
        let x = i as f64 * dx;
        for j in 0..ny {
            let y = j as f64 * dy;
            let mut w = 0.0;
            for (vc, &(px, py, str)) in centres.iter().enumerate() {
                let r2 = (x - px) * (x - px) + (y - py) * (y - py);
                w += str * amp * (-r2 / (core * core)).exp();
                let d = (-(r2 / (rd * rd)).powi(2)).exp();
                if vc == 0 {
                    gold[[i, j]] += d;
                } else {
                    rust[[i, j]] += d;
                }
            }
            omega[[i, j]] = w;
        }
    }
    if noise > 0.0 {
        let mut st = seed | 1;
        let mut rand = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as f64 / (1u64 << 31) as f64 // [0, 2)
        };
        let (kmin, kmax) = (3.0, 10.0);
        let mut modes: Vec<(f64, f64, f64)> = Vec::new();
        while modes.len() < 36 {
            let kxm = (rand() * kmax - kmax * 0.5).round() * 2.0;
            let kym = (rand() * kmax - kmax * 0.5).round() * 2.0;
            let kk = (kxm * kxm + kym * kym).sqrt();
            if kk >= kmin && kk <= kmax {
                modes.push((kxm, kym, rand() * std::f64::consts::PI));
            }
        }
        let scale = noise * amp / (modes.len() as f64).sqrt();
        for i in 0..nx {
            let x = i as f64 * dx;
            for j in 0..ny {
                let y = j as f64 * dy;
                let mut sm = 0.0;
                for &(kxm, kym, ph) in &modes {
                    sm += (kxm * x + kym * y + ph).cos();
                }
                omega[[i, j]] += scale * sm;
            }
        }
    }
    (
        spec.forward(&omega),
        spec.forward(&gold),
        spec.forward(&rust),
    )
}

/// Write the two dye densities to a PNG. On a black background gold + rust are
/// added (overlap blends to orange) with a gamma lift so thin frayed filaments
/// glow. On a white background the dyes composite as translucent inks (opacity
/// from density), an "ink in water on white paper" look; overlaps tend toward
/// orange and faint filaments read as light tints.
pub fn write_dye_png(gold: &Array2<f64>, rust: &Array2<f64>, path: &str, white_bg: bool) {
    let nx = gold.shape()[0] as u32;
    let ny = gold.shape()[1] as u32;
    let gcol = [1.0_f64, 0.78, 0.23];
    let rcol = [0.72_f64, 0.25, 0.05];
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(nx, ny);
    for (px, py, p) in img.enumerate_pixels_mut() {
        let i = px as usize;
        let j = (ny - 1 - py) as usize; // flip y for an upright image
        let g = gold[[i, j]].max(0.0);
        let r = rust[[i, j]].max(0.0);
        let mut rgb = [0u8; 3];
        if white_bg {
            let tot = g + r;
            let a = tot.min(1.0).powf(0.6); // opacity; lifts faint filaments
            for k in 0..3 {
                let ink = if tot > 1e-6 {
                    (gcol[k] * g + rcol[k] * r) / tot
                } else {
                    1.0
                };
                let v = (1.0 - a) + ink * a; // composite over white
                rgb[k] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        } else {
            for k in 0..3 {
                let v = (gcol[k] * g + rcol[k] * r).clamp(0.0, 1.0).powf(0.72);
                rgb[k] = (v * 255.0).round() as u8;
            }
        }
        *p = Rgb(rgb);
    }
    img.save(path).expect("write_dye_png: save failed");
}
