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

type C = Complex<f64>;

/// Spectral operators on an `n x n` periodic grid over `[0, 2*pi)^2`.
pub struct Spectral2D {
    n: usize,
    nh: usize, // n/2 + 1 (rFFT half-spectrum length)
    hx: FftHandler<f64>,    // axis 0: full complex FFT (kx)
    hy: R2cFftHandler<f64>, // axis 1: real-to-complex FFT (ky)
    kx: Array2<f64>,
    ky: Array2<f64>,
    k2inv: Array2<f64>,   // 1/|k|^2, zero at the mean mode
    dealias: Array2<f64>, // 2/3-rule mask (1 keep, 0 drop)
}

impl Spectral2D {
    pub fn new(n: usize) -> Self {
        let nh = n / 2 + 1;
        let mut kx = Array2::<f64>::zeros((n, nh));
        let mut ky = Array2::<f64>::zeros((n, nh));
        let mut k2inv = Array2::<f64>::zeros((n, nh));
        let mut dealias = Array2::<f64>::zeros((n, nh));
        let cut = (n as f64) / 3.0;
        for i in 0..n {
            let kxi = if i <= n / 2 { i as f64 } else { i as f64 - n as f64 };
            for j in 0..nh {
                let kyj = j as f64;
                kx[[i, j]] = kxi;
                ky[[i, j]] = kyj;
                let k2 = kxi * kxi + kyj * kyj;
                k2inv[[i, j]] = if k2 > 0.0 { 1.0 / k2 } else { 0.0 };
                dealias[[i, j]] = if kxi.abs() <= cut && kyj <= cut { 1.0 } else { 0.0 };
            }
        }
        Self {
            n,
            nh,
            hx: FftHandler::new(n),
            hy: R2cFftHandler::new(n),
            kx,
            ky,
            k2inv,
            dealias,
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Real field -> spectral (rFFT along axis 1, complex FFT along axis 0).
    pub fn forward(&self, real: &Array2<f64>) -> Array2<C> {
        let mut tmp = Array2::<C>::zeros((self.n, self.nh));
        ndfft_r2c(real, &mut tmp, &self.hy, 1);
        let mut out = Array2::<C>::zeros((self.n, self.nh));
        ndfft(&tmp, &mut out, &self.hx, 0);
        out
    }

    /// Spectral -> real field (inverse of [`Self::forward`], normalised).
    pub fn inverse(&self, spec: &Array2<C>) -> Array2<f64> {
        let mut tmp = Array2::<C>::zeros((self.n, self.nh));
        ndifft(spec, &mut tmp, &self.hx, 0);
        let mut out = Array2::<f64>::zeros((self.n, self.n));
        ndifft_r2c(&tmp, &mut out, &self.hy, 1);
        out
    }

    /// Physical velocity `(u, v)` from spectral vorticity:
    /// `u = +d(psi)/dy`, `v = -d(psi)/dx`, `psi_hat = omega_hat / |k|^2`.
    fn velocity(&self, what: &Array2<C>) -> (Array2<f64>, Array2<f64>) {
        let mut uh = Array2::<C>::zeros((self.n, self.nh));
        let mut vh = Array2::<C>::zeros((self.n, self.nh));
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

    /// Dealiased advective nonlinear term `-(u . grad s)` in spectral space for
    /// a scalar `s` (vorticity or dye) given the physical velocity.
    fn advect(&self, shat: &Array2<C>, u: &Array2<f64>, v: &Array2<f64>) -> Array2<C> {
        let i = C::new(0.0, 1.0);
        let mut sxh = Array2::<C>::zeros((self.n, self.nh));
        let mut syh = Array2::<C>::zeros((self.n, self.nh));
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
        let mut adv = Array2::<f64>::zeros((self.n, self.n));
        Zip::from(&mut adv)
            .and(u)
            .and(v)
            .and(&sx)
            .and(&sy)
            .for_each(|a, &u, &v, &sx, &sy| *a = -(u * sx + v * sy));
        let mut nh = self.forward(&adv);
        Zip::from(&mut nh).and(&self.dealias).for_each(|c, &m| *c *= m);
        nh
    }
}

/// Real integrating-factor `exp(-coeff * |k|^2 * dt)` over the half-spectrum.
fn integ_factor(s: &Spectral2D, coeff: f64, dt: f64) -> Array2<f64> {
    let mut e = Array2::<f64>::zeros((s.n, s.nh));
    for i in 0..s.n {
        let kxi = if i <= s.n / 2 { i as f64 } else { i as f64 - s.n as f64 };
        for j in 0..s.nh {
            let k2 = kxi * kxi + (j as f64) * (j as f64);
            e[[i, j]] = (-coeff * k2 * dt).exp();
        }
    }
    e
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
    /// Steady prescribed downward jet velocity (physical, `n x n`, the `v`
    /// component) added to the advecting velocity. A divergence-free curtain
    /// `v = v(x)` in a vertical strip that cuts through the dye top-to-bottom.
    v_jet: Array2<f64>,
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
        let n = spec.n();
        let v_jet = Array2::<f64>::zeros((n, n));
        Self { spec, wh, gh, rh, dt, ew, ew2, ec, ec2, v_jet }
    }

    /// Set a steady downward jet: a smooth top-hat vertical strip of width
    /// `width` centred at `centre` (in `x`), full height, flowing down at
    /// `speed`. Added to the advecting velocity, so it drags vorticity and dye
    /// downward where it passes, cutting a swath through the swirl. `v = v(x)`
    /// is divergence-free, so it needs no pressure correction.
    pub fn set_jet(&mut self, centre: f64, width: f64, speed: f64) {
        let n = self.spec.n();
        let dx = std::f64::consts::TAU / n as f64;
        let edge = 0.12; // smoothing width of the strip walls
        let (lo, hi) = (centre - 0.5 * width, centre + 0.5 * width);
        for i in 0..n {
            let x = i as f64 * dx;
            // smooth top-hat in x: 1 inside the strip, 0 outside
            let th = 0.5 * (((x - lo) / edge).tanh() - ((x - hi) / edge).tanh());
            for j in 0..n {
                self.v_jet[[i, j]] = -speed * th; // downward (-y)
            }
        }
    }

    /// One IF-RK4 step for vorticity + the two dyes, sharing one velocity per
    /// stage. Standard integrating-factor RK4 (reduces to RK4 with zero
    /// viscosity, exact with zero nonlinear term).
    pub fn step(&mut self) {
        let dt = self.dt;
        let s = &self.spec;
        let vjet = &self.v_jet;
        // Nonlinear RHS from one velocity per stage. The steady jet is added to
        // the DYE advection only (not the vorticity), so the vortices keep their
        // true orbit-and-fray dynamics while the jet drags the colours downward
        // in its strip, cutting a vertical streak through the swirl.
        let rhs = |w: &Array2<C>, g: &Array2<C>, r: &Array2<C>| {
            let (u, v) = s.velocity(w);
            let mut vd = v.clone();
            Zip::from(&mut vd).and(vjet).for_each(|vd, &vj| *vd += vj);
            (s.advect(w, &u, &v), s.advect(g, &u, &vd), s.advect(r, &u, &vd))
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
        let fin = |e: &Array2<f64>, e2: &Array2<f64>, base: &Array2<C>,
                   k1: &Array2<C>, k2: &Array2<C>, k3: &Array2<C>, k4: &Array2<C>| {
            let term = scale(k1, e) + scale(&(k2 + k3), e2).mapv(|c| c * 2.0) + k4;
            let mut out = scale(base, e);
            Zip::from(&mut out).and(&term).for_each(|o, &t| *o += t * (dt / 6.0));
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
    let n = spec.n();
    let dx = std::f64::consts::TAU / n as f64;
    let (cx, cy) = (std::f64::consts::PI, std::f64::consts::PI);
    let amp = circ / (std::f64::consts::PI * core * core);
    let mut omega = Array2::<f64>::zeros((n, n));
    let mut gold = Array2::<f64>::zeros((n, n));
    let mut rust = Array2::<f64>::zeros((n, n));
    let centres = [(cx - 0.5 * sep, cy, 1.0_f64), (cx + 0.5 * sep, cy, 0.97_f64)];
    let rd = 1.1 * core;
    for i in 0..n {
        let x = i as f64 * dx;
        for j in 0..n {
            let y = j as f64 * dx;
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
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
        for i in 0..n {
            let x = i as f64 * dx;
            for j in 0..n {
                let y = j as f64 * dx;
                let mut sm = 0.0;
                for &(kxm, kym, ph) in &modes {
                    sm += (kxm * x + kym * y + ph).cos();
                }
                omega[[i, j]] += scale * sm;
            }
        }
    }
    (spec.forward(&omega), spec.forward(&gold), spec.forward(&rust))
}

/// Write the two dye densities to a PNG. On a black background gold + rust are
/// added (overlap blends to orange) with a gamma lift so thin frayed filaments
/// glow. On a white background the dyes composite as translucent inks (opacity
/// from density), an "ink in water on white paper" look; overlaps tend toward
/// orange and faint filaments read as light tints.
pub fn write_dye_png(gold: &Array2<f64>, rust: &Array2<f64>, path: &str, white_bg: bool) {
    let n = gold.shape()[0] as u32;
    let gcol = [1.0_f64, 0.78, 0.23];
    let rcol = [0.72_f64, 0.25, 0.05];
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(n, n);
    for (px, py, p) in img.enumerate_pixels_mut() {
        let i = (n - 1 - py) as usize; // flip y for an upright image
        let j = px as usize;
        let g = gold[[i, j]].max(0.0);
        let r = rust[[i, j]].max(0.0);
        let mut rgb = [0u8; 3];
        if white_bg {
            let tot = g + r;
            let a = tot.min(1.0).powf(0.6); // opacity; lifts faint filaments
            for k in 0..3 {
                let ink = if tot > 1e-6 { (gcol[k] * g + rcol[k] * r) / tot } else { 1.0 };
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
