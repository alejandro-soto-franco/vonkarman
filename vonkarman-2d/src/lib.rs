//! vonkarman-2d: 2D pseudo-spectral incompressible Navier-Stokes on the torus
//! T^2, vorticity-streamfunction form, with passive scalar (dye) transport.
//!
//! Sibling of the 3D `vonkarman` solver, sharing its pseudo-spectral
//! conventions: state lives in spectral space, the nonlinear term is formed in
//! physical space and dealiased (2/3 rule), and viscosity is integrated exactly
//! via an integrating factor (here IF-RK4). Because 2D is cheap, it runs at
//! 1024^2+ where the fine filaments are genuinely RESOLVED, so a passive dye
//! advecting through it looks like the "incredible" graphics-fluid visuals
//! without any noise-upres or vorticity-confinement cheats.
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

    /// Nonlinear right-hand sides for vorticity and dye sharing one velocity.
    fn rhs(&self, wh: &Array2<C>, ch: &Array2<C>) -> (Array2<C>, Array2<C>) {
        let (u, v) = self.velocity(wh);
        (self.advect(wh, &u, &v), self.advect(ch, &u, &v))
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

/// A running 2D simulation: vorticity + dye state and the fixed-dt IF-RK4 maps.
pub struct Sim {
    pub spec: Spectral2D,
    pub wh: Array2<C>, // vorticity (spectral)
    pub ch: Array2<C>, // dye (spectral)
    dt: f64,
    ew: Array2<f64>,
    ew2: Array2<f64>,
    ec: Array2<f64>,
    ec2: Array2<f64>,
}

impl Sim {
    pub fn new(spec: Spectral2D, wh: Array2<C>, ch: Array2<C>, dt: f64, nu: f64, kappa: f64) -> Self {
        let ew = integ_factor(&spec, nu, dt);
        let ew2 = integ_factor(&spec, nu, dt * 0.5);
        let ec = integ_factor(&spec, kappa, dt);
        let ec2 = integ_factor(&spec, kappa, dt * 0.5);
        Self { spec, wh, ch, dt, ew, ew2, ec, ec2 }
    }

    /// One IF-RK4 step. Standard integrating-factor RK4 (reduces to RK4 when
    /// viscosity is zero, exact when the nonlinear term is zero), stepping
    /// vorticity and dye together through one shared velocity per stage.
    pub fn step(&mut self) {
        let dt = self.dt;
        let s = &self.spec;
        let comb = |e: &Array2<f64>, base: &Array2<C>, f: f64, k: &Array2<C>| -> Array2<C> {
            // e .* (base + f*k)
            scale(&(base + &k.mapv(|c| c * f)), e)
        };

        let (k1w, k1c) = s.rhs(&self.wh, &self.ch);
        let a2w = comb(&self.ew2, &self.wh, dt * 0.5, &k1w);
        let a2c = comb(&self.ec2, &self.ch, dt * 0.5, &k1c);
        let (k2w, k2c) = s.rhs(&a2w, &a2c);
        // e2 .* base + (dt/2) k
        let a3w = &scale(&self.wh, &self.ew2) + &k2w.mapv(|c| c * (dt * 0.5));
        let a3c = &scale(&self.ch, &self.ec2) + &k2c.mapv(|c| c * (dt * 0.5));
        let (k3w, k3c) = s.rhs(&a3w, &a3c);
        // e .* base + dt (e2 .* k)
        let a4w = &scale(&self.wh, &self.ew) + &scale(&k3w, &self.ew2).mapv(|c| c * dt);
        let a4c = &scale(&self.ch, &self.ec) + &scale(&k3c, &self.ec2).mapv(|c| c * dt);
        let (k4w, k4c) = s.rhs(&a4w, &a4c);

        let upd = |e: &Array2<f64>, e2: &Array2<f64>, base: &Array2<C>,
                   k1: &Array2<C>, k2: &Array2<C>, k3: &Array2<C>, k4: &Array2<C>| -> Array2<C> {
            let mut out = scale(base, e);
            let term = scale(k1, e)
                + scale(&(k2 + k3), e2).mapv(|c| c * 2.0)
                + k4;
            Zip::from(&mut out)
                .and(&term)
                .for_each(|o, &t| *o += t * (dt / 6.0));
            out
        };
        self.wh = upd(&self.ew, &self.ew2, &self.wh, &k1w, &k2w, &k3w, &k4w);
        self.ch = upd(&self.ec, &self.ec2, &self.ch, &k1c, &k2c, &k3c, &k4c);
    }

    /// Physical dye field (real space), for rendering.
    pub fn dye(&self) -> Array2<f64> {
        self.spec.inverse(&self.ch)
    }
}

/// Two co-rotating Lamb-Oseen vortices, with gold dye in one and rust-red dye
/// in the other. The same-sign vortices orbit and merge, winding the two dyes
/// into a double spiral (two "tornados" twisting together and connecting).
///
/// Returns spectral `(omega_hat, dye_hat)`. `dye` is signed: `+1` gold, `-1`
/// rust. Profiles are smooth (super-Gaussian) to avoid Gibbs ringing.
pub fn co_rotating_vortices(
    spec: &Spectral2D,
    sep: f64,
    core: f64,
    circ: f64,
) -> (Array2<C>, Array2<C>) {
    let n = spec.n();
    let dx = std::f64::consts::TAU / n as f64;
    let cx = std::f64::consts::PI;
    let cy = std::f64::consts::PI;
    let amp = circ / (std::f64::consts::PI * core * core);
    let mut omega = Array2::<f64>::zeros((n, n));
    let mut dye = Array2::<f64>::zeros((n, n));
    let centres = [(cx - 0.5 * sep, cy), (cx + 0.5 * sep, cy)];
    let rd = 1.1 * core;
    for i in 0..n {
        let x = i as f64 * dx;
        for j in 0..n {
            let y = j as f64 * dx;
            let mut w = 0.0;
            for (vc, &(px, py)) in centres.iter().enumerate() {
                let r2 = (x - px) * (x - px) + (y - py) * (y - py);
                w += amp * (-r2 / (core * core)).exp();
                // smooth dye disc (super-Gaussian), gold for vortex 0, rust for 1
                let d = (-(r2 / (rd * rd)).powi(2)).exp();
                dye[[i, j]] += if vc == 0 { d } else { -d };
            }
            omega[[i, j]] = w;
        }
    }
    (spec.forward(&omega), spec.forward(&dye))
}

/// Write the signed dye field to a PNG: gold for positive, rust-red for
/// negative, on black, with a gamma lift for punch. Values are clamped to the
/// seeded `[-1, 1]` range.
pub fn write_dye_png(dye: &Array2<f64>, path: &str) {
    let n = dye.shape()[0] as u32;
    let gold = [1.0_f64, 0.78, 0.23];
    let rust = [0.72_f64, 0.25, 0.05];
    let gamma = 0.80;
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(n, n);
    for (px, py, p) in img.enumerate_pixels_mut() {
        // image (x=col, y=row) -> field (i=row=y_img, j=col=x_img); flip y for
        // a conventional upright orientation.
        let i = (n - 1 - py) as usize;
        let j = px as usize;
        let c = dye[[i, j]];
        let g = c.clamp(0.0, 1.0);
        let r = (-c).clamp(0.0, 1.0);
        let mut rgb = [0u8; 3];
        for k in 0..3 {
            let v = (gold[k] * g + rust[k] * r).clamp(0.0, 1.0).powf(gamma);
            rgb[k] = (v * 255.0).round() as u8;
        }
        *p = Rgb(rgb);
    }
    img.save(path).expect("write_dye_png: save failed");
}
