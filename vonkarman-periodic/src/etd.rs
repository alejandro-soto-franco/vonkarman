use std::f64::consts::PI;

/// Number of contour integration points (Kassam-Trefethen trick).
/// 32 is standard; sufficient for f64.
const M: usize = 32;

/// phi_1(z) = (e^z - 1) / z, computed via contour integral to avoid
/// catastrophic cancellation when z is near zero.
///
/// Uses the Kassam-Trefethen (2005) trick: evaluate on a circle in the
/// complex plane and take the real part of the mean.
///
/// Reference: Kassam & Trefethen, "Fourth-order time-stepping for stiff PDEs",
/// SIAM J. Sci. Comput. 26(4), 2005.
pub fn phi1(z: f64) -> f64 {
    if z.abs() < 1e-300 {
        return 1.0;
    }
    let r = 1.0;
    let mut sum_re = 0.0;
    for j in 0..M {
        let theta = 2.0 * PI * (j as f64 + 0.5) / M as f64;
        let w_re = z + r * theta.cos();
        let w_im = r * theta.sin();
        let ew_re = w_re.exp() * w_im.cos();
        let ew_im = w_re.exp() * w_im.sin();
        let num_re = ew_re - 1.0;
        let num_im = ew_im;
        let denom = w_re * w_re + w_im * w_im;
        let result_re = (num_re * w_re + num_im * w_im) / denom;
        sum_re += result_re;
    }
    sum_re / M as f64
}

/// phi_2(z) = (e^z - 1 - z) / z^2, computed via contour integral.
pub fn phi2(z: f64) -> f64 {
    if z.abs() < 1e-300 {
        return 0.5;
    }
    let r = 1.0;
    let mut sum_re = 0.0;
    for j in 0..M {
        let theta = 2.0 * PI * (j as f64 + 0.5) / M as f64;
        let w_re = z + r * theta.cos();
        let w_im = r * theta.sin();
        let ew_re = w_re.exp() * w_im.cos();
        let ew_im = w_re.exp() * w_im.sin();
        let num_re = ew_re - 1.0 - w_re;
        let num_im = ew_im - w_im;
        let w2_re = w_re * w_re - w_im * w_im;
        let w2_im = 2.0 * w_re * w_im;
        let denom = w2_re * w2_re + w2_im * w2_im;
        let result_re = (num_re * w2_re + num_im * w2_im) / denom;
        sum_re += result_re;
    }
    sum_re / M as f64
}

/// phi_3(z) = (e^z - 1 - z - z^2/2) / z^3, computed via contour integral.
fn phi3(z: f64) -> f64 {
    if z.abs() < 1e-300 {
        return 1.0 / 6.0;
    }
    let r = 1.0;
    let mut sum_re = 0.0;
    for j in 0..M {
        let theta = 2.0 * PI * (j as f64 + 0.5) / M as f64;
        let w_re = z + r * theta.cos();
        let w_im = r * theta.sin();
        let ew_re = w_re.exp() * w_im.cos();
        let ew_im = w_re.exp() * w_im.sin();
        let w2_re = w_re * w_re - w_im * w_im;
        let w2_im = 2.0 * w_re * w_im;
        let num_re = ew_re - 1.0 - w_re - 0.5 * w2_re;
        let num_im = ew_im - w_im - 0.5 * w2_im;
        let w3_re = w_re * w2_re - w_im * w2_im;
        let w3_im = w_re * w2_im + w_im * w2_re;
        let denom = w3_re * w3_re + w3_im * w3_im;
        let result_re = (num_re * w3_re + num_im * w3_im) / denom;
        sum_re += result_re;
    }
    sum_re / M as f64
}

/// Precomputed ETD-RK4 coefficients for a single wavenumber.
///
/// Given lambda = -nu * |k|^2, these are the scalar factors needed
/// for the 4-stage ETD-RK4 scheme (Cox-Matthews / Kassam-Trefethen).
#[derive(Debug, Clone, Copy)]
pub struct EtdCoeffs {
    /// e^{lambda * dt}
    pub exp_full: f64,
    /// e^{lambda * dt/2}
    pub exp_half: f64,
    /// (1/2) phi_1(lambda * dt/2) for stage 2.
    ///
    /// The half is Cox-Matthews': the stage increment is `(h/2) phi_1(Lh/2) N`,
    /// and the solver multiplies by the full `h`, so the half lives here.
    pub a21: f64,
    /// (1/2) phi_1(lambda * dt/2) for stage 3. Equal to `a21`.
    pub a31: f64,
    /// (1/2) phi_1(lambda * dt/2) for stage 4.
    ///
    /// The argument is `lambda * dt/2`, not `lambda * dt`: stage 4 advances
    /// the stage-2 state over a further half step.
    pub a41: f64,
    /// Final combination coefficients.
    /// u_new = exp_full * u + dt * (b1*N1 + b23*(N2+N3) + b4*N4)
    pub b1: f64,
    pub b23: f64,
    pub b4: f64,
}

impl EtdCoeffs {
    /// Compute ETD-RK4 coefficients for a given lambda*dt product.
    ///
    /// lambda = -nu * |k|^2 (always non-positive for viscous diffusion).
    pub fn new(lambda_dt: f64) -> Self {
        let z = lambda_dt;
        let zh = z / 2.0;

        let exp_full = z.exp();
        let exp_half = zh.exp();

        let phi1_h = phi1(zh);
        let phi1_f = phi1(z);
        let phi2_f = phi2(z);
        let phi3_f = phi3(z);

        Self {
            exp_full,
            exp_half,
            a21: 0.5 * phi1_h,
            a31: 0.5 * phi1_h,
            a41: 0.5 * phi1_h,
            b1: phi1_f - 3.0 * phi2_f + 4.0 * phi3_f,
            b23: 2.0 * phi2_f - 4.0 * phi3_f,
            b4: -phi2_f + 4.0 * phi3_f,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At `lambda = 0` the exponential integrator must BE classical RK4.
    ///
    /// With no linear operator the integrating factor is the identity and every
    /// `phi_k` takes its limiting value, so the tableau has to reduce to the
    /// classical one: stage coefficients `1/2, 1/2, 1/2` and weights
    /// `1/6, 1/3, 1/3, 1/6`. This is the check the stage-coefficient error
    /// failed: it carried `a41 = phi_1(lambda h)`, which is `1` at
    /// `lambda = 0`, twice the classical value, so the scheme was not the
    /// order-4 method it claimed. Cox-Matthews takes stage 4 from the stage-2
    /// state over a further half step, which is where the half belongs.
    #[test]
    fn reduces_to_classical_rk4_at_zero_lambda() {
        let c = EtdCoeffs::new(0.0);
        let tol = 1e-12;
        for (name, got, want) in [
            ("exp_full", c.exp_full, 1.0),
            ("exp_half", c.exp_half, 1.0),
            ("a21", c.a21, 0.5),
            ("a31", c.a31, 0.5),
            ("a41", c.a41, 0.5),
            ("b1", c.b1, 1.0 / 6.0),
            ("b23", c.b23, 1.0 / 3.0),
            ("b4", c.b4, 1.0 / 6.0),
        ] {
            assert!(
                (got - want).abs() < tol,
                "{name} = {got}, classical RK4 requires {want}"
            );
        }
        // The weights sum to one, so a constant right-hand side advances exactly.
        let sum = c.b1 + 2.0 * c.b23 + c.b4;
        assert!((sum - 1.0).abs() < tol, "weights sum to {sum}, expected 1");
    }

    #[test]
    fn phi1_limit_at_zero() {
        let p = phi1(0.0);
        assert!((p - 1.0).abs() < 1e-12, "phi1(0) = {p}, expected 1.0");
    }

    #[test]
    fn phi2_limit_at_zero() {
        let p = phi2(0.0);
        assert!((p - 0.5).abs() < 1e-12, "phi2(0) = {p}, expected 0.5");
    }

    #[test]
    fn phi1_known_value() {
        let p = phi1(1.0);
        let expected = std::f64::consts::E - 1.0;
        assert!(
            (p - expected).abs() < 1e-10,
            "phi1(1) = {p}, expected {expected}"
        );
    }

    #[test]
    fn phi2_known_value() {
        let p = phi2(1.0);
        let expected = std::f64::consts::E - 2.0;
        assert!(
            (p - expected).abs() < 1e-10,
            "phi2(1) = {p}, expected {expected}"
        );
    }

    #[test]
    fn phi1_large_negative() {
        let p = phi1(-100.0);
        let expected = ((-100.0_f64).exp() - 1.0) / (-100.0);
        assert!(
            (p - expected).abs() / expected.abs() < 1e-8,
            "phi1(-100) = {p}, expected {expected}"
        );
    }

    #[test]
    fn phi_functions_near_zero_stable() {
        for &z in &[1e-10, 1e-8, 1e-6, 1e-4, 1e-2] {
            let p1 = phi1(z);
            let p2 = phi2(z);
            assert!(
                (p1 - 1.0).abs() < z.abs() * 2.0,
                "phi1({z}) = {p1} not close to 1"
            );
            assert!(
                (p2 - 0.5).abs() < z.abs() * 2.0,
                "phi2({z}) = {p2} not close to 0.5"
            );
        }
    }

    #[test]
    fn etd_coeffs_zero_lambda() {
        // For z=0: exp_full=1, exp_half=1, phi1=1, phi2=0.5, phi3=1/6
        // b1 = 1 - 1.5 + 4/6 = 1/6, b23 = 1 - 4/6 = 2/6 = 1/3, b4 = -0.5 + 4/6 = 1/6
        let c = EtdCoeffs::new(0.0);
        assert!((c.exp_full - 1.0).abs() < 1e-12);
        assert!((c.exp_half - 1.0).abs() < 1e-12);
        // b1 + 2*b23 + b4 = 1 (consistency: sum of weights = phi1)
        let sum = c.b1 + 2.0 * c.b23 + c.b4;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "weight sum = {sum}, expected 1.0"
        );
    }
}
