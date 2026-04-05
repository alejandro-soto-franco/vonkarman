use num_complex::Complex;
use crate::float::Float;

/// Create a zero complex number.
#[inline]
pub fn complex_zero<F: Float>() -> Complex<F> {
    Complex { re: F::ZERO, im: F::ZERO }
}

/// Create a complex number from a real part (imaginary = 0).
#[inline]
pub fn complex_from_re<F: Float>(re: F) -> Complex<F> {
    Complex { re, im: F::ZERO }
}

/// Squared magnitude |z|^2 = re^2 + im^2.
#[inline]
pub fn norm_sq<F: Float>(z: Complex<F>) -> F {
    z.re * z.re + z.im * z.im
}

/// Multiply by the imaginary unit: i * z = -im + i*re.
#[inline]
pub fn i_times<F: Float>(z: Complex<F>) -> Complex<F> {
    Complex { re: -z.im, im: z.re }
}

/// Multiply two complex numbers: (a+bi)(c+di) = (ac-bd) + (ad+bc)i.
#[inline]
pub fn complex_mul<F: Float>(a: Complex<F>, b: Complex<F>) -> Complex<F> {
    Complex {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    }
}

/// Scale a complex number by a real factor.
#[inline]
pub fn complex_scale<F: Float>(z: Complex<F>, s: F) -> Complex<F> {
    Complex { re: z.re * s, im: z.im * s }
}

/// Complex exponential: exp(a + bi) = exp(a)(cos(b) + i sin(b)).
#[inline]
pub fn complex_exp<F: Float>(z: Complex<F>) -> Complex<F> {
    let ea = z.re.exp();
    let (sb, cb) = z.im.sin_cos();
    Complex { re: ea * cb, im: ea * sb }
}

/// Complex division: a / b.
#[inline]
pub fn complex_div<F: Float>(a: Complex<F>, b: Complex<F>) -> Complex<F> {
    let denom = b.re * b.re + b.im * b.im;
    Complex {
        re: (a.re * b.re + a.im * b.im) / denom,
        im: (a.im * b.re - a.re * b.im) / denom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_zero_one() {
        let z: Complex<f64> = complex_zero();
        assert_eq!(z.re, 0.0);
        assert_eq!(z.im, 0.0);
        let o: Complex<f64> = complex_from_re(1.0);
        assert_eq!(o.re, 1.0);
        assert_eq!(o.im, 0.0);
    }

    #[test]
    fn complex_norm_sq() {
        let z = Complex { re: 3.0_f64, im: 4.0 };
        assert!((norm_sq(z) - 25.0).abs() < 1e-14);
    }

    #[test]
    fn complex_i_times() {
        // i * (a + bi) = -b + ai
        let z = Complex { re: 3.0_f64, im: 4.0 };
        let iz = i_times(z);
        assert!((iz.re - (-4.0)).abs() < 1e-14);
        assert!((iz.im - 3.0).abs() < 1e-14);
    }
}
