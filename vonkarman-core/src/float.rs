use std::fmt::{Debug, Display};
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};

/// Multi-precision floating-point trait.
///
/// Implemented for f64 (Phase 1). DDReal, QDReal, MpfrFloat in Phase 2+.
/// Every arithmetic operation, transcendental, and conversion needed by
/// the solver and diagnostics pipeline.
pub trait Float:
    Copy
    + Send
    + Sync
    + Debug
    + Display
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const PI: Self;
    const EPSILON_VAL: Self;

    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn abs(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, e: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;

    fn recip(self) -> Self {
        Self::ONE / self
    }

    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
}

impl Float for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const PI: Self = std::f64::consts::PI;
    const EPSILON_VAL: Self = f64::EPSILON;

    #[inline] fn from_f64(x: f64) -> Self { x }
    #[inline] fn to_f64(self) -> f64 { self }
    #[inline] fn sqrt(self) -> Self { f64::sqrt(self) }
    #[inline] fn cbrt(self) -> Self { f64::cbrt(self) }
    #[inline] fn sin(self) -> Self { f64::sin(self) }
    #[inline] fn cos(self) -> Self { f64::cos(self) }
    #[inline] fn exp(self) -> Self { f64::exp(self) }
    #[inline] fn ln(self) -> Self { f64::ln(self) }
    #[inline] fn abs(self) -> Self { f64::abs(self) }
    #[inline] fn powi(self, n: i32) -> Self { f64::powi(self, n) }
    #[inline] fn powf(self, e: Self) -> Self { f64::powf(self, e) }
    #[inline] fn max(self, other: Self) -> Self { f64::max(self, other) }
    #[inline] fn min(self, other: Self) -> Self { f64::min(self, other) }
    #[inline] fn is_finite(self) -> bool { f64::is_finite(self) }
    #[inline] fn is_nan(self) -> bool { f64::is_nan(self) }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        f64::sin_cos(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_constants() {
        assert_eq!(f64::ZERO, 0.0);
        assert_eq!(f64::ONE, 1.0);
        assert!((f64::PI - std::f64::consts::PI).abs() < 1e-15);
        assert_eq!(f64::EPSILON_VAL, f64::EPSILON);
    }

    #[test]
    fn f64_from_to() {
        let x = <f64 as Float>::from_f64(3.14);
        assert_eq!(x.to_f64(), 3.14);
    }

    #[test]
    fn f64_transcendentals() {
        let x = <f64 as Float>::from_f64(1.0);
        assert!((Float::sin(x) - 1.0_f64.sin()).abs() < 1e-15);
        assert!((Float::cos(x) - 1.0_f64.cos()).abs() < 1e-15);
        assert!((Float::exp(x) - 1.0_f64.exp()).abs() < 1e-15);
        assert!((Float::ln(x)).abs() < 1e-15);
        assert!((Float::sqrt(x) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn f64_powi() {
        let x = <f64 as Float>::from_f64(2.0);
        assert!((Float::powi(x, 10) - 1024.0).abs() < 1e-10);
    }

    #[test]
    fn f64_powf() {
        let x = <f64 as Float>::from_f64(2.0);
        let e = <f64 as Float>::from_f64(0.5);
        assert!((Float::powf(x, e) - std::f64::consts::SQRT_2).abs() < 1e-14);
    }
}
