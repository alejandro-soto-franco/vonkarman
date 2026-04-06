use crate::float::Float;
use ndarray::Array3;

/// Kahan-compensated summation over a 3D array.
///
/// Recovers ~7 digits lost by naive summation at N^3 ~ 5.7e7 elements.
/// Used by every global reduction (energy, enstrophy, helicity).
pub fn kahan_sum<F: Float>(arr: &Array3<F>) -> F {
    let mut sum = F::ZERO;
    let mut comp = F::ZERO; // running compensation
    for &val in arr.iter() {
        let y = val - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Kahan-compensated dot product of two 3D arrays (element-wise multiply, then sum).
pub fn kahan_dot<F: Float>(a: &Array3<F>, b: &Array3<F>) -> F {
    debug_assert_eq!(a.shape(), b.shape());
    let mut sum = F::ZERO;
    let mut comp = F::ZERO;
    for (av, bv) in a.iter().zip(b.iter()) {
        let prod = *av * *bv;
        let y = prod - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn kahan_sum_exact_for_small() {
        let a = Array3::from_elem((4, 4, 4), 1.0_f64);
        let s = kahan_sum(&a);
        assert!((s - 64.0).abs() < 1e-14);
    }

    #[test]
    fn kahan_sum_beats_naive_for_large() {
        let n = 100_000;
        let eps = 1e-15_f64;
        let mut a = Array3::from_elem((1, 1, n), eps);
        a[[0, 0, 0]] = 1.0;
        let s = kahan_sum(&a);
        let expected = 1.0 + (n as f64 - 1.0) * eps;
        assert!(
            (s - expected).abs() < 1e-14,
            "kahan_sum={s}, expected={expected}"
        );
    }

    #[test]
    fn kahan_dot_basic() {
        let a = Array3::from_elem((4, 4, 4), 2.0_f64);
        let b = Array3::from_elem((4, 4, 4), 3.0_f64);
        let d = kahan_dot(&a, &b);
        assert!((d - 64.0 * 6.0).abs() < 1e-12);
    }
}
