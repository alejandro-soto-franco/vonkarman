//! Pointwise spectral operator bodies.
//!
//! Each operator is a pure function over flat slices, generic over `F: Float`,
//! with no `ndarray` dependency. This is deliberate: the same body is intended
//! to become a GPU kernel, where the CPU result is the differential oracle.

pub mod cross;

pub use cross::cross_product_inplace;
