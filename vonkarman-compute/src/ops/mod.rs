//! Pointwise spectral operator bodies.
//!
//! Each operator is a pure function over flat slices, generic over `F: Float`,
//! with no `ndarray` dependency. This is deliberate: the same body is intended
//! to become a GPU kernel, where the CPU result is the differential oracle.

pub mod cross;
pub mod curl;
pub mod etd;
pub mod leray;
pub mod pad;
pub mod scale;

pub use cross::cross_product_inplace;
pub use curl::curl_inplace;
pub use etd::{etd_final_inplace, etd_stage_axpy_inplace, etd_stage4_inplace};
pub use leray::leray_inplace;
pub use pad::{truncate_inplace, zero_pad_inplace};
pub use scale::scale_cplx_inplace;
