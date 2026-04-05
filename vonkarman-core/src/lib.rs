pub mod float;
pub mod complex;
pub mod field;
pub mod spectral_ops;
pub mod kahan;
pub mod domain;

// Convenient re-exports
pub use float::Float;
pub use field::{GridSpec, ScalarField, VectorField};
pub use domain::{Domain, DomainType, PhysicsParams, Snapshot};
pub use spectral_ops::SpectralOps;
