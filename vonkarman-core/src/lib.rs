pub mod complex;
pub mod domain;
pub mod field;
pub mod float;
pub mod kahan;
pub mod spectral_ops;

// Convenient re-exports
pub use domain::{Domain, DomainType, PhysicsParams, Snapshot};
pub use field::{GridSpec, ScalarField, VectorField};
pub use float::Float;
pub use spectral_ops::SpectralOps;
