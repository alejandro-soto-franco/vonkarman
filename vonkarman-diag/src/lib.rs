pub mod audit;
pub mod scalar;
pub mod spectrum;

pub use audit::{ConservationAudit, Violation};
pub use scalar::ScalarDiagnostics;
pub use spectrum::{compensated_spectrum, dissipation_spectrum, energy_spectrum};
