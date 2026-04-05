pub mod scalar;
pub mod audit;
pub mod spectrum;

pub use scalar::ScalarDiagnostics;
pub use audit::{ConservationAudit, Violation};
pub use spectrum::{energy_spectrum, compensated_spectrum, dissipation_spectrum};
