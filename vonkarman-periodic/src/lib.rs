// Component iteration (for c in 0..3) is idiomatic in CFD vector field code.
#![allow(clippy::needless_range_loop)]

pub mod etd;
pub mod ic;
pub mod nonlinear;
#[cfg(feature = "cuda")]
pub mod resident;
pub mod rk4;
pub mod solver;
pub mod state;

pub use ic::IcType;
pub use solver::{Periodic3D, frame_diagnostics_uhat};
pub use state::{SpectralState, StateDims};

#[cfg(feature = "cuda")]
pub use resident::ResidentSolver;
