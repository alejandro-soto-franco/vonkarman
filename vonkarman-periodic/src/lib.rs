// Component iteration (for c in 0..3) is idiomatic in CFD vector field code.
#![allow(clippy::needless_range_loop)]

pub mod etd;
pub mod ic;
pub mod nonlinear;
pub mod rk4;
pub mod solver;

pub use ic::IcType;
pub use solver::Periodic3D;
