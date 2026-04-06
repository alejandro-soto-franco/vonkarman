use crate::field::{GridSpec, VectorField};
use crate::float::Float;
use ndarray::Array3;
use num_complex::Complex;
use serde::{Deserialize, Serialize};

/// Which domain topology the solver uses.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DomainType {
    Periodic3D,
    Axisymmetric,
}

/// Physical parameters for the simulation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PhysicsParams {
    /// Kinematic viscosity.
    pub nu: f64,
    /// Reynolds number (redundant with nu, but convenient).
    pub re: f64,
    /// Domain type.
    pub domain: DomainType,
}

/// Full-state snapshot for I/O and diagnostics.
pub struct Snapshot<F: Float> {
    pub time: F,
    pub step: u64,
    pub dt: F,
    pub velocity: VectorField<F>,
    pub vorticity: VectorField<F>,
    /// Spectral velocity coefficients (for exact restart).
    pub u_hat: [Array3<Complex<F>>; 3],
    pub grid: GridSpec,
    pub params: PhysicsParams,
}

/// Solver interface. Implemented by Periodic3D and Axisymmetric.
pub trait Domain<F: Float> {
    /// Advance one timestep.
    fn step(&mut self);

    /// Current simulation time.
    fn time(&self) -> F;

    /// Current step count.
    fn step_count(&self) -> u64;

    /// Current timestep size.
    fn dt(&self) -> F;

    /// Total kinetic energy: (1/2) * integral |u|^2 dx.
    fn energy(&self) -> F;

    /// Enstrophy: integral |omega|^2 dx.
    fn enstrophy(&self) -> F;

    /// Helicity: integral u . omega dx.
    fn helicity(&self) -> F;

    /// Superhelicity: integral omega . curl(omega) dx.
    fn superhelicity(&self) -> F;

    /// Max vorticity magnitude: ||omega||_inf.
    fn max_vorticity(&self) -> F;

    /// CFL-based adaptive timestep.
    fn cfl_dt(&self) -> F;

    /// Borrow the spectral velocity state.
    fn u_hat(&self) -> &[Array3<Complex<F>>; 3];

    /// Grid specification.
    fn grid(&self) -> &GridSpec;

    /// Physics parameters.
    fn params(&self) -> &PhysicsParams;

    /// Full-state snapshot for I/O.
    fn snapshot(&self) -> Snapshot<F>;
}
