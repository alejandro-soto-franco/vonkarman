use vonkarman_core::field::{GridSpec, VectorField};
use vonkarman_core::float::Float;

/// Colliding vortex rings initial condition (implemented in Task 2).
///
/// Two coaxial vortex rings with opposite circulation propagate toward each
/// other along `axis` (0=x, 1=y, 2=z). They collide and reconnect.
///
/// Parameters:
/// - `ring_radius`: torus major radius R
/// - `core_radius`: Gaussian core radius sigma
/// - `circulation`: ring circulation Gamma
/// - `separation`: initial centre-to-centre separation along the axis
/// - `axis`: propagation axis (0=x, 1=y, 2=z)
#[allow(dead_code)]
pub fn colliding_vortex_rings<F: Float>(
    _grid: &GridSpec,
    _ring_radius: f64,
    _core_radius: f64,
    _circulation: f64,
    _separation: f64,
    _axis: usize,
) -> VectorField<F> {
    unimplemented!("colliding_vortex_rings: implemented in Task 2")
}
