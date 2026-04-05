use serde::Serialize;
use vonkarman_core::domain::Domain;

/// Tier 1 scalar diagnostics, computed every timestep.
#[derive(Debug, Clone, Serialize)]
pub struct ScalarDiagnostics {
    pub time: f64,
    pub step: u64,
    pub dt: f64,
    pub energy: f64,
    pub enstrophy: f64,
    pub helicity: f64,
    pub superhelicity: f64,
    pub max_vorticity: f64,
    pub energy_dissipation_rate: f64,
    pub helicity_dissipation_rate: f64,
    pub cfl_number: f64,
}

impl ScalarDiagnostics {
    /// Extract diagnostics from any Domain implementation.
    pub fn from_domain(domain: &dyn Domain<f64>) -> Self {
        let nu = domain.params().nu;
        let enstrophy = domain.enstrophy();
        let superhelicity = domain.superhelicity();
        let (edr, hdr) = Self::compute_rates(nu, enstrophy, superhelicity);

        let dt = domain.dt();
        let dx = domain.grid().dx();
        let max_vort = domain.max_vorticity();
        let cfl = max_vort * dt / dx;

        Self {
            time: domain.time(),
            step: domain.step_count(),
            dt,
            energy: domain.energy(),
            enstrophy,
            helicity: domain.helicity(),
            superhelicity,
            max_vorticity: max_vort,
            energy_dissipation_rate: edr,
            helicity_dissipation_rate: hdr,
            cfl_number: cfl,
        }
    }

    /// Compute dissipation rates from physics.
    /// Returns (dE/dt, dH/dt).
    /// dE/dt = -2 * nu * Omega (enstrophy)
    /// dH/dt = -2 * nu * H_2 (superhelicity)
    pub fn compute_rates(nu: f64, enstrophy: f64, superhelicity: f64) -> (f64, f64) {
        (-2.0 * nu * enstrophy, -2.0 * nu * superhelicity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_from_zero_energy() {
        let d = ScalarDiagnostics {
            time: 0.0,
            step: 0,
            dt: 0.01,
            energy: 0.0,
            enstrophy: 0.0,
            helicity: 0.0,
            superhelicity: 0.0,
            max_vorticity: 0.0,
            energy_dissipation_rate: 0.0,
            helicity_dissipation_rate: 0.0,
            cfl_number: 0.0,
        };
        assert_eq!(d.energy, 0.0);
    }

    #[test]
    fn energy_dissipation_identity() {
        let nu = 0.01;
        let enstrophy = 10.0;
        let expected_rate = -2.0 * nu * enstrophy;
        let (rate, _) = ScalarDiagnostics::compute_rates(nu, enstrophy, 5.0);
        assert!((rate - expected_rate).abs() < 1e-14);
    }
}
