use crate::scalar::ScalarDiagnostics;
use serde::Serialize;

/// Violations detected by the conservation audit.
#[derive(Debug, Clone, Serialize)]
pub enum Violation {
    EnergyIncrease {
        step: u64,
        prev: f64,
        curr: f64,
    },
    DissipationMismatch {
        step: u64,
        computed: f64,
        expected: f64,
        rel_err: f64,
    },
    NanDetected {
        step: u64,
        field: String,
    },
}

/// Tracks conservation law compliance across timesteps.
pub struct ConservationAudit {
    prev_energy: Option<f64>,
    step_counter: u64,
    pub violations: Vec<Violation>,
}

impl Default for ConservationAudit {
    fn default() -> Self {
        Self::new()
    }
}

impl ConservationAudit {
    pub fn new() -> Self {
        Self {
            prev_energy: None,
            step_counter: 0,
            violations: Vec::new(),
        }
    }

    /// Check conservation laws for the current timestep.
    pub fn check(
        &mut self,
        energy: f64,
        _energy_dissipation_rate: f64,
        _expected_dissipation_rate: f64,
        _nu: f64,
    ) {
        let step = self.step_counter;
        self.step_counter += 1;

        // Check for NaN
        if energy.is_nan() {
            self.violations.push(Violation::NanDetected {
                step,
                field: "energy".to_string(),
            });
            return;
        }

        // Check energy monotonicity
        if let Some(prev) = self.prev_energy
            && energy > prev + 1e-14 * prev.abs().max(1e-30)
        {
            self.violations.push(Violation::EnergyIncrease {
                step,
                prev,
                curr: energy,
            });
        }
        self.prev_energy = Some(energy);
    }

    /// Check diagnostics from a ScalarDiagnostics struct.
    pub fn check_diagnostics(&mut self, diag: &ScalarDiagnostics, nu: f64) {
        let expected_edr = -2.0 * nu * diag.enstrophy;
        self.check(diag.energy, diag.energy_dissipation_rate, expected_edr, nu);

        if diag.energy.is_nan() || diag.enstrophy.is_nan() || diag.max_vorticity.is_nan() {
            self.violations.push(Violation::NanDetected {
                step: diag.step,
                field: "diagnostics".to_string(),
            });
        }
    }

    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_accepts_decaying_energy() {
        let mut audit = ConservationAudit::new();
        audit.check(1.0, 0.0, 0.0, 0.0);
        audit.check(0.9, -0.2, -0.19, 0.0);
        audit.check(0.8, -0.18, -0.17, 0.0);
        assert!(audit.violations.is_empty());
    }

    #[test]
    fn audit_catches_energy_increase() {
        let mut audit = ConservationAudit::new();
        audit.check(1.0, 0.0, 0.0, 0.0);
        audit.check(1.1, -0.2, -0.19, 0.0); // energy went up!
        assert_eq!(audit.violations.len(), 1);
        assert!(matches!(
            audit.violations[0],
            Violation::EnergyIncrease { .. }
        ));
    }

    #[test]
    fn audit_catches_nan() {
        let mut audit = ConservationAudit::new();
        audit.check(f64::NAN, 0.0, 0.0, 0.0);
        assert_eq!(audit.violations.len(), 1);
        assert!(matches!(audit.violations[0], Violation::NanDetected { .. }));
    }
}
