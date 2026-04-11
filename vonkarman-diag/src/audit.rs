use serde::Serialize;

/// Configuration for conservation audit tolerances.
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Tolerance for energy budget residual |dE/dt + 2*nu*Omega| / |2*nu*Omega|.
    pub energy_budget_tol: f64,
    /// Tolerance for divergence-free check: max |k . u_hat|.
    pub divergence_tol: f64,
    /// Tolerance for Parseval identity: |E_spectral - E_physical| / E_spectral.
    pub parseval_tol: f64,
    /// If true, abort on first violation.
    pub halt_on_violation: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            energy_budget_tol: 1e-4,
            divergence_tol: 1e-10,
            parseval_tol: 1e-12,
            halt_on_violation: false,
        }
    }
}

/// Violations detected by the conservation audit.
#[derive(Debug, Clone, Serialize)]
pub enum Violation {
    EnergyIncrease {
        step: u64,
        prev: f64,
        curr: f64,
    },
    NanDetected {
        step: u64,
        field: String,
    },
    DivergenceViolation {
        step: u64,
        max_div: f64,
    },
    ParsevalViolation {
        step: u64,
        residual: f64,
    },
    EnergyBudgetViolation {
        step: u64,
        residual: f64,
        de_dt: f64,
        expected: f64,
    },
    DissipationMismatch {
        step: u64,
        computed: f64,
        expected: f64,
        rel_err: f64,
    },
}

/// Tracks conservation law compliance across timesteps.
pub struct ConservationAudit {
    config: AuditConfig,
    prev_energy: Option<f64>,
    prev_dt: Option<f64>,
    step_counter: u64,
    pub violations: Vec<Violation>,
}

impl Default for ConservationAudit {
    fn default() -> Self {
        Self::new()
    }
}

impl ConservationAudit {
    /// Create an audit with default tolerances.
    pub fn new() -> Self {
        Self::with_config(AuditConfig::default())
    }

    /// Create an audit with custom tolerances.
    pub fn with_config(config: AuditConfig) -> Self {
        Self {
            config,
            prev_energy: None,
            prev_dt: None,
            step_counter: 0,
            violations: Vec::new(),
        }
    }

    /// Full conservation check for one timestep.
    ///
    /// Arguments:
    /// - `energy`: current kinetic energy
    /// - `enstrophy`: current enstrophy
    /// - `nu`: kinematic viscosity
    /// - `dt`: current timestep size
    /// - `max_divergence`: max |k . u_hat(k)| (caller computes)
    /// - `parseval_residual`: |E_spectral - E_physical| / E_spectral (caller computes)
    pub fn check_full(
        &mut self,
        energy: f64,
        enstrophy: f64,
        nu: f64,
        dt: f64,
        max_divergence: f64,
        parseval_residual: f64,
    ) {
        let step = self.step_counter;
        self.step_counter += 1;

        // NaN check
        if energy.is_nan() || enstrophy.is_nan() {
            self.violations.push(Violation::NanDetected {
                step,
                field: "energy/enstrophy".to_string(),
            });
            self.prev_energy = Some(energy);
            self.prev_dt = Some(dt);
            return;
        }

        // Energy monotonicity
        if let Some(prev) = self.prev_energy {
            if energy > prev + 1e-14 * prev.abs().max(1e-30) {
                self.violations.push(Violation::EnergyIncrease {
                    step,
                    prev,
                    curr: energy,
                });
            }

            // Energy budget: |dE/dt + 2*nu*Omega| / |2*nu*Omega|
            if let Some(prev_dt) = self.prev_dt {
                if prev_dt > 0.0 {
                    let de_dt = (energy - prev) / prev_dt;
                    let expected = -2.0 * nu * enstrophy;
                    let denom = expected.abs().max(1e-30);
                    let residual = (de_dt - expected).abs() / denom;
                    if residual > self.config.energy_budget_tol {
                        self.violations.push(Violation::EnergyBudgetViolation {
                            step,
                            residual,
                            de_dt,
                            expected,
                        });
                    }
                }
            }
        }

        // Divergence-free
        if max_divergence > self.config.divergence_tol {
            self.violations.push(Violation::DivergenceViolation {
                step,
                max_div: max_divergence,
            });
        }

        // Parseval identity
        if parseval_residual > self.config.parseval_tol {
            self.violations.push(Violation::ParsevalViolation {
                step,
                residual: parseval_residual,
            });
        }

        self.prev_energy = Some(energy);
        self.prev_dt = Some(dt);
    }

    /// Simple check for backward compatibility (used in run loop).
    pub fn check_diagnostics(&mut self, diag: &crate::scalar::ScalarDiagnostics, _nu: f64) {
        let step = diag.step;

        // NaN check on all fields
        if diag.energy.is_nan() || diag.enstrophy.is_nan() || diag.max_vorticity.is_nan() {
            self.violations.push(Violation::NanDetected {
                step,
                field: "diagnostics".to_string(),
            });
        }

        // Energy monotonicity
        if let Some(prev) = self.prev_energy {
            if diag.energy > prev + 1e-14 * prev.abs().max(1e-30) {
                self.violations.push(Violation::EnergyIncrease {
                    step,
                    prev,
                    curr: diag.energy,
                });
            }
        }
        self.prev_energy = Some(diag.energy);
    }

    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }

    pub fn should_halt(&self) -> bool {
        self.config.halt_on_violation && self.has_violations()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_accepts_decaying_energy() {
        let mut audit = ConservationAudit::new();
        audit.check_full(1.0, 10.0, 0.01, 0.01, 0.0, 1e-14);
        audit.check_full(0.9, 9.0, 0.01, 0.01, 0.0, 1e-14);
        // Only energy monotonicity checked here (budget may flag due to artificial data)
        assert!(
            !audit
                .violations
                .iter()
                .any(|v| matches!(v, Violation::EnergyIncrease { .. }))
        );
    }

    #[test]
    fn audit_catches_energy_increase() {
        let mut audit = ConservationAudit::new();
        audit.check_full(1.0, 10.0, 0.01, 0.01, 0.0, 1e-14);
        audit.check_full(1.1, 10.0, 0.01, 0.01, 0.0, 1e-14);
        assert!(
            audit
                .violations
                .iter()
                .any(|v| matches!(v, Violation::EnergyIncrease { .. }))
        );
    }

    #[test]
    fn audit_catches_nan() {
        let mut audit = ConservationAudit::new();
        audit.check_full(f64::NAN, 0.0, 0.01, 0.01, 0.0, 1e-14);
        assert!(
            audit
                .violations
                .iter()
                .any(|v| matches!(v, Violation::NanDetected { .. }))
        );
    }

    #[test]
    fn audit_catches_divergence_violation() {
        let mut audit = ConservationAudit::new();
        audit.check_full(1.0, 10.0, 0.01, 0.01, 1.0, 1e-14);
        assert!(
            audit
                .violations
                .iter()
                .any(|v| matches!(v, Violation::DivergenceViolation { .. }))
        );
    }

    #[test]
    fn audit_catches_parseval_violation() {
        let mut audit = ConservationAudit::new();
        audit.check_full(1.0, 10.0, 0.01, 0.01, 0.0, 0.1);
        assert!(
            audit
                .violations
                .iter()
                .any(|v| matches!(v, Violation::ParsevalViolation { .. }))
        );
    }

    #[test]
    fn audit_config_defaults() {
        let cfg = AuditConfig::default();
        assert!((cfg.energy_budget_tol - 1e-4).abs() < 1e-10);
        assert!((cfg.divergence_tol - 1e-10).abs() < 1e-15);
        assert!((cfg.parseval_tol - 1e-12).abs() < 1e-17);
        assert!(!cfg.halt_on_violation);
    }

    #[test]
    fn should_halt_when_configured() {
        let cfg = AuditConfig {
            halt_on_violation: true,
            ..AuditConfig::default()
        };
        let mut audit = ConservationAudit::with_config(cfg);
        assert!(!audit.should_halt());
        audit.check_full(f64::NAN, 0.0, 0.01, 0.01, 0.0, 1e-14);
        assert!(audit.should_halt());
    }
}
