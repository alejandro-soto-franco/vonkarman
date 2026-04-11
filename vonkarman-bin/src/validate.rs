//! Input config validation. Catches bad configs before solver construction.

use crate::config::ExperimentConfig;

/// Validate an experiment config. Returns a list of errors (empty = valid).
pub fn validate_config(config: &ExperimentConfig) -> Vec<String> {
    let mut errors = Vec::new();

    // Grid size: must be power of 2 and >= 8
    let n = config.domain.n;
    if n < 8 || !n.is_power_of_two() {
        errors.push(format!("domain.n must be a power of 2 and >= 8, got {n}"));
    }

    // Viscosity: must be positive
    if config.physics.nu <= 0.0 {
        errors.push(format!("physics.nu must be > 0, got {}", config.physics.nu));
    }

    // Domain length: must be positive
    if config.domain.l <= 0.0 {
        errors.push(format!("domain.l must be > 0, got {}", config.domain.l));
    }

    // Termination: at least one condition
    let has_termination = config.termination.max_steps.is_some()
        || config.termination.max_time.is_some()
        || config.termination.max_wall_hours.is_some();
    if !has_termination {
        errors.push(
            "at least one termination condition required (max_steps, max_time, or max_wall_hours)"
                .to_string(),
        );
    }

    // Backend string
    let valid_backends = ["auto", "cufft", "gpu", "cuda", "cpu", "ndrustfft"];
    let backend = config.domain.backend.to_lowercase();
    if !valid_backends.contains(&backend.as_str()) {
        errors.push(format!(
            "domain.backend must be one of {valid_backends:?}, got \"{}\"",
            config.domain.backend
        ));
    }

    // CFL safety factor (if specified)
    if let Some(ref integrator) = config.integrator
        && (integrator.cfl_safety <= 0.0 || integrator.cfl_safety > 1.0)
    {
        errors.push(format!(
            "integrator.cfl_safety must be in (0, 1], got {}",
            integrator.cfl_safety
        ));
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ExperimentConfig;

    fn minimal_valid_toml() -> &'static str {
        r#"
[run]
name = "test"
output_dir = "/tmp/vonkarman_test"

[domain]
type = "periodic3d"
n = 16

[physics]
nu = 0.01

[initial_condition]
type = "taylor-green"

[termination]
max_steps = 10
"#
    }

    #[test]
    fn valid_config_passes() {
        let cfg: ExperimentConfig = toml::from_str(minimal_valid_toml()).unwrap();
        let errors = validate_config(&cfg);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn rejects_non_power_of_two_n() {
        let toml = minimal_valid_toml().replace("n = 16", "n = 15");
        let cfg: ExperimentConfig = toml::from_str(&toml).unwrap();
        let errors = validate_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("power of 2")));
    }

    #[test]
    fn rejects_n_too_small() {
        let toml = minimal_valid_toml().replace("n = 16", "n = 4");
        let cfg: ExperimentConfig = toml::from_str(&toml).unwrap();
        let errors = validate_config(&cfg);
        assert!(errors.iter().any(|e| e.contains(">= 8")));
    }

    #[test]
    fn rejects_zero_viscosity() {
        let toml = minimal_valid_toml().replace("nu = 0.01", "nu = 0.0");
        let cfg: ExperimentConfig = toml::from_str(&toml).unwrap();
        let errors = validate_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("nu must be > 0")));
    }

    #[test]
    fn rejects_no_termination() {
        let toml = r#"
[run]
name = "test"
output_dir = "/tmp/vonkarman_test"

[domain]
type = "periodic3d"
n = 16

[physics]
nu = 0.01

[initial_condition]
type = "taylor-green"

[termination]
"#;
        let cfg: ExperimentConfig = toml::from_str(toml).unwrap();
        let errors = validate_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("termination condition")));
    }

    #[test]
    fn rejects_invalid_backend() {
        let toml = minimal_valid_toml().replace(
            "[domain]\ntype = \"periodic3d\"\nn = 16",
            "[domain]\ntype = \"periodic3d\"\nn = 16\nbackend = \"vulkan\"",
        );
        let cfg: ExperimentConfig = toml::from_str(&toml).unwrap();
        let errors = validate_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("backend")));
    }

    #[test]
    fn collects_multiple_errors() {
        let toml = r#"
[run]
name = "test"
output_dir = "/tmp/vonkarman_test"

[domain]
type = "periodic3d"
n = 7

[physics]
nu = -1.0

[initial_condition]
type = "taylor-green"

[termination]
"#;
        let cfg: ExperimentConfig = toml::from_str(toml).unwrap();
        let errors = validate_config(&cfg);
        assert!(
            errors.len() >= 3,
            "expected at least 3 errors, got {}: {errors:?}",
            errors.len()
        );
    }
}
