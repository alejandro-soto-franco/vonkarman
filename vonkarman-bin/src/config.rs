// Phase 2+ fields are parsed but not yet consumed by the runner.
#![allow(dead_code)]

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ExperimentConfig {
    pub run: RunConfig,
    pub domain: DomainConfig,
    pub physics: PhysicsConfig,
    pub precision: Option<PrecisionConfig>,
    pub integrator: Option<IntegratorConfig>,
    pub initial_condition: IcConfig,
    pub commit_cycle: Option<CommitCycleConfig>,
    pub diagnostics: Option<DiagnosticsConfig>,
    pub termination: TerminationConfig,
}

#[derive(Debug, Deserialize)]
pub struct RunConfig {
    pub name: String,
    pub output_dir: String,
}

#[derive(Debug, Deserialize)]
pub struct DomainConfig {
    #[serde(rename = "type")]
    pub domain_type: String,
    /// Grid points per axis (cubic grid).
    pub n: usize,
    /// Domain length (default: 2*pi).
    #[serde(default = "default_domain_length")]
    pub l: f64,
    /// FFT backend: "auto" (default), "cufft", or "cpu".
    #[serde(default = "default_backend")]
    pub backend: String,
}

fn default_backend() -> String {
    "auto".to_string()
}

fn default_domain_length() -> f64 {
    2.0 * std::f64::consts::PI
}

#[derive(Debug, Deserialize)]
pub struct PhysicsConfig {
    pub nu: f64,
}

#[derive(Debug, Deserialize)]
pub struct PrecisionConfig {
    #[serde(default = "default_precision")]
    pub tier: String,
}

fn default_precision() -> String {
    "f64".to_string()
}

#[derive(Debug, Deserialize)]
pub struct IntegratorConfig {
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default = "default_cfl_safety")]
    pub cfl_safety: f64,
}

fn default_method() -> String {
    "etd-rk4".to_string()
}

fn default_cfl_safety() -> f64 {
    0.5
}

#[derive(Debug, Deserialize)]
pub struct IcConfig {
    #[serde(rename = "type")]
    pub ic_type: String,
}

#[derive(Debug, Deserialize)]
pub struct CommitCycleConfig {
    #[serde(default = "default_one")]
    pub diagnostics_interval: u64,
    #[serde(default = "default_snapshot_interval")]
    pub snapshot_interval: u64,
    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval: u64,
}

fn default_one() -> u64 {
    1
}
fn default_snapshot_interval() -> u64 {
    1000
}
fn default_checkpoint_interval() -> u64 {
    5000
}

#[derive(Debug, Deserialize)]
pub struct DiagnosticsConfig {
    #[serde(default)]
    pub conservation_audit: bool,
}

#[derive(Debug, Deserialize)]
pub struct TerminationConfig {
    pub max_steps: Option<u64>,
    pub max_time: Option<f64>,
    pub max_wall_hours: Option<f64>,
    pub max_vorticity_threshold: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_config() {
        let toml_str = r#"
[run]
name = "test-run"
output_dir = "./output/test"

[domain]
type = "periodic3d"
n = 64

[physics]
nu = 6.25e-4

[initial_condition]
type = "taylor-green"

[termination]
max_steps = 1000
"#;
        let cfg: ExperimentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.run.name, "test-run");
        assert_eq!(cfg.domain.n, 64);
        assert!((cfg.physics.nu - 6.25e-4).abs() < 1e-10);
        assert_eq!(cfg.termination.max_steps, Some(1000));
    }

    #[test]
    fn parse_config_with_backend() {
        let toml_str = r#"
[run]
name = "test"
output_dir = "./output/test"

[domain]
type = "periodic3d"
n = 64
backend = "cufft"

[physics]
nu = 1e-3

[initial_condition]
type = "taylor-green"

[termination]
max_steps = 10
"#;
        let cfg: ExperimentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.domain.backend, "cufft");
    }

    #[test]
    fn parse_config_default_backend() {
        let toml_str = r#"
[run]
name = "test"
output_dir = "./output/test"

[domain]
type = "periodic3d"
n = 64

[physics]
nu = 1e-3

[initial_condition]
type = "taylor-green"

[termination]
max_steps = 10
"#;
        let cfg: ExperimentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.domain.backend, "auto");
    }

    #[test]
    fn parse_full_config() {
        let toml_str = r#"
[run]
name = "tg-re1600"
output_dir = "./output/tg-re1600"

[domain]
type = "periodic3d"
n = 384
l = 6.283185307179586

[physics]
nu = 6.25e-4

[precision]
tier = "f64"

[integrator]
method = "etd-rk4"
cfl_safety = 0.5

[initial_condition]
type = "taylor-green"

[commit_cycle]
diagnostics_interval = 1
snapshot_interval = 1000
checkpoint_interval = 5000

[diagnostics]
conservation_audit = true

[termination]
max_steps = 1_000_000
max_time = 10.0
max_wall_hours = 48.0
max_vorticity_threshold = 1e12
"#;
        let cfg: ExperimentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.domain.n, 384);
        assert_eq!(cfg.integrator.as_ref().unwrap().method, "etd-rk4");
        assert_eq!(cfg.termination.max_steps, Some(1_000_000));
        assert!((cfg.termination.max_time.unwrap() - 10.0).abs() < 1e-14);
    }
}
