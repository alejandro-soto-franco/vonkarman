pub mod abc;
pub mod anti_parallel;
pub mod colliding_vortex_rings;
pub mod kida_pelz;
pub mod random_isotropic;
pub mod taylor_green;

pub use abc::abc_flow;
pub use anti_parallel::anti_parallel_tubes;
pub use colliding_vortex_rings::colliding_vortex_rings;
pub use kida_pelz::kida_pelz;
pub use random_isotropic::random_isotropic;
pub use taylor_green::taylor_green;

use serde::{Deserialize, Serialize};

/// Supported initial condition types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum IcType {
    TaylorGreen,
    Abc {
        #[serde(default = "default_one")]
        a: f64,
        #[serde(default = "default_one")]
        b: f64,
        #[serde(default = "default_one")]
        c: f64,
    },
    AntiParallelTubes {
        #[serde(default = "default_one")]
        circulation: f64,
        #[serde(default = "default_core_radius")]
        core_radius: f64,
        #[serde(default = "default_one")]
        separation: f64,
        #[serde(default = "default_perturbation")]
        perturbation: f64,
    },
    KidaPelz,
    RandomIsotropic {
        #[serde(default = "default_k_peak")]
        k_peak: f64,
        #[serde(default = "default_energy")]
        energy: f64,
        #[serde(default = "default_seed")]
        seed: u64,
    },
    CollidingVortexRings {
        #[serde(default = "default_ring_radius")]
        ring_radius: f64,
        #[serde(default = "default_ring_core_radius")]
        core_radius: f64,
        #[serde(default = "default_one")]
        circulation: f64,
        #[serde(default = "default_ring_separation")]
        separation: f64,
        #[serde(default = "default_ring_axis")]
        axis: usize,
    },
}

fn default_one() -> f64 {
    1.0
}
fn default_core_radius() -> f64 {
    0.3
}
fn default_perturbation() -> f64 {
    0.1
}
fn default_k_peak() -> f64 {
    4.0
}
fn default_energy() -> f64 {
    0.5
}
fn default_seed() -> u64 {
    42
}
fn default_ring_radius() -> f64 {
    1.0
}
fn default_ring_core_radius() -> f64 {
    0.35
}
fn default_ring_separation() -> f64 {
    2.0
}
fn default_ring_axis() -> usize {
    2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colliding_rings_toml_defaults() {
        let toml_src = r#"type = "colliding-vortex-rings""#;
        let ic: IcType = toml::from_str(toml_src).unwrap();
        match ic {
            IcType::CollidingVortexRings {
                ring_radius,
                core_radius,
                circulation,
                separation,
                axis,
            } => {
                assert_eq!(ring_radius, 1.0);
                assert_eq!(core_radius, 0.35);
                assert_eq!(circulation, 1.0);
                assert_eq!(separation, 2.0);
                assert_eq!(axis, 2);
            }
            _ => panic!("wrong variant"),
        }
    }
}
