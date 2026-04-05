pub mod taylor_green;

pub use taylor_green::taylor_green;

use serde::{Deserialize, Serialize};

/// Supported initial condition types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum IcType {
    TaylorGreen,
}
