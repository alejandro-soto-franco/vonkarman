//! Frame / coherence / pressure diagnostics for the Clifford-NS regularity programme.
//!
//! These probe the geometry-aware quantities that a norm-based diagnostic cannot see,
//! along a real (possibly stressed) trajectory on T^3:
//!
//! - `rho` = ||alpha_p|| / ||f|| with the frame-projected pressure
//!   alpha_p = -xi_i xi_j R_i R_j f and the CLMS null form f = |S|^2 - (1/2)|omega|^2.
//!   rho in [~1/3, 1] measures how close the flow sits to the pressure WALL (rho -> 1)
//!   versus the transverse CRACK (rho -> 0). Emitted whole-domain and over the
//!   high-|omega| region where the Constantin-Fefferman depletion is relevant.
//! - `xi_energy` = <|grad xi|^2> and `nem_energy` = <|grad Xi|^2>, Xi = xi (x) xi.
//!   The Xi (nematic) energy is defect-honest: it stays finite across orientation
//!   flips (antiparallel geometry) where the vector xi-energy blows up. Comparing the
//!   two along a real flow is the numeric face of "Xi-Lipschitz suffices".
//! - `coherence_w` = <|omega|^{1/2} |grad Xi|^2>, the omega_Xi_crit density: the
//!   critical, geometry-aware, defect-honest coherence functional whose dissipation
//!   constant kappa is the object the coupled-Liouville closure hinges on.
//!
//! The dissipation constant kappa and the closure margin kappa*g - p^2 are estimated
//! in post-processing from the emitted time series (kappa from the coherence-energy
//! decay, p = rho); this module supplies the raw per-step ingredients.

use serde::Serialize;

/// Per-step frame / coherence / pressure diagnostics.
#[derive(Debug, Clone, Serialize)]
pub struct FrameDiagnostics {
    pub time: f64,
    pub step: u64,
    /// Enstrophy <|omega|^2> and peak vorticity, for cross-reference / stress level.
    pub enstrophy: f64,
    pub max_vorticity: f64,
    /// RMS of the CLMS null form f = |S|^2 - (1/2)|omega|^2.
    pub f_rms: f64,
    /// RMS of the frame-projected pressure alpha_p = -xi_i xi_j R_i R_j f.
    pub alpha_p_rms: f64,
    /// rho = ||alpha_p||/||f|| over the whole domain (wall = 1, crack = 0).
    pub rho_all: f64,
    /// rho over the high-|omega| region (|omega| > 0.3 max|omega|), the CF-relevant zone.
    pub rho_hi: f64,
    /// Coherence energies: vector <|grad xi|^2> and nematic <|grad Xi|^2>.
    pub xi_energy: f64,
    pub nem_energy: f64,
    /// The same over the high-|omega| region.
    pub xi_energy_hi: f64,
    pub nem_energy_hi: f64,
    /// omega_Xi_crit density <|omega|^{1/2} |grad Xi|^2> (critical, defect-honest).
    pub coherence_w: f64,
    /// Volume fraction of the high-|omega| region (context for the _hi columns).
    pub hi_fraction: f64,
}

impl FrameDiagnostics {
    /// CSV header matching `csv_row`.
    pub fn csv_header() -> &'static str {
        "step,time,enstrophy,max_vorticity,f_rms,alpha_p_rms,rho_all,rho_hi,\
xi_energy,nem_energy,xi_energy_hi,nem_energy_hi,coherence_w,hi_fraction"
    }

    /// One CSV row (no trailing newline).
    pub fn csv_row(&self) -> String {
        format!(
            "{},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
            self.step,
            self.time,
            self.enstrophy,
            self.max_vorticity,
            self.f_rms,
            self.alpha_p_rms,
            self.rho_all,
            self.rho_hi,
            self.xi_energy,
            self.nem_energy,
            self.xi_energy_hi,
            self.nem_energy_hi,
            self.coherence_w,
            self.hi_fraction,
        )
    }
}
