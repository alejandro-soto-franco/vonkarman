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
    /// `xi_energy` is the BAND-LIMITED value, from the exact identity
    /// `|grad omega|^2 = |grad rho|^2 + rho^2 |grad xi|^2` with spectral derivatives of
    /// omega. `xi_energy_fd` is the old second-order finite-difference value, kept so
    /// the damping stays visible: it recovers only ~0.36 of the true dissipation at
    /// n=64 and ~0.59 at n=128 in a stressed flow.
    pub xi_energy: f64,
    pub xi_energy_fd: f64,
    /// `<|grad omega|^2>` from the physical-space spectral gradients, which must agree
    /// with the Parseval `full_dissipation`. `parseval_residual` is their relative gap:
    /// two independent paths to one quantity, so a nonzero residual means the
    /// spectral-derivative batch, the Hermitian weighting or the normalisation is wrong.
    pub full_dissipation_grad: f64,
    pub parseval_residual: f64,
    /// Transverse dissipation by the old finite-difference route, and the recovery
    /// fraction `<|grad omega|^2>_fd / <|grad omega|^2>_spectral`, which measures the
    /// finite-difference damping directly (omega is band-limited, so the spectral value
    /// is exact). Retained as the honest record of how much the old estimator lost.
    pub transverse_dissipation_fd: f64,
    pub fd_recovery: f64,
    pub nem_energy: f64,
    /// The same over the high-|omega| region.
    pub xi_energy_hi: f64,
    pub nem_energy_hi: f64,
    /// omega_Xi_crit density <|omega|^{1/2} |grad Xi|^2> (critical, defect-honest).
    pub coherence_w: f64,
    /// Volume fraction of the high-|omega| region (context for the _hi columns).
    pub hi_fraction: f64,
    /// Kinematic viscosity, carried so the CSV is self-contained (the payoff ratio
    /// divides by it, and the ratio is meaningless without knowing which nu it used).
    pub nu: f64,
    /// Enstrophy production <omega . S omega> = <rho^2 alpha>, the left side of (PAYOFF).
    pub production: f64,
    /// Transverse dissipation density <|omega|^2 |grad xi|^2> = <rho^2 Phi>, the right
    /// side of (PAYOFF) before multiplying by nu. Taken BAND-LIMITED, as
    /// `<|grad omega|^2 - |grad rho|^2>` with spectral derivatives of omega, so it needs
    /// no division and neither aliases nor damps.
    pub transverse_dissipation: f64,
    /// Full enstrophy dissipation <|grad omega|^2>, by Parseval in spectral space
    /// (omega IS band-limited, so this is exact and costs no transform). Equals the
    /// transverse part plus the longitudinal part <|grad |omega||^2>.
    pub full_dissipation: f64,
    /// THE MEASUREMENT: R = <omega . S omega> / (nu <|omega|^2 |grad xi|^2>).
    /// (PAYOFF) requires R <= 1 up to the subcritical remainder, and the programme's
    /// specification requires the depletion to saturate at rate 1/rho, which is exactly
    /// the statement that R stays bounded as the flow stresses.
    pub payoff_ratio: f64,
    /// The same three over the high-|omega| region, where the Constantin-Fefferman
    /// depletion is the relevant mechanism.
    pub production_hi: f64,
    pub transverse_dissipation_hi: f64,
    pub payoff_ratio_hi: f64,
    /// Fraction of the full dissipation carried by the transverse (director) part,
    /// measured like-for-like against a finite-difference `<|grad omega|^2>` since the
    /// numerator is finite-difference too. (PAYOFF) discards the longitudinal part, so
    /// this says how much is given away: a small fraction means the transverse-only form
    /// is a strong weakening of the enstrophy budget.
    pub transverse_fraction: f64,
    /// THE CONDITIONAL TEST. The specification requires `alpha <~ nu Phi` with
    /// `alpha = xi . S xi`, and that is a statement about behaviour at HIGH vorticity,
    /// where a singularity would form. The volume-integrated `payoff_ratio` cannot see
    /// it, being dominated by the bulk. These are `<alpha | rho> / (nu <Phi | rho>)` in
    /// four bins of `|omega| / max|omega|`: [0,1/4), [1/4,1/2), [1/2,3/4), [3/4,1].
    /// Counts cancel in the ratio, so each is a clean conditional average.
    pub cond_ratio_q1: f64,
    pub cond_ratio_q2: f64,
    pub cond_ratio_q3: f64,
    pub cond_ratio_q4: f64,
    /// Conditional mean `|omega|` in the top bin, for context on where q4 sits.
    pub cond_rho_q4: f64,
    /// Log-log slope of the conditional ratio against the conditional mean `|omega|`,
    /// over the bins that carry samples and a positive ratio. THIS IS THE VERDICT:
    /// slope <= 0 means the ratio is bounded or decaying as the vorticity grows, so the
    /// depletion saturates and the mechanism holds where it matters; slope > 0 means it
    /// grows with the amplitude and the route is refuted.
    pub cond_slope: f64,
}

impl FrameDiagnostics {
    /// CSV header matching `csv_row`.
    pub fn csv_header() -> &'static str {
        "step,time,enstrophy,max_vorticity,f_rms,alpha_p_rms,rho_all,rho_hi,\
xi_energy,nem_energy,xi_energy_hi,nem_energy_hi,coherence_w,hi_fraction,\
nu,production,transverse_dissipation,full_dissipation,payoff_ratio,\
production_hi,transverse_dissipation_hi,payoff_ratio_hi,transverse_fraction,\
cond_ratio_q1,cond_ratio_q2,cond_ratio_q3,cond_ratio_q4,cond_rho_q4,cond_slope,\
xi_energy_fd,full_dissipation_grad,parseval_residual,transverse_dissipation_fd,fd_recovery"
    }

    /// One CSV row (no trailing newline).
    pub fn csv_row(&self) -> String {
        format!(
            "{},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
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
            self.nu,
            self.production,
            self.transverse_dissipation,
            self.full_dissipation,
            self.payoff_ratio,
            self.production_hi,
            self.transverse_dissipation_hi,
            self.payoff_ratio_hi,
            self.transverse_fraction,
            self.cond_ratio_q1,
            self.cond_ratio_q2,
            self.cond_ratio_q3,
            self.cond_ratio_q4,
            self.cond_rho_q4,
            self.cond_slope,
            self.xi_energy_fd,
            self.full_dissipation_grad,
            self.parseval_residual,
            self.transverse_dissipation_fd,
            self.fd_recovery,
        )
    }
}
