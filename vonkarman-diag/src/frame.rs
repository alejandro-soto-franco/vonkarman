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
#[derive(Debug, Clone, Default, Serialize)]
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
    /// THE CONDITIONAL TEST. The specification requires `alpha <~ nu Phi`, a claim about
    /// HIGH vorticity where a singularity would form, which the volume-integrated
    /// `payoff_ratio` cannot see because it is dominated by the bulk.
    ///
    /// Binned on the BUDGET DENSITIES `rho^2 alpha = omega . S omega` and
    /// `rho^2 Phi = |grad omega|^2 - |grad rho|^2`, over 12 logarithmic bins of
    /// `|omega| / max|omega|` above a floor of 1e-3, and grouped in fours for these
    /// columns. Binning the densities rather than `alpha` and `Phi` separately means no
    /// division by `rho^2`, so the low-vorticity void where the director spins
    /// arbitrarily fast stops contaminating the measurement, and each value is literally
    /// that vorticity band's contribution to (PAYOFF).
    pub cond_ratio_q1: f64,
    pub cond_ratio_q2: f64,
    pub cond_ratio_q3: f64,
    pub cond_ratio_q4: f64,
    /// Conditional mean `|omega|` in the top bin, for context on where q4 sits.
    pub cond_rho_q4: f64,
    /// Count-weighted log-log fit of the per-bin (PAYOFF) ratio against the bin's mean
    /// `|omega|`, over 12 logarithmic vorticity bins, with its standard error, R^2 and
    /// the number of bins that carried enough samples to be used. The error and R^2 are
    /// there so a noisy fit is recognised as noisy instead of being read as a trend.
    pub cond_slope_stderr: f64,
    pub cond_r2: f64,
    pub cond_nbins: f64,
    /// The SAME fit run against the FULL dissipation `<|grad omega|^2>` instead of the
    /// transverse part, i.e. against the actual enstrophy budget rather than the
    /// (PAYOFF) weakening of it. If the transverse slope is positive while this one is
    /// not, the longitudinal dissipation `<|grad rho|^2>` is what protects the intense
    /// regions and (PAYOFF) fails precisely because it discards the term doing the work.
    pub cond_slope_full: f64,
    pub cond_r2_full: f64,
    /// The exact decomposition of FA`ViscousLength: `ratio = Ghat (l/l_nu)^2` with
    /// `Ghat = alpha/rho` (amplitude degree 0, purely geometric), `l = rho/|grad omega|`
    /// and `l_nu = Sqrt[nu/rho]` the scale at which production and dissipation balance.
    /// `cond_ghat_slope` tests whether Ghat is amplitude-flat, and
    /// `cond_lratio_slope = (cond_slope - cond_ghat_slope)/2` is then the exponent in
    /// `l/l_nu ~ rho^e`. e > 0 means intense structures sit ABOVE viscous equilibrium;
    /// regularity in this frame would want e <= 0.
    pub cond_ghat_slope: f64,
    pub cond_lratio_slope: f64,
    /// THE DEFINITIONAL TEST. The exponent e in `<l/l_nu> ~ rho^e` for three independent
    /// structure scales: `l1 = rho/|grad omega|` (first order, mixes modulus and
    /// director), `l2 = rho/|grad rho|` (first order, MODULUS ONLY), and
    /// `l3 = Sqrt(rho/|Lap omega|)` (SECOND order). Only `l1` satisfies the exact
    /// decomposition `ratio = Ghat (l1/l_nu)^2`; the other two are separate probes. If e
    /// agrees across all three the departure from viscous equilibrium is physical; if it
    /// moves with the definition, the framing was ours rather than the flow's.
    pub e_grad_omega: f64,
    pub e_grad_rho: f64,
    pub e_laplacian: f64,
    /// THE COLLOCATION CHECK. Every direction-field quantity below divides
    /// by `|omega|`, so a null sitting ON a mesh point is evaluated at the
    /// singularity rather than near it and returns a plausible number that
    /// is entirely quadrature artefact. The unperturbed Taylor-Green datum
    /// does exactly that: its nulls lie on the lines `x, y in {0, pi}` of
    /// the planes `z in {0, pi}`, all mesh points for even `n`. At 128^3
    /// the geodesic curvature read of order `1e16` there, and a near-null
    /// share returned exactly `1.0000`.
    ///
    /// `min_vorticity` is `min |omega|` over the grid. `null_cell_margin`
    /// is the distance from the mesh point that attains it to the null it
    /// is approaching, in cells: `min|omega| / (|grad omega| h)` at that
    /// point, first order in the mesh. Below about `1e-3` a null is on a
    /// node in all but name and no direction-field column of that frame
    /// means anything. `null_fraction` is the volume fraction below the
    /// `1e-6 max|omega|` floor at which `xi` is regularised, which says how
    /// much of the domain the regularisation is holding up.
    ///
    /// Measured on the canonical datum at 32^3, `xi_energy` reads `1.96e29`
    /// against `5.55` for the same field sampled a half cell off the nulls,
    /// with 1.15 per cent of the domain under the floor. At 64^3 it is
    /// `1.32e28` against `7.13`. The finite-difference column `xi_energy_fd`
    /// moves from `4.85` to `5.08` over the same pair, which is why the
    /// condition survived: one route to the quantity looked sane.
    ///
    /// The hazard generalises to any symmetric datum whose critical points
    /// land on mesh points. `ic::taylor_green_shifted` moves the mesh off
    /// them for the common one.
    pub min_vorticity: f64,
    pub null_cell_margin: f64,
    pub null_fraction: f64,
    /// Log-log slope of the conditional ratio against the conditional mean `|omega|`,
    /// over the bins that carry samples and a positive ratio. THIS IS THE VERDICT:
    /// slope <= 0 means the ratio is bounded or decaying as the vorticity grows, so the
    /// depletion saturates and the mechanism holds where it matters; slope > 0 means it
    /// grows with the amplitude and the route is refuted.
    pub cond_slope: f64,
}

/// Below this many cells, a vorticity null is on a mesh point in all but
/// name and every direction-field column of that frame is quadrature
/// artefact rather than flow.
pub const NULL_COLLOCATION_CELLS: f64 = 1e-3;

impl FrameDiagnostics {
    /// Whether a vorticity null sits close enough to a mesh point to make
    /// the direction-field columns of this frame meaningless.
    ///
    /// The signature is recognisable without this: a near-null share of
    /// exactly `1.0000`, or a curvature at machine-overflow scale, in a
    /// flow whose vorticity is smooth and bounded. Recognising it after the
    /// fact is what this replaces.
    pub fn null_is_collocated(&self) -> bool {
        self.null_cell_margin < NULL_COLLOCATION_CELLS
    }

    /// One line naming the condition, for a caller that logs it. `None`
    /// when the mesh clears the nulls.
    pub fn null_collocation_warning(&self) -> Option<String> {
        self.null_is_collocated().then(|| {
            format!(
                "step {}: a vorticity null sits {:.2e} cells from a mesh point \
                 (min |omega| = {:.3e}, {:.4} of the volume under the xi floor). \
                 Every direction-field column of this frame is quadrature \
                 artefact. Use a datum whose critical points miss the mesh: \
                 taylor-green takes a `shift`.",
                self.step, self.null_cell_margin, self.min_vorticity, self.null_fraction
            )
        })
    }

    /// CSV header matching `csv_row`.
    pub fn csv_header() -> &'static str {
        "step,time,enstrophy,max_vorticity,f_rms,alpha_p_rms,rho_all,rho_hi,\
xi_energy,nem_energy,xi_energy_hi,nem_energy_hi,coherence_w,hi_fraction,\
nu,production,transverse_dissipation,full_dissipation,payoff_ratio,\
production_hi,transverse_dissipation_hi,payoff_ratio_hi,transverse_fraction,\
cond_ratio_q1,cond_ratio_q2,cond_ratio_q3,cond_ratio_q4,cond_rho_q4,cond_slope,\
cond_slope_stderr,cond_r2,cond_nbins,cond_slope_full,cond_r2_full,cond_ghat_slope,cond_lratio_slope,e_grad_omega,e_grad_rho,e_laplacian,\
xi_energy_fd,full_dissipation_grad,parseval_residual,transverse_dissipation_fd,fd_recovery,\
min_vorticity,null_cell_margin,null_fraction"
    }

    /// One CSV row (no trailing newline).
    pub fn csv_row(&self) -> String {
        format!(
            "{},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
{:.9e},{:.9e},{:.9e}",
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
            self.cond_slope_stderr,
            self.cond_r2,
            self.cond_nbins,
            self.cond_slope_full,
            self.cond_r2_full,
            self.cond_ghat_slope,
            self.cond_lratio_slope,
            self.e_grad_omega,
            self.e_grad_rho,
            self.e_laplacian,
            self.xi_energy_fd,
            self.full_dissipation_grad,
            self.parseval_residual,
            self.transverse_dissipation_fd,
            self.fd_recovery,
            self.min_vorticity,
            self.null_cell_margin,
            self.null_fraction,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The header and the row are written by hand from the same field list,
    /// so they drift apart the moment a column is added to one of them. A
    /// reader that maps columns by position then reads every value past the
    /// insertion under the wrong name, silently.
    #[test]
    fn the_csv_header_and_row_agree_on_column_count() {
        let header = FrameDiagnostics::csv_header();
        let row = FrameDiagnostics::default().csv_row();
        assert_eq!(
            header.split(',').count(),
            row.split(',').count(),
            "header:\n{header}\nrow:\n{row}"
        );
    }

    /// The three columns #1 adds are at the end, where a reader mapping by
    /// name finds them and one mapping by position is unaffected.
    #[test]
    fn the_null_columns_are_present_and_last() {
        let header = FrameDiagnostics::csv_header();
        let cols: Vec<&str> = header.split(',').collect();
        assert_eq!(
            &cols[cols.len() - 3..],
            &["min_vorticity", "null_cell_margin", "null_fraction"]
        );
    }

    /// `null_is_collocated` reads the margin against one stated threshold,
    /// so the warning and any downstream filter agree by construction.
    #[test]
    fn a_zero_margin_is_collocated_and_a_half_cell_is_not() {
        let mut d = FrameDiagnostics {
            null_cell_margin: 0.0,
            ..FrameDiagnostics::default()
        };
        assert!(d.null_is_collocated());
        assert!(d.null_collocation_warning().is_some());

        d.null_cell_margin = 0.5;
        assert!(!d.null_is_collocated());
        assert!(d.null_collocation_warning().is_none());
    }
}
