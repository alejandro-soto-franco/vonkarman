//! Brinkman volume penalisation of a stationary body, and a downstream fringe.
//!
//! The body is represented by a smooth mask `chi` in `[0, 1]`, one inside and
//! zero outside, and enforces no-slip by relaxing the velocity toward zero with
//! time constant `eta_p`:
//!
//! ```text
//!   d(u)/dt = - chi / eta_p * u
//! ```
//!
//! Integrating that term explicitly would force `dt <~ eta_p` and so cap how
//! hard no-slip can be enforced. It is instead applied as its exact solution
//! over a half step, Strang split around the IF-RK4 step, which is
//! unconditionally stable and frees `eta_p` from the step size.
//!
//! The fringe is a strip at the downstream end where vorticity is relaxed to
//! zero, so the wake does not re-enter the periodic box as inflow.
//!
//! The exponential substep leaves the state un-band-limited. Multiplying the
//! velocity by `exp(-chi * h / eta_p)` is a product in physical space, so it
//! spreads energy across the whole spectrum, and the mask edge is the sharpest
//! feature on the grid. The substep applies no 2/3 mask, and masking there is
//! ruled out: it would truncate the body's vorticity sheet and soften no-slip.
//! The 2/3 rule the advective term applies removes aliasing only from a state
//! band-limited to two thirds of `k_max`, so with content above that cutoff
//! some aliases fold back below it where the mask cannot reach, and the
//! advection mask therefore does not remove all aliasing. What holds the
//! residue down is the mask edge width `delta`, which spans about three cells
//! and so keeps the sharpest feature resolved, together with viscosity, which
//! drains the top of the spectrum every step. Sharpening `delta` or raising
//! `sigma_max` both push the residue up.
//! `the_penalisation_substep_leaves_the_state_close_to_band_limited` in
//! `tests/penalisation.rs` measures it and holds it under a threshold.
//!
//! The force the body exerts on the flow is not computed here. The Angot
//! estimator `F = (1 / eta_p) * integral(chi * u)` is exact only for an
//! *explicit* penalisation term, where the interior velocity scales linearly
//! with `eta_p`. Under the exponential substep this crate uses, it does not:
//! the estimator's `1 / eta_p` prefactor amplifies the velocity that survives
//! across the smoothed mask edge, and that surviving band shrinks slower than
//! `1 / eta_p` grows, so the estimate diverges as `eta_p -> 0` rather than
//! converging. [`crate::Sim::body_force`] instead measures the momentum the
//! substeps actually remove, which is exact for the scheme as implemented.

use ndarray::Array2;

use crate::Spectral2D;

/// A stationary penalised body together with a downstream fringe.
pub struct Penalisation {
    /// Mask `chi` in `[0, 1]`, one inside the body.
    chi: Array2<f64>,
    /// Fringe relaxation rate `sigma(x)`, zero outside the strip.
    sigma: Array2<f64>,
    /// Penalisation time constant.
    eta_p: f64,
    /// Cell area, for area-weighted integrals over the grid (currently the
    /// momentum-removed force estimate in [`crate::Sim::body_force`]).
    cell_area: f64,
}

impl Penalisation {
    /// A circular cylinder of radius `radius` centred at `(cx, cy)`, plus a
    /// fringe rising from `fringe_start` over `fringe_width` to `sigma_max`.
    ///
    /// The mask edge is smoothed over roughly three cells, which keeps the
    /// Fourier representation of `chi * u` from ringing at the interface.
    ///
    /// The edge width is a single physical length sampled along both axes, so
    /// it is set from the coarser of `dx` and `dy`. Sizing it from `dx` alone
    /// under-resolves the edge in `y` on tall cells: at a cell aspect ratio of
    /// 4 the mask becomes a hard 0/1 step in `y`, which is the interface
    /// ringing the smoothing exists to prevent. The aspect ratio is asserted to
    /// stay within 2, beyond which the smoothing costs more resolution than the
    /// geometry is worth.
    #[allow(clippy::too_many_arguments)]
    pub fn cylinder(
        spec: &Spectral2D,
        cx: f64,
        cy: f64,
        radius: f64,
        eta_p: f64,
        fringe_start: f64,
        fringe_width: f64,
        sigma_max: f64,
    ) -> Self {
        assert!(radius > 0.0, "radius must be positive");
        assert!(eta_p > 0.0, "eta_p must be positive");
        assert!(fringe_width > 0.0, "fringe_width must be positive");
        // A negative rate makes exp(-sigma * h) > 1, an amplifying fringe.
        assert!(sigma_max >= 0.0, "sigma_max must not be negative");
        let (nx, ny) = (spec.nx(), spec.ny());
        let (dx, dy) = spec.spacing();
        let aspect = (dx / dy).max(dy / dx);
        assert!(
            aspect <= 2.0,
            "cell aspect ratio {aspect} exceeds 2: the mask edge is a single \
             physical length sampled along both axes, so it under-resolves on \
             cells this anisotropic"
        );
        let delta = 0.75 * dx.max(dy);
        let mut chi = Array2::<f64>::zeros((nx, ny));
        let mut sigma = Array2::<f64>::zeros((nx, ny));
        for i in 0..nx {
            let x = i as f64 * dx;
            // Smooth ramp 0 -> 1 across the fringe strip.
            let t = ((x - fringe_start) / fringe_width).clamp(0.0, 1.0);
            let s = sigma_max * t * t * (3.0 - 2.0 * t);
            for j in 0..ny {
                let y = j as f64 * dy;
                let r = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                chi[[i, j]] = 0.5 * (1.0 - ((r - radius) / delta).tanh());
                sigma[[i, j]] = s;
            }
        }
        Self {
            chi,
            sigma,
            eta_p,
            cell_area: dx * dy,
        }
    }

    /// The body mask.
    pub fn chi(&self) -> &Array2<f64> {
        &self.chi
    }

    /// Velocity decay factor `exp(-chi * h / eta_p)` for a substep of length `h`.
    pub(crate) fn velocity_decay(&self, h: f64) -> Array2<f64> {
        self.chi.mapv(|c| (-c * h / self.eta_p).exp())
    }

    /// Vorticity decay factor `exp(-sigma * h)` for a substep of length `h`.
    pub(crate) fn vorticity_decay(&self, h: f64) -> Array2<f64> {
        self.sigma.mapv(|s| (-s * h).exp())
    }

    /// Grid cell area `dx * dy`, for area-weighting a sum over the grid into
    /// an integral.
    pub(crate) fn cell_area(&self) -> f64 {
        self.cell_area
    }
}
