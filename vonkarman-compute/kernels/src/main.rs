//! GPU kernel sources for vonkarman-compute.
//!
//! These kernels are compiled to PTX by cuda-oxide (`cargo oxide build`) using
//! the FMA-capable fork backend, and the resulting PTX is checked into
//! `../src/ptx/`. The `vonkarman-compute` LIBRARY then loads that PTX at
//! RUNTIME via `cuda_core::CudaContext::load_module_from_ptx_src`, so the
//! library itself builds with a plain `cargo build` and needs no special
//! kernel-embedding step (Option B). See `../KERNEL_BUILD.md`.
//!
//! The `main` below exists only because `cargo oxide build` needs a binary
//! target; it is never run. Building emits `vonkarman_kernels.ptx`.

use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Pointwise physical-space cross product `c = u x omega`.
    ///
    /// Mirrors `vonkarman_compute::ops::cross::cross_product_inplace`:
    ///
    /// ```text
    /// c_x = u_y * o_z - u_z * o_y
    /// c_y = u_z * o_x - u_x * o_z
    /// c_z = u_x * o_y - u_y * o_x
    /// ```
    ///
    /// All slices have one entry per physical grid point. The mul-add form is
    /// written so the fork's FMA contraction fuses `a * b - c` into a single
    /// `fma.rn.f64` per component (verified by grepping the emitted PTX).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn cross_product(
        ux: &[f64],
        uy: &[f64],
        uz: &[f64],
        ox: &[f64],
        oy: &[f64],
        oz: &[f64],
        mut cx: DisjointSlice<f64>,
        mut cy: DisjointSlice<f64>,
        mut cz: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        // One thread-unique index `i` feeds all three disjoint outputs. The
        // single bounds check guards every write (all output slices share the
        // grid-point length), and `get_unchecked_mut(i)` is sound because `i`
        // is thread-unique (no two threads write the same slot) and the three
        // buffers are distinct (no cross-output aliasing).
        if i < cx.len() {
            // SAFETY: `i < cx.len()` checked above; all output slices have the
            // same length; `i` is a thread-unique index minted from hardware
            // special registers.
            unsafe {
                *cx.get_unchecked_mut(i) = uy[i] * oz[i] - uz[i] * oy[i];
                *cy.get_unchecked_mut(i) = uz[i] * ox[i] - ux[i] * oz[i];
                *cz.get_unchecked_mut(i) = ux[i] * oy[i] - uy[i] * ox[i];
            }
        }
    }

    /// Spectral curl `omega_hat = i k x u_hat`, pointwise.
    ///
    /// Mirrors `vonkarman_compute::ops::curl::curl_inplace` and
    /// `vonkarman_core::spectral_ops::SpectralOps::curl`. The velocity inputs
    /// `ux`/`uy`/`uz` and the vorticity outputs `ox`/`oy`/`oz` are interleaved
    /// complex buffers (`[re0, im0, re1, im1, ...]`, length `2 n`); the
    /// wavenumbers `kx`/`ky`/`kz` are real, length `n`. One thread per spectral
    /// grid point `i` writes the interleaved `(re, im)` slot `2 i, 2 i + 1`.
    ///
    /// Each output component is a difference of two products
    /// (`ky uz - kz uy`, and cyclic), written so the fork's FMA contraction
    /// fuses `a * b - c * d` into one `fma.rn.f64` per real output.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn curl(
        ux: &[f64],
        uy: &[f64],
        uz: &[f64],
        kx: &[f64],
        ky: &[f64],
        kz: &[f64],
        mut ox: DisjointSlice<f64>,
        mut oy: DisjointSlice<f64>,
        mut oz: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        // `kx.len()` is the spectral grid-point count `n`; the complex buffers
        // have length `2 n`, so guarding on `n` covers both interleaved slots.
        if i < kx.len() {
            let re = 2 * i;
            let im = 2 * i + 1;
            let kxi = kx[i];
            let kyi = ky[i];
            let kzi = kz[i];
            let uxr = ux[re];
            let uxi = ux[im];
            let uyr = uy[re];
            let uyi = uy[im];
            let uzr = uz[re];
            let uzi = uz[im];
            // SAFETY: `i < n` checked above; all complex output slices have
            // length `2 n` so both `re` and `im` are in bounds; `i` is a
            // thread-unique index, and the three outputs are distinct buffers.
            unsafe {
                // omega_x = i (ky uz - kz uy)
                *ox.get_unchecked_mut(re) = -(kyi * uzi - kzi * uyi);
                *ox.get_unchecked_mut(im) = kyi * uzr - kzi * uyr;
                // omega_y = i (kz ux - kx uz)
                *oy.get_unchecked_mut(re) = -(kzi * uxi - kxi * uzi);
                *oy.get_unchecked_mut(im) = kzi * uxr - kxi * uzr;
                // omega_z = i (kx uy - ky ux)
                *oz.get_unchecked_mut(re) = -(kxi * uyi - kyi * uxi);
                *oz.get_unchecked_mut(im) = kxi * uyr - kyi * uxr;
            }
        }
    }

    /// Leray projection `u_hat -= k (k . u_hat) / |k|^2`, pointwise in place.
    ///
    /// Mirrors `vonkarman_compute::ops::leray::leray_inplace` and
    /// `vonkarman_core::spectral_ops::SpectralOps::leray_project`. The velocity
    /// components `ux`/`uy`/`uz` are interleaved complex buffers (length `2 n`)
    /// modified in place; `kx`/`ky`/`kz` and `k2` (`= |k|^2`) are real (length
    /// `n`). The `k = 0` mode (`k2 < 1e-30`) is left unchanged (mean flow).
    ///
    /// Each update is `u - (k / k2) (k . u)`, written so the fork fuses the
    /// final `a - b * c` subtraction into one `fma.rn.f64` per real output.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn leray(
        kx: &[f64],
        ky: &[f64],
        kz: &[f64],
        k2: &[f64],
        mut ux: DisjointSlice<f64>,
        mut uy: DisjointSlice<f64>,
        mut uz: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < kx.len() {
            let k2i = k2[i];
            // Skip the k = 0 mode (preserve mean flow). The tiny floor matches
            // the CPU reference exactly.
            if k2i >= 1e-30 {
                let re = 2 * i;
                let im = 2 * i + 1;
                let kxi = kx[i];
                let kyi = ky[i];
                let kzi = kz[i];
                let inv_k2 = 1.0 / k2i;
                // SAFETY: `i < n` (kx.len()) checked above; the complex slices
                // have length `2 n` so `re` and `im` are in bounds; `i` is a
                // thread-unique index and the three buffers are distinct.
                unsafe {
                    let uxr = *ux.get_unchecked_mut(re);
                    let uxi = *ux.get_unchecked_mut(im);
                    let uyr = *uy.get_unchecked_mut(re);
                    let uyi = *uy.get_unchecked_mut(im);
                    let uzr = *uz.get_unchecked_mut(re);
                    let uzi = *uz.get_unchecked_mut(im);

                    // k . u_hat (complex dot with real k).
                    let kdu_re = kxi * uxr + kyi * uyr + kzi * uzr;
                    let kdu_im = kxi * uxi + kyi * uyi + kzi * uzi;

                    // u_hat -= (k / k2) (k . u_hat). Pre-scale each k by 1/k2 so
                    // the final subtraction `u - scale * kdu` fuses to one FMA.
                    let sx = kxi * inv_k2;
                    let sy = kyi * inv_k2;
                    let sz = kzi * inv_k2;
                    *ux.get_unchecked_mut(re) = uxr - sx * kdu_re;
                    *ux.get_unchecked_mut(im) = uxi - sx * kdu_im;
                    *uy.get_unchecked_mut(re) = uyr - sy * kdu_re;
                    *uy.get_unchecked_mut(im) = uyi - sy * kdu_im;
                    *uz.get_unchecked_mut(re) = uzr - sz * kdu_re;
                    *uz.get_unchecked_mut(im) = uzi - sz * kdu_im;
                }
            }
        }
    }

    /// ETD-RK4 stage 2 / stage 3 multiply: `out = exp_half * u + dt * a * n`.
    ///
    /// Mirrors `vonkarman_compute::ops::etd::etd_stage_axpy_inplace`. Acts on
    /// one velocity component: `u`, `n`, `out` are interleaved complex buffers
    /// (`[re0, im0, ...]`, length `2 n`); `exp_half` and `a` are real per-mode
    /// buffers (length `n`), each value applied to both interleaved slots of
    /// its mode. One thread per spectral mode `i` writes slots `2 i, 2 i + 1`.
    ///
    /// `dt * a[i]` is precomputed once per mode, then each slot is written as
    /// `exp_half * u + (dt a) * n` with plain arithmetic (no explicit
    /// `mul_add`), so the fork's FMA CONTRACTION folds the `(dt a) * n` product
    /// into the add as one `fma.rn.f64` per real slot (the cross/curl/leray
    /// convention; an explicit `mul_add` would instead emit a libdevice call).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn etd_stage_axpy(
        exp_half: &[f64],
        a: &[f64],
        dt: f64,
        u: &[f64],
        n: &[f64],
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        // `exp_half.len()` is the spectral mode count `n`; the complex buffers
        // have length `2 n`, so guarding on `n` covers both interleaved slots.
        if i < exp_half.len() {
            let re = 2 * i;
            let im = 2 * i + 1;
            let eh = exp_half[i];
            let dta = dt * a[i];
            // SAFETY: `i < n` checked above; `out` has length `2 n` so both
            // `re` and `im` are in bounds; `i` is a thread-unique index.
            unsafe {
                // exp_half * u + (dta) * n; the trailing product contracts into
                // the add as one fma.rn.f64 (fma(exp_half, u, dta * n)).
                *out.get_unchecked_mut(re) = eh * u[re] + dta * n[re];
                *out.get_unchecked_mut(im) = eh * u[im] + dta * n[im];
            }
        }
    }

    /// ETD-RK4 stage 4 multiply: `out = exp_full * u + dt * a41 * (2 n3 - n1)`.
    ///
    /// Mirrors `vonkarman_compute::ops::etd::etd_stage4_inplace`. Acts on one
    /// velocity component: `u`, `n1`, `n3`, `out` are interleaved complex
    /// buffers (length `2 n`); `exp_full` and `a41` are real per-mode buffers
    /// (length `n`). One thread per spectral mode `i`.
    ///
    /// `dt * a41[i]` is precomputed once per mode; the difference `2 n3 - n1`
    /// is written plainly so the contraction folds it to `fma(2, n3, -n1)`, and
    /// the outer `exp_full * u + dta * dn` folds to `fma(exp_full, u, dta * dn)`,
    /// two `fma.rn.f64` per real slot (no explicit `mul_add`, which would emit a
    /// libdevice call instead of letting the backend contract).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn etd_stage4(
        exp_full: &[f64],
        a41: &[f64],
        dt: f64,
        u: &[f64],
        n1: &[f64],
        n3: &[f64],
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < exp_full.len() {
            let re = 2 * i;
            let im = 2 * i + 1;
            let ef = exp_full[i];
            let dta = dt * a41[i];
            // SAFETY: `i < n` checked above; `out` has length `2 n` so both
            // `re` and `im` are in bounds; `i` is a thread-unique index.
            unsafe {
                // dn = 2 n3 - n1 contracts to fma(2, n3, -n1); the outer
                // exp_full * u + dta * dn contracts to fma(exp_full, u, dta * dn).
                let dn_re = 2.0 * n3[re] - n1[re];
                let dn_im = 2.0 * n3[im] - n1[im];
                *out.get_unchecked_mut(re) = ef * u[re] + dta * dn_re;
                *out.get_unchecked_mut(im) = ef * u[im] + dta * dn_im;
            }
        }
    }

    /// ETD-RK4 final update, in place on `u`:
    /// `u = exp_full * u + dt * (b1 n1 + b23 (n2 + n3) + b4 n4)`.
    ///
    /// Mirrors `vonkarman_compute::ops::etd::etd_final_inplace`. Acts on one
    /// velocity component: `n1`, `n2`, `n3`, `n4` are interleaved complex
    /// buffers (length `2 n`), `u` (length `2 n`) is read then written in
    /// place; `exp_full`, `b1`, `b23`, `b4` are real per-mode buffers
    /// (length `n`). One thread per spectral mode `i`.
    ///
    /// The right-hand side is written left-to-right as
    /// `b1 n1 + b23 (n2 + n3) + b4 n4` with plain arithmetic; the contraction
    /// folds each trailing product into the running sum (`fma(b23, n2 + n3,
    /// b1 n1)`, then `fma(b4, n4, ...)`), and the outer `exp_full * u + dt * rhs`
    /// folds to `fma(exp_full, u, dt * rhs)`, so each real slot contracts to a
    /// small chain of `fma.rn.f64` (no explicit `mul_add`, which would emit a
    /// libdevice call instead).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn etd_final(
        exp_full: &[f64],
        b1: &[f64],
        b23: &[f64],
        b4: &[f64],
        dt: f64,
        n1: &[f64],
        n2: &[f64],
        n3: &[f64],
        n4: &[f64],
        mut u: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < exp_full.len() {
            let re = 2 * i;
            let im = 2 * i + 1;
            let ef = exp_full[i];
            let b1i = b1[i];
            let b23i = b23[i];
            let b4i = b4[i];
            // SAFETY: `i < n` checked above; `u` has length `2 n` so both `re`
            // and `im` are in bounds; `i` is a thread-unique index. `u` is read
            // then written through the same DisjointSlice, like the leray kernel.
            unsafe {
                let ur = *u.get_unchecked_mut(re);
                let ui = *u.get_unchecked_mut(im);
                // rhs = b1 n1 + b23 (n2 + n3) + b4 n4, written so the trailing
                // products contract into the running sum:
                //   fma(b4, n4, fma(b23, n2 + n3, b1 * n1)).
                let s23_re = n2[re] + n3[re];
                let s23_im = n2[im] + n3[im];
                let rhs_re = b1i * n1[re] + b23i * s23_re + b4i * n4[re];
                let rhs_im = b1i * n1[im] + b23i * s23_im + b4i * n4[im];
                // u = exp_full * u + dt * rhs contracts to fma(exp_full, u, dt * rhs).
                *u.get_unchecked_mut(re) = ef * ur + dt * rhs_re;
                *u.get_unchecked_mut(im) = ef * ui + dt * rhs_im;
            }
        }
    }
}

fn main() {}
