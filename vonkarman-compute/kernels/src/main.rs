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
}

fn main() {}
