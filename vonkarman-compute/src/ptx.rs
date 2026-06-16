//! Offline-compiled kernel PTX, embedded at compile time.
//!
//! The PTX is produced by the isolated `kernels/` crate via
//! `cargo oxide build` with the FMA-capable cuda-oxide fork backend and
//! checked into `src/ptx/`. The library loads it at runtime through
//! `cuda_core::CudaContext::load_module_from_ptx_src`. See `KERNEL_BUILD.md`.

/// PTX for all vonkarman GPU kernels (currently: `cross_product`).
pub const KERNELS: &str = include_str!("ptx/vonkarman_kernels.ptx");
