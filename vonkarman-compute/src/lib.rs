//! Compute backends for the vonkarman solver.
//!
//! This crate provides a [`CpuBackend`] (rayon over host `Vec`s) implementing
//! the [`vonkarman_core::ComputeBackend`] trait, plus the pointwise spectral
//! operator bodies (in [`ops`]). Each operator body is written once, generic
//! over `F: Float`, so the same source can later be compiled to a GPU kernel
//! and verified against the CPU result as the differential oracle.
//!
//! The optional `cuda` feature adds a GPU backend ([`CudaBackend`]) over
//! `cuda-core`; the CPU path has no GPU dependencies, so the default workspace
//! build stays pure Rust and needs no CUDA toolkit.

pub mod cpu;
pub mod ops;

pub use cpu::CpuBackend;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
mod ptx;

#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;
