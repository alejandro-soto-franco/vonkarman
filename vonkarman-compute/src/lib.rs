//! Compute backends for the vonkarman solver.
//!
//! This crate provides a [`CpuBackend`] (rayon over host `Vec`s) implementing
//! the [`vonkarman_core::ComputeBackend`] trait, plus the pointwise spectral
//! operator bodies (in [`ops`]). Each operator body is written once, generic
//! over `F: Float`, so the same source can later be compiled to a GPU kernel
//! and verified against the CPU result as the differential oracle.
//!
//! A future `cuda` feature will add a GPU backend; the CPU path here has no
//! GPU dependencies so the default workspace build stays pure Rust.

pub mod cpu;
pub mod ops;

pub use cpu::CpuBackend;
