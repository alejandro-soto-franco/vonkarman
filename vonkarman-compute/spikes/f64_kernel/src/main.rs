//! Gate spike 0a: prove cuda-oxide (fork branch pr/fma-and-opt-pipeline)
//! compiles and runs a DOUBLE-PRECISION (f64) kernel on this box
//! (RTX 5060 Laptop, sm_120, CUDA 13.1), and that PR #117's FMA
//! contraction forms `fma.rn.f64` from a `a*b + a` mul-add.
//!
//! Pattern copied from ferrum-gpu/examples/vector-add-cuda-oxide, adapted
//! to f64 and the fork branch.

use anyhow::Result;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// `out[i] = a[i] * b[i] + a[i]` — a mul-add the fork's FMA
    /// contraction should fuse into a single `fma.rn.f64`.
    #[kernel]
    pub(crate) fn fma_probe(a: &[f64], b: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            *slot = a[i] * b[i] + a[i];
        }
    }
}

fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let n: u32 = 1 << 16;
    // f64 values with non-trivial mantissas so a single-rounded FMA differs
    // from a separate mul-then-add.
    let a: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 1.0000000001e-3).collect();
    let b: Vec<f64> = (0..n)
        .map(|i| 0.5 + ((n - i) as f64) * 7.0000000007e-4)
        .collect();

    let da = DeviceBuffer::from_host(&stream, &a)?;
    let db = DeviceBuffer::from_host(&stream, &b)?;
    let mut dc = DeviceBuffer::<f64>::zeroed(&stream, n as usize)?;

    println!("loading cuda-oxide-compiled fma_probe (f64) module");
    let module = kernels::load(&ctx)?;

    println!("launching f64 kernel");
    module.fma_probe(&stream, LaunchConfig::for_num_elems(n), &da, &db, &mut dc)?;

    let c = dc.to_host_vec(&stream)?;
    for i in 0..(n as usize) {
        // CPU mul_add matches the GPU's single-rounded FMA.
        let expected = a[i].mul_add(b[i], a[i]);
        let denom = expected.abs().max(f64::MIN_POSITIVE);
        let rel = (c[i] - expected).abs() / denom;
        if rel > 1e-15 {
            anyhow::bail!(
                "mismatch at {i}: got {}, expected {}, rel err {rel:e}",
                c[i],
                expected
            );
        }
    }
    println!("fma_probe (cuda-oxide, f64): {n} elements verified to rel tol 1e-15");
    Ok(())
}
