# GPU kernel hosting and build (cuda-oxide, Option B)

`CudaBackend` is a type in the `vonkarman-compute` **library**, consumed by the
`vonkarman-bin` binary. cuda-oxide's `#[cuda_module]` macro plus
`cargo oxide build` are designed around a *binary* whose PTX is embedded into
the executable image, so the natural "Option A" (kernels in the library,
embedded by a special build of the binary) does not fit a plain library that is
itself built with `cargo build`. We use **Option B** instead.

## Decision: Option B (offline PTX, runtime load)

1. The kernel sources live in a small, isolated crate `kernels/`
   (its own `[workspace]`, pinned to the cuda-oxide fork branch
   `pr/fma-and-opt-pipeline` and the nightly the fork requires). It hosts the
   `#[cuda_module] mod kernels { ... }` and a no-op `main` (a binary target is
   required only so `cargo oxide build` has something to compile).

2. `cargo oxide build` (with the FMA-capable fork backend) compiles those
   kernels to a single PTX file, `kernels/vonkarman_kernels.ptx`.

3. That PTX is copied into the library source tree at
   `src/ptx/vonkarman_kernels.ptx` and pulled into the library with
   `include_str!` (compile-time text, no codegen at library build time).

4. At RUNTIME the library loads the PTX through `cuda_core`:

   ```rust
   let module = ctx.load_module_from_ptx_src(PTX)?;     // cuModuleLoadData (JIT)
   let func   = module.load_function("cross_product")?; // cuModuleGetFunction
   // launch with cuda_core::launch_kernel_on_stream and raw (ptr, len) args
   ```

This keeps the GPU code path entirely inside the library (Option B requirement)
and means the library builds with a plain `cargo build`: the only compile-time
dependency the `cuda` feature adds is the CUDA SDK that `cuda-bindings` needs
for the *host* driver bindings, not any kernel codegen. Building the PTX is an
explicit, occasional offline step.

This was de-risked by experiment: a plain `cargo build` binary (no
`#[cuda_module]`, no `cargo oxide`) loaded the spike PTX via
`load_module_from_ptx_src`, looked up the entry by name, and launched it raw,
producing correct results. The same path is what `CudaBackend` uses.

## Kernel ABI

Each `&[f64]` / `DisjointSlice<f64>` kernel parameter lowers to a **pair** of
PTX params `(ptr: u64, len: u64)`. So the host builds the `cuLaunchKernel`
parameter array as one `&CUdeviceptr` plus one `&u64` (length) per slice, in
source order. The cross-product kernel takes nine slices
(`ux, uy, uz, ox, oy, oz, cx, cy, cz`) and therefore 18 launch params.

## Build command

A STALE cached backend (`~/.cargo/cuda-oxide/librustc_codegen_cuda.so`) emits
PTX with NO `fma.rn.f64`. To get fused mul-add you must point at a freshly
built fork backend. Build it once:

```bash
cd <cuda-oxide>/crates/rustc-codegen-cuda \
  && CARGO_TARGET_DIR=/home/cargo-targets/cuda-oxide-codegen \
     cargo +nightly-2026-04-03 build
# -> /home/cargo-targets/cuda-oxide-codegen/debug/librustc_codegen_cuda.so
```

Then regenerate and refresh the checked-in PTX:

```bash
CUDA_OXIDE_BACKEND=/home/cargo-targets/cuda-oxide-codegen/debug/librustc_codegen_cuda.so \
  vonkarman-compute/kernels/build-ptx.sh
```

The script builds `kernels/` with `cargo oxide build`, reports the
`fma.rn.f64` count (warns if zero, i.e. a stale backend), and copies the PTX to
`src/ptx/vonkarman_kernels.ptx`. The current checked-in cross-product PTX has
three `fma.rn.f64`, one fused mul-add per output component.
