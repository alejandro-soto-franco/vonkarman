# vonkarman benchmark results

Taylor-Green vortex, Re=1600, on a single NVIDIA RTX 5060 (8 GB, sm_120, CUDA 13.1),
all double precision (f64). Numbers are measured in release builds, warm, and are
reported exactly as observed. Where a result is unfavourable or limited by the
hardware, that is stated rather than hidden.

## What this is

vonkarman is a pure-Rust pseudo-spectral incompressible Navier-Stokes DNS solver on
the periodic torus. The same five pointwise spectral operators (curl, physical cross
product, Leray projection, the ETD-RK4 phi-multiply, and the 3/2 pad/truncate) are
written once, generic over the scalar type, and run on two interchangeable compute
backends:

- a CPU backend (rayon over host arrays, ndrustfft transforms), and
- a GPU backend (cuda-oxide kernels compiled to PTX, cuFFT transforms), with the
  spectral state kept device-resident so a full timestep performs zero host to device
  copies except a small scalar diagnostic pull.

The GPU stack is pure Rust down to the kernels: the operator kernels are written in
Rust and lowered to PTX by a fork of cuda-oxide (FMA contraction plus an O3 pass), and
the CPU body of each operator is the differential oracle that the GPU kernel is tested
against to near machine precision.

## Headline results

1. Unified accuracy. The GPU backend reproduces the validated CPU solver to 1.52e-15
   relative per ETD-RK4 step (a full nonlinear evaluation plus the four-stage update),
   which is at the f64 rounding floor. The CPU and GPU paths are the same algorithm and
   agree to machine precision.
2. Speed. The device-resident GPU step is about 28 to 35 times faster than the pure-Rust
   CPU step, and about 14 to 19 times faster than a non-resident cuFFT step that bounces
   each transform through host memory. The latter ratio isolates the value of device
   residency on its own.
3. Capacity. A 256^3 f64 Taylor-Green run fits in the 8 GB of a consumer RTX 5060, with
   a measured peak of 6.467 GiB and 0.893 GiB free, stable and NaN-free with monotone
   energy decay.
4. Pure-Rust unified CPU and GPU. One operator source serves both backends; the CPU body
   is the per-operator differential oracle for the GPU kernel.

## Speed

Per-step wall-clock for one ETD-RK4 step, measured in process after a warm-up, mean plus
or minus sample standard deviation. The resident path uses a fixed dt; the CPU and the
non-resident cuFFT baselines use adaptive-dt stepping (a few extra inverse transforms for
the CFL bound). The dominant cost, the 36 padded transforms per step, is shared by all
three. Five repeats for the resident path, three for the baselines.

| N   | CPU (ms/step)      | non-resident cuFFT (ms/step) | resident GPU (ms/step) | resident vs CPU | resident vs non-resident |
|-----|--------------------|------------------------------|------------------------|-----------------|--------------------------|
| 64  | 483.11 +/- 6.96    | 324.70 +/- 5.85              | 17.27 +/- 0.01         | 27.98x          | 18.81x                   |
| 128 | 5701.24 +/- 418.28 | 2873.41 +/- 400.13           | 164.22 +/- 0.30        | 34.72x          | 17.50x                   |
| 256 | 43850.72 +/- 69.20 | 21361.44 +/- 67.83           | 1504.46 +/- 5.98       | 29.15x          | 14.20x                   |

Notes.

- The resident path is the headline. Its variance is tiny (under 0.4 percent) because the
  step is GPU compute-bound and deterministic.
- The CPU and non-resident baselines are noisier, with a single slow outlier each at
  N=128 (thermal or scheduling); the minimum-of-repeats ratios are similar to the means.
- Profiling (nsys) shows the resident step at N >= 128 is compute-bound, with the GPU at
  about 99.96 percent utilisation and cuFFT taking about 81 percent of GPU time. The
  remaining throughput lever is therefore cuFFT throughput, not host-side launch overhead.
  We tested the obvious candidate, cuFFT batching, directly; see the next section.

Data: `results/timing_vonkarman_3way.csv` (one row per repeat). Figures: `results/figures/`
(`speed_table.png`, `steps_per_s.png`, `pareto.png`), produced by `pareto.py`.

## cuFFT batching: tested and measured, a negative result

A resident nonlinear evaluation runs nine padded transforms: six inverse (three velocity,
three vorticity) and three forward. The natural next optimisation is cuFFT batching, fusing
the three same-shape transforms of each group into one `cufftMakePlanMany` plan so a single
launch processes all three. We implemented it (`CufftBackend::new_device_only_batched`,
verified bit-identical to per-component transforms in both directions) and measured the
transform-only ceiling it could buy, with no pointwise kernels or host transfers in the way:

| N   | nine transforms separate (ms) | as three batched launches (ms) | speedup |
|-----|-------------------------------|--------------------------------|---------|
| 64  | 4.61                          | 4.44                           | 1.040x  |
| 128 | 38.29                         | 37.97                          | 1.008x  |
| 256 | 342.32                        | 341.72                         | 1.002x  |

This is the most batching could possibly give, and it is small: 4 percent at N=64, falling
to 0.2 percent at N=256. Because the transforms are about 81 percent of the step, the
end-to-end step win is bounded above by roughly 3 percent at N=64 and is effectively zero at
N >= 128. The reason is exactly what the profiling said: a single 3/2-padded transform
(96^3, 192^3, 384^3) already saturates the RTX 5060, so fusing the launches removes only the
negligible per-launch overhead, not any real serial bottleneck. Batching also costs memory:
a three-wide padded complex scratch is about 0.9 GiB at N=256, which would break the
256^3-in-8 GB result for no speed return.

We therefore measured it, kept the verified batched backend available, and deliberately did
NOT wire it into the validated step. The honest conclusion is that the resident step is
cuFFT-throughput-bound and batching the launches does not add throughput here; the real
levers are algorithmic (precision, or a different dealiasing) rather than launch fusion.
Reproduce with `cargo +nightly-2026-04-03 test --release -p vonkarman-periodic --features
cuda batched_vs_separate_transform_timing -- --ignored --nocapture`.

## Accuracy

Two independent statements.

1. GPU equals CPU. A full ETD-RK4 step on the resident GPU backend matches the CPU
   backend to 1.52e-15 relative from the same initial condition and dt, which is the f64
   rounding floor. Run to completion, the resident GPU validation at N=128 reproduces the
   CPU enstrophy peak to under one percent (16.5 versus 16.6, near t = 8.7 to 8.8); the
   small residual is the two runs choosing their adaptive dt sequences independently, not
   a difference in the physics. The two backends are the same algorithm.

2. Cross-solver validation. vonkarman is compared against hit3d (an independent Fortran
   pseudo-spectral DNS) and the Brachet 1983 and van Rees 2011 reference curves. vonkarman
   at N=128 and hit3d at N=256 agree to 1.1 percent L2 on the energy decay, and both place
   the enstrophy peak near t = 8.8 to 8.9. The peak enstrophy value differs (16.6 at N=128
   versus 20.7 at N=256), which is the expected resolution difference between the two grids.

A caveat reported honestly. vonkarman and hit3d both give a peak dissipation rate near
epsilon = 0.020, which is above the Brachet and van Rees reference band. The two
independent codes agree with each other, which points to a grid-resolution and
normalisation-convention difference rather than a solver error; a higher-resolution or a
higher-precision (double-double) run would tighten this. Double-double precision is the
planned Phase 2 and is not part of these results.

Figures: the energy, enstrophy, and dissipation overlays plus the L2-deviation table are
produced by `run_all.py compare` from the per-solver CSVs.

## Memory: 256^3 in 8 GB

The 256^3 resident solver was packed to fit a single 8 GB card. The 3/2-dealiasing grid is
384^3, so naive scratch overflows the card. Two changes brought it under budget: the
physical cross product is computed in place over the vorticity buffers (a read-all-then-write
kernel, verified bit-identical to the out-of-place version), removing three padded real
buffers; and the resident cuFFT plans are device-only with a single shared work area, which
reclaimed about 1.1 GiB that the host-bounce constructor had used. Measured peak device
memory at 256^3 is 6.467 GiB used with 0.893 GiB free. The dt bound is computed on the
device (a max-reduction), so the step needs no host pull for the CFL condition.

## Reference solver coverage on this box

See `results/reference_coverage.md` for the full picture. In short: hit3d (Fortran) is
runnable and its N=256 Re=1600 curve is on disk; spectraldns (a serial numpy pseudo-spectral
DNS) and dedalus (energy only) are runnable but slow on CPU; turbogenpy provides an
initial-condition spectrum only. The accuracy comparison here uses hit3d as the independent
reference.

## Honest caveats

- Single consumer GPU, 8 GB. No multi-GPU. f64 to 256^3, no double-double yet.
- The resident step is cuFFT-bound at N >= 128. We tested cuFFT batching as the candidate
  next lever and measured that it buys at most 4 percent (N=64) down to 0.2 percent (N=256)
  of transform time, so it is not integrated; see the batching section above. Further gains
  would have to come from algorithmic changes, not launch fusion.
- The CPU and non-resident baselines use adaptive dt while the resident path uses a fixed
  dt in the timing harness; the per-step work (36 padded transforms) is the same.
- Reference-solver coverage is partial on this box, as documented.

## Reproduction

- Speed: `cargo +nightly-2026-04-03 test --release -p vonkarman-periodic --features cuda
  three_way_step_timing -- --ignored --nocapture`, then `python gen_3way_timing_csv.py`
  and `python pareto.py results/timing_vonkarman_3way.csv`.
- GPU vs CPU accuracy: the resident and CPU step match is checked by the cuda tests in
  vonkarman-periodic; the N=128 trajectory is `taylor_green_re1600_resident_128`.
- Cross-solver accuracy: `python run_all.py compare`.
