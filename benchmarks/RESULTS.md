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

## Reference-solver wall-clock head-to-head

How fast is vonkarman against independent DNS codes solving the same problem
(Taylor-Green, Re = 1600) on this box? Two reference solvers run here: hit3d (the
JHU Fortran pseudo-spectral DNS) and a serial numpy pseudo-spectral DNS in the
style of spectralDNS. Both reference solvers were measured to be single-threaded
on this machine (hit3d at 97 percent of one core; the numpy solver under
`OMP_NUM_THREADS=1`), so this is a best-tool-on-this-box comparison (resident GPU
versus serial CPU), not a same-hardware one. Both references use the 2/3-rule on
the N grid and a fixed dt = 0.001 as shipped; vonkarman uses 3/2 padding (a
1.5x-larger transform grid) and an adaptive ETD-RK4 step. The numbers below are
reported exactly as measured.

### Per-step throughput

Wall-clock for one integration step (pure stepping; for hit3d and the numpy solver
this is a 200-step run differenced against a 20-step run to remove one-time setup;
the references are single measurements, vonkarman is the gamut mean).

| N   | vonkarman resident GPU (ms) | hit3d Fortran, 1 core (ms) | numpy spectral, 1 thread (ms) |
|-----|-----------------------------|----------------------------|-------------------------------|
| 64  | 17.27                       | 29.94                      | 148.67                        |
| 128 | 164.22                      | 101.77                     | 2741.01                       |
| 256 | 1504.46                     | 876.50                     | 26806.71                      |

Read this with the work-per-step caveat. At N=64 the GPU step is the fastest even
though it transforms a 96^3 padded grid against hit3d's 64^3. At N=128 and N=256
hit3d's single-core step edges the GPU step (102 versus 164 ms, then 877 versus
1504 ms), but the GPU step is doing about 3.4x more FFT points (the 3/2 padded grid
versus the 2/3-rule N grid) and four nonlinear evaluations per ETD-RK4 step. Per-step
is therefore not the right axis on its own: the GPU does much more work per step, and
the solvers take very different step counts to reach a given time. The end-to-end
figure below settles it.

### Time to solution (to t = 10)

The honest end-to-end figure folds in the step count. hit3d and the numpy solver
take 10000 fixed steps of dt = 0.001 to reach t = 10. vonkarman's exponential
ETD-RK4 integrator treats the stiff viscous term exactly and takes far larger stable
steps under its adaptive CFL bound. Both vonkarman runs below are measured end to
end, including every CFL evaluation and ETD-coefficient rebuild.

| N   | vonkarman resident GPU | hit3d Fortran, 1 core      | numpy spectral, 1 thread    |
|-----|------------------------|----------------------------|-----------------------------|
| 64  | 219 steps, 6.42 s      | 10000 steps, 299 s (46.6x) | 10000 steps, 1487 s (232x)  |
| 128 | 465 steps, 120.0 s     | 10000 steps, 1018 s (8.5x) | 10000 steps, 27410 s (228x) |

vonkarman reaches t = 10 about 47x faster than the single-core Fortran reference at
N = 64 and about 8.5x faster at N = 128 (and roughly 230x faster than the serial
numpy solver at both). The lever is the step count: the exponential integrator needs
219 and 465 steps where the fixed-dt references need 10000, a 21x to 46x reduction.
The ratio shrinks from N = 64 to N = 128 because hit3d's lighter 2/3-rule step (no
padding, one core) scales better per step than vonkarman's heavier 3/2-padded
four-stage step, narrowing the per-step gap while the step-count advantage persists.

Two honest notes. First, hit3d's dt = 0.001 is its shipped, conservative value; an
explicit RK code could take a somewhat larger stable step and close part of the
step-count gap, so these ratios are against the references as configured, not an
algorithmic ceiling. Second, vonkarman's adaptive driver rebuilds the per-mode ETD
coefficients (Kassam-Trefethen contour integrals over all modes) whenever the CFL dt
drifts more than one percent, which the Taylor-Green spin-up triggers repeatedly.
That host rebuild is now parallelised across cores (values bit-identical to the
serial build, verified by the unchanged 1.52e-15 GPU-versus-CPU step match), which is
what brings the N = 128 end-to-end run inside budget: before it, the serial recompute
dominated and the N = 128 integration did not finish. The remaining adaptive overhead
(the CFL reduction still allocates and frees a small device buffer per pass) is minor.
The N = 256 end-to-end run is about half a GPU-hour (roughly 930 adaptive steps near
1.5 s each plus the CFL bound) and is left for a dedicated artifact run rather than
this measurement budget; the per-step and step-count scaling above bracket it.

Reproduce: vonkarman `cargo +nightly-2026-04-03 test --release -p vonkarman-periodic
--features cuda resident_walltime_to_t10 -- --ignored --nocapture` (N = 64/128/256;
the 256 case is the long one); hit3d via a bounded run of `benchmarks/hit3d`
(differenced ITMAX = 220 minus 20); numpy via the `benchmarks/spectraldns` venv
stepping the same RK4.

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

### Dissipation peak: a diagnostic convention factor plus resolution convergence

An earlier draft flagged that vonkarman over-predicted the peak dissipation rate
(epsilon ~ 0.020) versus the Brachet 1983 / van Rees 2011 value (~0.0126) and attributed
it loosely to resolution and convention. A full 256^3 run to t = 10 pins it down: two
effects, both now quantified.

First, a factor of two in the dissipation DIAGNOSTIC, not the solver, now FIXED. The
diagnostics had reported epsilon = 2 nu z with z the mean square vorticity. That z is
exactly <|omega|^2>: its measured initial value 0.74997 matches the analytic Taylor-Green
<|omega|^2> = 1/8 + 1/8 + 1/2 = 3/4 to five figures. The standard incompressible
dissipation rate is epsilon = nu <|omega|^2> = nu z (equivalently 2 nu times the mean
enstrophy, with mean enstrophy = (1/2)<|omega|^2>), so the old figure was a factor of two
high. The diagnostic is now epsilon = nu z everywhere it is computed (the resident and CPU
reference tests, the conservation audit, and the ScalarDiagnostics output); the helicity
dissipation, which carries no 1/2, correctly keeps its 2 nu factor, as does the
dissipation spectrum 2 nu k^2 E(k) where E(k) already holds the 1/2. The fix is confirmed
by the energy budget identity dE/dt = -nu <|omega|^2>: its closure residual drops from
about 50 percent (the masked factor of two) to about 3 percent (the O(dt) discretisation
floor) on the Taylor-Green test. This is a diagnostic-only convention: the solver evolves
u_hat and never uses epsilon, so the GPU == CPU and cross-solver results above are
unaffected.

Second, genuine resolution convergence. With the standard convention epsilon = nu z:

| run                | peak enstrophy z | peak epsilon = nu z | peak time |
|--------------------|------------------|---------------------|-----------|
| vonkarman N=128    | 16.50            | 0.0103              | 8.68      |
| vonkarman N=256    | 18.01            | 0.0113              | 8.81      |
| Brachet / van Rees | ~20.2            | 0.0126              | ~9.0      |

The peak rises 0.0103 to 0.0113 from N=128 to N=256, converging toward the literature
0.0126 FROM BELOW, with the peak time moving 8.68 to 8.81 to ~9.0. This is textbook
spectral-DNS behaviour: an under-resolved grid under-predicts the dissipation peak (it
misses the smallest-scale enstrophy) and approaches the converged value as resolution
rises. At N=256 the peak is within about 11 percent of the reference, up from about 18
percent at N=128; a 512^3 run would close most of the remainder. With the convention
corrected, both resolutions sit inside the [0.005, 0.015] band the acceptance gate was
written for, so the long-standing gate "overshoot" was the factor of two, not a physics
or solver error.

The 256^3 run integrated Taylor-Green Re=1600 to t = 10 in 1023 adaptive ETD-RK4 steps
(about 51 minutes on the RTX 5060), energy monotone throughout and retaining 58.6 percent
of E0 at t = 10 (the correct Re=1600 decay; only ~41 percent dissipates by t = 10),
NaN-free, inside the 8 GB card. Reproduce with the `taylor_green_re1600_resident_256_dissipation`
test in `vonkarman-periodic/tests/resident_reference.rs` (about half a GPU-hour).

Double-double precision (Phase 2) would tighten the f64 floor further but is not part of
these results.

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
