//! Device-resident spectral solver (CUDA), zero per-step host transfers.
//!
//! [`ResidentSolver`] reproduces the validated CPU physics of
//! [`crate::nonlinear::compute_nonlinear`] and [`Periodic3D::etd_rk4_step`]
//! entirely on the GPU. Once the velocity field and the per-mode coefficients
//! are uploaded at construction, a [`ResidentSolver::nonlinear`] call or a
//! [`ResidentSolver::step`] runs only device kernels and cuFFT transforms on
//! resident buffers, moving NOTHING across the PCIe bus. The only transfer
//! points are the setup uploads (wavenumbers, the initial field, the ETD
//! coefficients) and any explicit diagnostic download the caller asks for.
//!
//! The CPU [`Periodic3D`] is the ground truth: this module is verified AGAINST
//! it (see the cuda-gated tests), never the other way round.
//!
//! # The dealiasing normalisation (device form)
//!
//! The host [`vonkarman_fft::dealiased_cross_product`] uses NORMALISED host
//! transforms: it multiplies the padded spectrum by `scale = N_padded / N`, then
//! its `c2r_3d` divides by `N_padded` (so the two collapse to `scale / N_padded
//! = 1 / N`); after the forward `r2c_3d` (unnormalised) and the truncation it
//! multiplies by `inv_scale = N / N_padded`. Here `N = nx * ny * nz` is the
//! ORIGINAL grid real length and `N_padded = pnx * pny * pnz` the padded one.
//!
//! The resident cuFFT transforms ([`CufftBackend::c2r_3d_dev`] /
//! [`CufftBackend::r2c_3d_dev`]) are BOTH unnormalised, so the device path:
//!
//! 1. pads, then multiplies the padded spectrum by `1 / N` (the collapsed
//!    factor), then `c2r_3d_dev`;
//! 2. after `r2c_3d_dev` and truncation, multiplies the N-grid result by
//!    `N / N_padded` (`= inv_scale`).
//!
//! These two factors are applied with the [`CudaBackend::scale_cplx`] kernel,
//! deliberately not folded into the pad/truncate kernels (which are pure,
//! bit-identity-tested index remaps).
//!
//! # Scratch and residency
//!
//! All device buffers are allocated ONCE at construction and reused every call
//! (no per-step allocation). The cross product needs all six padded physical
//! fields live at once; the solver holds three padded `u_phys` and three padded
//! `omega_phys` real buffers, plus ONE padded complex scratch reused for every
//! pad/transform and every forward-transform/truncate. The cross product is
//! computed IN PLACE over `omega_phys` by the aliasing-safe, register-first
//! `cross_product_inplace` kernel (it latches all six inputs before any write),
//! so there is NO separate `cross_phys` triple: that saves three padded real
//! buffers (~1.27 GiB at N=256), the change that brings N=256 inside 8 GB. The
//! N-grid `omega_hat[3]`, the ETD stage buffers and per-mode coefficient buffers
//! complete the resident set. The nonlinear term is written DIRECTLY into the
//! requested stage buffer, so there is no extra `n_hat` triple or any
//! device-to-device copy. See the module test for the peak-memory budget
//! (which includes the cuFFT work areas, queried live via `cufftGetSize`).

use crate::etd::EtdCoeffs;
use num_complex::Complex;
use vonkarman_compute::CudaBackend;
use vonkarman_compute::cuda::{CplxBuf, RealBuf};
use vonkarman_core::ComputeBackend;
use vonkarman_core::field::GridSpec;
use vonkarman_core::spectral_ops::SpectralOps;
use vonkarman_fft::CufftBackend;

/// Device-resident GPU solver mirroring the CPU [`Periodic3D`] physics.
///
/// Holds the CUDA backend, an N-grid and a padded cuFFT plan (both bound to the
/// backend stream), resident wavenumber buffers, the velocity state `u_hat`, and
/// all scratch and ETD coefficient buffers, every one allocated once. A
/// [`Self::nonlinear`] or [`Self::step`] call issues only device work.
///
/// [`Periodic3D`]: crate::solver::Periodic3D
pub struct ResidentSolver {
    /// The GPU backend (owns the primary context, kernel launches, transfers).
    backend: CudaBackend,
    /// Grid metadata (original `N` grid).
    grid: GridSpec,
    /// cuFFT plan on the original `N` grid, bound to the backend stream.
    fft: CufftBackend,
    /// cuFFT plan on the 3/2-padded grid, bound to the backend stream.
    fft_padded: CufftBackend,

    /// Number of spectral (half-complex) points on the `N` grid.
    cplx_len: usize,
    /// Number of spectral (half-complex) points on the padded grid.
    pcplx_len: usize,
    /// Number of real points on the `N` grid (`= N`).
    real_len: usize,
    /// Number of real points on the padded grid (`= N_padded`).
    preal_len: usize,

    /// Original-grid spectral shape `(snx, sny, snz)`.
    sshape: (usize, usize, usize),
    /// Padded-grid spectral shape `(pnx, pny, pnz)`.
    pshape: (usize, usize, usize),

    /// Wavenumber buffers, expanded flat to length `cplx_len` in `[ix, iy, iz]`
    /// row-major order (uploaded once from `SpectralOps`).
    kx: RealBuf<f64>,
    ky: RealBuf<f64>,
    kz: RealBuf<f64>,
    /// `|k|^2` at each spectral point, flat length `cplx_len`.
    k2: RealBuf<f64>,

    /// Spectral velocity state `u_hat[3]` on the `N` grid (length `cplx_len`).
    u_hat: [CplxBuf<f64>; 3],
    /// Spectral vorticity `omega_hat[3]` on the `N` grid (curl of the nonlinear
    /// input field, scratch reused every nonlinear call).
    omega_hat: [CplxBuf<f64>; 3],

    /// One padded complex scratch buffer (length `pcplx_len`), reused for every
    /// pad/transform and every forward-transform/truncate (one component at a
    /// time, so a single padded spectral buffer suffices).
    pad_scratch: CplxBuf<f64>,
    /// Three padded physical velocity buffers (length `preal_len`).
    u_phys: [RealBuf<f64>; 3],
    /// Three padded physical vorticity buffers (length `preal_len`). The
    /// aliasing-safe cross product writes its result IN PLACE over these (the
    /// register-first kernel latches all six inputs before any write), so no
    /// separate `cross_phys` triple is held. This removes 3 padded real buffers
    /// (~1.27 GiB at N=256), the single change that brings N=256 inside 8 GB.
    omega_phys: [RealBuf<f64>; 3],

    /// ETD stage nonlinear terms `n1, n2, n3, n4`, each three `N`-grid spectral
    /// components. Allocated once; the nonlinear writes directly into these.
    stage_n: [[CplxBuf<f64>; 3]; 4],
    /// ETD intermediate state `temp[3]` on the `N` grid.
    temp: [CplxBuf<f64>; 3],

    /// One N-grid complex scratch (length `cplx_len`) used only by the resident
    /// CFL path: `u_hat[c]` is copied here before the destructive cuFFT C2R so
    /// the primary state survives. Small (one N-grid complex buffer).
    cfl_scratch: CplxBuf<f64>,

    /// Per-mode ETD coefficient buffers (flat length `cplx_len`), recomputed on
    /// the host and re-uploaded only when `dt` changes.
    etd_exp_full: RealBuf<f64>,
    etd_exp_half: RealBuf<f64>,
    etd_a21: RealBuf<f64>,
    etd_a41: RealBuf<f64>,
    etd_b1: RealBuf<f64>,
    etd_b23: RealBuf<f64>,
    etd_b4: RealBuf<f64>,
    /// The `dt` the resident ETD coefficient buffers were built for (`< 0` means
    /// "not yet built", forcing the first upload).
    etd_dt: f64,
    /// Kinematic viscosity (for recomputing ETD coefficients when `dt` changes).
    nu: f64,
    /// Host-side flat `|k|^2`, kept to rebuild ETD coefficients without a
    /// download (`lambda = -nu k^2` is a host scalar computation per mode).
    k2_host: Vec<f64>,
}

/// Which N-grid spectral velocity field a nonlinear evaluation reads from.
#[derive(Clone, Copy)]
enum NlInput {
    /// The primary velocity state `u_hat`.
    UHat,
    /// The ETD intermediate state `temp`.
    Temp,
}

impl ResidentSolver {
    /// Builds a resident GPU solver for `grid` at viscosity `nu`, uploading the
    /// initial spectral velocity and all wavenumbers ONCE.
    ///
    /// Allocates every device buffer up front; subsequent [`Self::nonlinear`]
    /// and [`Self::step`] calls allocate nothing and transfer nothing. The two
    /// cuFFT plans are created for the `N` grid and the 3/2-padded grid and bound
    /// to the backend stream so they run on the same ordered stream as the
    /// pointwise kernels.
    ///
    /// `u_hat_host` is the host spectral velocity in the backend's flat
    /// `[ix, iy, iz]` interleaved-complex order (length `cplx_len` per
    /// component), the layout [`SpectralOps`] and the CPU solver produce.
    ///
    /// # Panics
    ///
    /// Panics if a cuFFT plan cannot be created or bound (no usable GPU); the
    /// caller is expected to have a working CUDA backend already.
    pub fn new(
        backend: CudaBackend,
        grid: GridSpec,
        nu: f64,
        u_hat_host: &[Vec<Complex<f64>>; 3],
    ) -> Self {
        let ops = SpectralOps::<f64>::new(&grid);
        let (snx, sny, snz) = grid.spectral_shape();
        let pg = grid.padded_3half();
        let (pnx, pny, pnz) = pg.spectral_shape();

        let cplx_len = snx * sny * snz;
        let pcplx_len = pnx * pny * pnz;
        let real_len = grid.nx * grid.ny * grid.nz;
        let preal_len = pg.nx * pg.ny * pg.nz;

        // cuFFT plans on both grids, bound to the backend stream so the FFT and
        // the pointwise kernels share one ordered stream (no cross-stream sync).
        // DEVICE-ONLY plans: the resident path uses only the `*_dev` entry points
        // on its own resident buffers, never the host FFT methods, so we skip the
        // per-plan host-path d_real/d_complex scratch (~1.1 GiB at N=256) and let
        // the two plans share ONE work area. This is what brings N=256 inside the
        // 8 GB card; with the host-path constructor the scratch alone overflows it.
        let fft = CufftBackend::new_device_only(grid.nx, grid.ny, grid.nz)
            .expect("ResidentSolver: cuFFT N-grid plan creation failed");
        fft.set_stream(backend.stream_raw())
            .expect("ResidentSolver: cuFFT N-grid set_stream failed");
        let fft_padded = CufftBackend::new_device_only(pg.nx, pg.ny, pg.nz)
            .expect("ResidentSolver: cuFFT padded plan creation failed");
        fft_padded
            .set_stream(backend.stream_raw())
            .expect("ResidentSolver: cuFFT padded set_stream failed");

        // Expand the wavenumbers flat to length cplx_len in [ix, iy, iz] order,
        // and keep |k|^2 on host too (for rebuilding ETD coefficients).
        let mut fkx = Vec::with_capacity(cplx_len);
        let mut fky = Vec::with_capacity(cplx_len);
        let mut fkz = Vec::with_capacity(cplx_len);
        let mut fk2 = Vec::with_capacity(cplx_len);
        for ix in 0..snx {
            for iy in 0..sny {
                for iz in 0..snz {
                    fkx.push(ops.kx[ix]);
                    fky.push(ops.ky[iy]);
                    fkz.push(ops.kz[iz]);
                    fk2.push(ops.k_mag_sq[[ix, iy, iz]]);
                }
            }
        }
        let kx = backend.upload_real::<f64>(&fkx);
        let ky = backend.upload_real::<f64>(&fky);
        let kz = backend.upload_real::<f64>(&fkz);
        let k2 = backend.upload_real::<f64>(&fk2);

        // Upload the initial velocity state.
        let mut u_hat = [
            backend.alloc_cplx::<f64>(cplx_len),
            backend.alloc_cplx::<f64>(cplx_len),
            backend.alloc_cplx::<f64>(cplx_len),
        ];
        for c in 0..3 {
            debug_assert_eq!(u_hat_host[c].len(), cplx_len, "u_hat component length");
            backend.upload_cplx::<f64>(&u_hat_host[c], &mut u_hat[c]);
        }

        let alloc_cplx3 = |b: &CudaBackend, len: usize| {
            [
                b.alloc_cplx::<f64>(len),
                b.alloc_cplx::<f64>(len),
                b.alloc_cplx::<f64>(len),
            ]
        };
        let alloc_real3 = |b: &CudaBackend, len: usize| {
            [
                b.alloc_real::<f64>(len),
                b.alloc_real::<f64>(len),
                b.alloc_real::<f64>(len),
            ]
        };

        let omega_hat = alloc_cplx3(&backend, cplx_len);
        let pad_scratch = backend.alloc_cplx::<f64>(pcplx_len);
        let u_phys = alloc_real3(&backend, preal_len);
        let omega_phys = alloc_real3(&backend, preal_len);

        let stage_n = [
            alloc_cplx3(&backend, cplx_len),
            alloc_cplx3(&backend, cplx_len),
            alloc_cplx3(&backend, cplx_len),
            alloc_cplx3(&backend, cplx_len),
        ];
        let temp = alloc_cplx3(&backend, cplx_len);
        let cfl_scratch = backend.alloc_cplx::<f64>(cplx_len);

        // ETD coefficient buffers: allocated now, filled by ensure_etd later.
        let etd_exp_full = backend.alloc_real::<f64>(cplx_len);
        let etd_exp_half = backend.alloc_real::<f64>(cplx_len);
        let etd_a21 = backend.alloc_real::<f64>(cplx_len);
        let etd_a41 = backend.alloc_real::<f64>(cplx_len);
        let etd_b1 = backend.alloc_real::<f64>(cplx_len);
        let etd_b23 = backend.alloc_real::<f64>(cplx_len);
        let etd_b4 = backend.alloc_real::<f64>(cplx_len);

        Self {
            backend,
            grid,
            fft,
            fft_padded,
            cplx_len,
            pcplx_len,
            real_len,
            preal_len,
            sshape: (snx, sny, snz),
            pshape: (pnx, pny, pnz),
            kx,
            ky,
            kz,
            k2,
            u_hat,
            omega_hat,
            pad_scratch,
            u_phys,
            omega_phys,
            stage_n,
            temp,
            cfl_scratch,
            etd_exp_full,
            etd_exp_half,
            etd_a21,
            etd_a41,
            etd_b1,
            etd_b23,
            etd_b4,
            etd_dt: -1.0,
            nu,
            k2_host: fk2,
        }
    }

    /// Borrows the GPU backend (for the transfer counter and diagnostics).
    pub fn backend(&self) -> &CudaBackend {
        &self.backend
    }

    /// Number of spectral (half-complex) points per component on the `N` grid.
    pub fn cplx_len(&self) -> usize {
        self.cplx_len
    }

    /// Analytic peak device-memory footprint of the resident solver, in bytes,
    /// INCLUDING the cuFFT work areas.
    ///
    /// Counts the large persistent allocations, all `f64` (8 bytes):
    ///
    /// - `N`-grid complex buffers (`2 * cplx_len` f64 each): `u_hat` (3),
    ///   `omega_hat` (3), four ETD stage triples (12), `temp` (3) = 21 complex
    ///   buffers, plus the one padded complex scratch (`2 * pcplx_len` f64).
    /// - padded real buffers (`preal_len` f64 each): `u_phys` (3) and
    ///   `omega_phys` (3) = 6. The cross product writes in place over
    ///   `omega_phys`, so there is NO separate `cross_phys` triple (the packing
    ///   change that brought `N = 256` under 8 GB).
    /// - real coefficient/wavenumber buffers (`cplx_len` f64 each): `kx`, `ky`,
    ///   `kz`, `k2` and the seven ETD coefficient buffers = 11.
    /// - the two cuFFT padded-plan work areas (forward D2Z + inverse Z2D),
    ///   queried live from the plan via `cufftGetSize`. The `N`-grid plan's work
    ///   areas are also counted. cuFFT's own `d_real`/`d_complex` host-path
    ///   scratch (one padded real + one padded complex per plan) is counted too,
    ///   since each [`CufftBackend`] allocates it at construction.
    ///
    /// Used by the memory-budget test and the resident run to compare the
    /// analytic estimate against the ACTUAL `cudaMemGetInfo` peak.
    pub fn peak_memory_bytes(&self) -> usize {
        let f64_bytes = std::mem::size_of::<f64>();
        // u_hat(3) + omega_hat(3) + 4 stage triples(12) + temp(3) + cfl_scratch(1)
        let cplx_n_buffers = 22;
        let n_complex = cplx_n_buffers * 2 * self.cplx_len;
        let padded_complex = 2 * self.pcplx_len; // one padded complex scratch
        let padded_real = 6 * self.preal_len; // u_phys + omega_phys (cross in place)
        let real_coeffs = 11 * self.cplx_len; // kx, ky, kz, k2 + 7 ETD coeffs
        let buffers = (n_complex + padded_complex + padded_real + real_coeffs) * f64_bytes;

        // cuFFT work areas ONLY (device-only plans: no host-path d_real/d_complex
        // scratch, and the two plans of each backend SHARE one work area, so the
        // real cost is the max of the two plan sizes per backend, not the sum).
        let padded_work = self.fft_padded.allocated_workarea_bytes().unwrap_or(0);
        let n_work = self.fft.allocated_workarea_bytes().unwrap_or(0);

        buffers + padded_work + n_work
    }

    /// ACTUAL cuFFT work-area bytes allocated for the padded (384^3) plans (the
    /// single shared buffer of a device-only backend), for the budget report.
    pub fn padded_workarea_bytes(&self) -> usize {
        self.fft_padded.allocated_workarea_bytes().unwrap_or(0)
    }

    /// Queries `(free, total)` device memory in bytes via `cudaMemGetInfo`, so
    /// the run can record ACTUAL peak usage (free before alloc minus free now).
    pub fn mem_get_info(&self) -> (usize, usize) {
        self.backend
            .mem_get_info()
            .expect("ResidentSolver: cudaMemGetInfo failed")
    }

    /// Downloads spectral velocity component `c` into a host buffer
    /// (diagnostics; counts as one transfer).
    pub fn download_u_hat(&self, c: usize, host: &mut [Complex<f64>]) {
        debug_assert!(c < 3, "velocity component index out of range");
        self.backend.download_cplx::<f64>(&self.u_hat[c], host);
    }

    /// Downloads nonlinear stage `k` (0..4), component `c`, into a host buffer
    /// (diagnostics; counts as one transfer). Stage 0 holds the nonlinear term
    /// of the current `u_hat` after a [`Self::nonlinear`] call.
    pub fn download_stage(&self, k: usize, c: usize, host: &mut [Complex<f64>]) {
        debug_assert!(k < 4 && c < 3, "stage or component index out of range");
        self.backend.download_cplx::<f64>(&self.stage_n[k][c], host);
    }

    /// Computes the nonlinear term of the current `u_hat` into stage buffer 0,
    /// entirely on device (zero host transfers).
    ///
    /// A thin wrapper over [`Self::nonlinear_into`] for `u_hat -> stage 0`, used
    /// both by the nonlinear-only verification and as stage 1 of the step.
    pub fn nonlinear(&mut self) {
        self.nonlinear_into(NlInput::UHat, 0);
    }

    /// Computes the nonlinear term of the selected input field into stage buffer
    /// `out_stage`, entirely on device, reproducing
    /// [`crate::nonlinear::compute_nonlinear`].
    ///
    /// Steps (all on resident buffers, zero host transfers):
    ///
    /// 1. device `curl`: input `-> omega_hat` (`N` grid).
    /// 2. for each of the six fields (input `[0..3]`, `omega_hat[0..3]`):
    ///    `spectral_pad` into the padded complex scratch, `scale_cplx` by
    ///    `1 / N` (the collapsed host factor), then `c2r_3d_dev` into the
    ///    matching padded physical buffer.
    /// 3. device `cross_product` over the six padded physical fields into the
    ///    three padded cross buffers.
    /// 4. for each component: `r2c_3d_dev` (cross_phys -> padded complex
    ///    scratch), `spectral_truncate` into `stage_n[out_stage]`, then
    ///    `scale_cplx` by `N / N_padded` (`= inv_scale`).
    /// 5. device `leray` projection on `stage_n[out_stage]`.
    fn nonlinear_into(&mut self, input: NlInput, out_stage: usize) {
        let (snx, sny, snz) = self.sshape;
        let (pnx, pny, pnz) = self.pshape;
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        // Step 1: curl input -> omega_hat (N grid). The input triple, the
        // wavenumbers, the backend and omega_hat are all DISTINCT fields, so the
        // disjoint-field borrows (input/kx immutable, omega_hat mutable) are
        // accepted without any placeholder allocation.
        {
            let inp: &[CplxBuf<f64>; 3] = match input {
                NlInput::UHat => &self.u_hat,
                NlInput::Temp => &self.temp,
            };
            let [o0, o1, o2] = &mut self.omega_hat;
            self.backend
                .curl::<f64>(
                    &inp[0], &inp[1], &inp[2], &self.kx, &self.ky, &self.kz, o0, o1, o2,
                )
                .expect("ResidentSolver: curl launch failed");
        }

        // Step 2: pad, scale by 1/N, inverse-transform each of the 6 fields into
        // its padded physical buffer. The padded inverse cuFFT is unnormalised,
        // and the host's (scale then 1/N_padded) collapses to a single 1/N.
        let inv_n = 1.0 / (self.real_len as f64);
        for c in 0..3 {
            self.pad_scale_inverse(input, FieldKind::Velocity, c, inv_n);
        }
        for c in 0..3 {
            self.pad_scale_inverse(input, FieldKind::Vorticity, c, inv_n);
        }

        // Step 3: aliasing-safe pointwise cross product over the 6 padded
        // physical fields, written IN PLACE over omega_phys (cx==ox, cy==oy,
        // cz==oz). The register-first kernel latches all six inputs before any
        // write, so this is bit-identical to the disjoint cross while saving the
        // three cross_phys buffers. omega_phys is no longer needed as a vorticity
        // source after this point (the forward transform below reads the cross
        // result it now holds).
        //
        // The borrow has u_phys (immutable, 3) plus omega_phys appearing BOTH as
        // the immutable o-inputs and the mutable c-outputs, which Rust cannot
        // express as overlapping borrows of one array. We therefore raw-alias the
        // omega_phys pointers for the o-inputs while holding the &mut for the
        // c-outputs; the kernel's read-before-write contract makes that sound.
        {
            let [up0, up1, up2] = &self.u_phys;
            // SAFETY: the cross_product_inplace kernel reads all six inputs into
            // registers before writing any output, and each thread owns a unique
            // index, so reading omega_phys through these shared references while
            // the kernel writes the same buffers is sound (no read observes a
            // post-write value, no two threads write one slot). The buffers stay
            // resident in the active context for the whole synchronous launch
            // (we hold &mut self).
            let omega_ptr = &self.omega_phys as *const [RealBuf<f64>; 3];
            let [op0, op1, op2] = unsafe { &*omega_ptr };
            let [cp0, cp1, cp2] = &mut self.omega_phys;
            self.backend
                .cross_product_inplace_aliased::<f64>(up0, up1, up2, op0, op1, op2, cp0, cp1, cp2)
                .expect("ResidentSolver: cross_product_inplace launch failed");
        }

        // Step 4: forward-transform each padded cross field (now in omega_phys),
        // truncate to N into stage_n[out_stage], then scale by inv_scale.
        let inv_scale = (self.real_len as f64) / (self.preal_len as f64);
        for c in 0..3 {
            // Forward transform padded cross[c] (held in omega_phys) -> scratch.
            let d_real = self.backend.real_device_ptr(&self.omega_phys[c]);
            let d_cplx = self.backend.cplx_device_ptr(&self.pad_scratch);
            // SAFETY: pointers from resident buffers in the active context; the
            // padded plan matches (pnx, pny, pnz); cross_phys[c] is preal_len f64
            // and pad_scratch is 2 * pcplx_len f64; plan bound to the backend
            // stream; buffers outlive the call (we hold &mut self).
            unsafe {
                self.fft_padded.r2c_3d_dev(d_real, d_cplx);
            }
            // Truncate padded spectrum -> N grid into the stage buffer.
            self.backend
                .spectral_truncate::<f64>(
                    &self.pad_scratch,
                    &mut self.stage_n[out_stage][c],
                    pnx,
                    pny,
                    pnz,
                    snx,
                    sny,
                    snz,
                    nx,
                    ny,
                )
                .expect("ResidentSolver: spectral_truncate launch failed");
            // Scale by inv_scale = N / N_padded.
            self.backend
                .scale_cplx::<f64>(&mut self.stage_n[out_stage][c], inv_scale)
                .expect("ResidentSolver: truncate scale launch failed");
        }

        // Step 5: Leray projection on the stage result (enforce div-free).
        // kx/ky/kz/k2, backend and the stage triple are distinct fields.
        {
            let [s0, s1, s2] = &mut self.stage_n[out_stage];
            self.backend
                .leray_project::<f64>(&self.kx, &self.ky, &self.kz, &self.k2, s0, s1, s2)
                .expect("ResidentSolver: leray launch failed");
        }
    }

    /// Pads input field component `c` (velocity or vorticity) into the padded
    /// complex scratch, scales by `1 / N`, then inverse-transforms into the
    /// matching padded physical buffer. One component at a time (reuses the
    /// single padded complex scratch).
    fn pad_scale_inverse(&mut self, input: NlInput, kind: FieldKind, c: usize, inv_n: f64) {
        let (snx, sny, snz) = self.sshape;
        let (pnx, pny, pnz) = self.pshape;
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        // Re-zero the padded scratch before each scatter: spectral_pad only
        // writes the source points, leaving the padding region untouched, so a
        // stale previous component would otherwise leak into the padding.
        // scale_cplx by 0 is the cheapest device zero (reuses a verified kernel).
        self.backend
            .scale_cplx::<f64>(&mut self.pad_scratch, 0.0)
            .expect("ResidentSolver: pad scratch zero failed");

        // Pad the N-grid spectral source into the padded complex scratch.
        {
            let src_buf: &CplxBuf<f64> = match (kind, input) {
                (FieldKind::Velocity, NlInput::UHat) => &self.u_hat[c],
                (FieldKind::Velocity, NlInput::Temp) => &self.temp[c],
                // Vorticity always comes from omega_hat (curl of the input).
                (FieldKind::Vorticity, _) => &self.omega_hat[c],
            };
            self.backend
                .spectral_pad::<f64>(
                    src_buf,
                    &mut self.pad_scratch,
                    snx,
                    sny,
                    snz,
                    pnx,
                    pny,
                    pnz,
                    nx,
                    ny,
                )
                .expect("ResidentSolver: spectral_pad launch failed");
        }

        // Scale the padded spectrum by 1/N (collapsed host factor).
        self.backend
            .scale_cplx::<f64>(&mut self.pad_scratch, inv_n)
            .expect("ResidentSolver: pad scale launch failed");

        // Inverse transform into the matching padded physical buffer
        // (unnormalised); choose the destination by field kind.
        let d_cplx = self.backend.cplx_device_ptr(&self.pad_scratch);
        let dst_real = match kind {
            FieldKind::Velocity => self.backend.real_device_ptr(&self.u_phys[c]),
            FieldKind::Vorticity => self.backend.real_device_ptr(&self.omega_phys[c]),
        };
        // SAFETY: both pointers come from this backend's resident buffers in the
        // active primary context; the padded plan matches (pnx, pny, pnz); the
        // scratch is 2 * pcplx_len f64 and the target padded physical buffer is
        // preal_len f64, the sizes the plan expects; the plan is bound to the
        // backend stream and the buffers outlive the call (we hold &mut self).
        unsafe {
            self.fft_padded.c2r_3d_dev(d_cplx, dst_real);
        }
    }

    /// Ensures the resident ETD coefficient buffers are built for `dt`,
    /// recomputing on the host and re-uploading only when `dt` changed.
    ///
    /// The per-mode coefficients come from `EtdCoeffs::new(lambda * dt)` with
    /// `lambda = -nu * |k|^2`, the same construction as the CPU solver. They are
    /// uploaded as seven real device buffers of length `cplx_len`.
    fn ensure_etd(&mut self, dt: f64) {
        if self.etd_dt == dt {
            return;
        }
        let n = self.cplx_len;
        let mut exp_full = Vec::with_capacity(n);
        let mut exp_half = Vec::with_capacity(n);
        let mut a21 = Vec::with_capacity(n);
        let mut a41 = Vec::with_capacity(n);
        let mut b1 = Vec::with_capacity(n);
        let mut b23 = Vec::with_capacity(n);
        let mut b4 = Vec::with_capacity(n);
        for &k2 in &self.k2_host {
            let lambda = -self.nu * k2;
            let ec = EtdCoeffs::new(lambda * dt);
            exp_full.push(ec.exp_full);
            exp_half.push(ec.exp_half);
            a21.push(ec.a21);
            a41.push(ec.a41);
            b1.push(ec.b1);
            b23.push(ec.b23);
            b4.push(ec.b4);
        }
        self.etd_exp_full = self.backend.upload_real::<f64>(&exp_full);
        self.etd_exp_half = self.backend.upload_real::<f64>(&exp_half);
        self.etd_a21 = self.backend.upload_real::<f64>(&a21);
        self.etd_a41 = self.backend.upload_real::<f64>(&a41);
        self.etd_b1 = self.backend.upload_real::<f64>(&b1);
        self.etd_b23 = self.backend.upload_real::<f64>(&b23);
        self.etd_b4 = self.backend.upload_real::<f64>(&b4);
        self.etd_dt = dt;
    }

    /// One ETD-RK4 step of size `dt`, reproducing [`Periodic3D::etd_rk4_step`]
    /// entirely on device with zero host transfers (beyond a one-off ETD
    /// coefficient upload when `dt` changes).
    ///
    /// The four stages mirror the CPU algebra precisely:
    ///
    /// - stage 1: `n1 = nonlinear(u_hat)` (into stage buffer 0).
    /// - stage 2: `temp = exp_half * u + dt * a21 * n1`; `n2 = nonlinear(temp)`.
    /// - stage 3: `temp = exp_half * u + dt * a21 * n2`; `n3 = nonlinear(temp)`.
    ///   (`a31 == a21`, as in the CPU solver.)
    /// - stage 4: `temp = exp_full * u + dt * a41 * (2 n3 - n1)`;
    ///   `n4 = nonlinear(temp)`.
    /// - final: `u = exp_full * u + dt * (b1 n1 + b23 (n2 + n3) + b4 n4)`.
    ///
    /// Each `nonlinear(temp)` writes directly into the matching stage buffer; no
    /// device-to-device copy or extra `n_hat` triple is needed. The per-mode
    /// combinations use the resident `etd_stage_axpy`, `etd_stage4` and
    /// `etd_final` kernels.
    ///
    /// [`Periodic3D::etd_rk4_step`]: crate::solver::Periodic3D
    pub fn step(&mut self, dt: f64) {
        self.ensure_etd(dt);

        // Stage 1: n1 = nonlinear(u_hat) -> stage 0.
        self.nonlinear_into(NlInput::UHat, 0);

        // Stage 2: temp = exp_half * u_hat + dt * a21 * n1.
        self.stage_axpy(dt, 0);
        self.nonlinear_into(NlInput::Temp, 1);

        // Stage 3: temp = exp_half * u_hat + dt * a21 * n2.
        self.stage_axpy(dt, 1);
        self.nonlinear_into(NlInput::Temp, 2);

        // Stage 4: temp = exp_full * u_hat + dt * a41 * (2 n3 - n1).
        self.stage4_combine(dt);
        self.nonlinear_into(NlInput::Temp, 3);

        // Final: u_hat = exp_full * u_hat + dt * (b1 n1 + b23 (n2 + n3) + b4 n4).
        self.final_update(dt);
    }

    /// Stage 2/3 combination `temp = exp_half * u_hat + dt * a21 * n[stage]`,
    /// per component, on device.
    fn stage_axpy(&mut self, dt: f64, stage: usize) {
        for c in 0..3 {
            // temp, u_hat, the stage buffer, the coefficient buffers and backend
            // are distinct fields: writing temp[c] while reading u_hat[c] and
            // stage_n[stage][c] is a disjoint-field borrow (no placeholder).
            let (temp, stage_n, u_hat) = (&mut self.temp, &self.stage_n, &self.u_hat);
            self.backend
                .etd_stage_axpy::<f64>(
                    &self.etd_exp_half,
                    &self.etd_a21,
                    dt,
                    &u_hat[c],
                    &stage_n[stage][c],
                    &mut temp[c],
                )
                .expect("ResidentSolver: stage axpy failed");
        }
    }

    /// Stage 4 combination `temp = exp_full * u_hat + dt * a41 * (2 n3 - n1)`,
    /// per component, on device.
    fn stage4_combine(&mut self, dt: f64) {
        for c in 0..3 {
            let (temp, stage_n, u_hat) = (&mut self.temp, &self.stage_n, &self.u_hat);
            self.backend
                .etd_stage4::<f64>(
                    &self.etd_exp_full,
                    &self.etd_a41,
                    dt,
                    &u_hat[c],
                    &stage_n[0][c],
                    &stage_n[2][c],
                    &mut temp[c],
                )
                .expect("ResidentSolver: stage 4 combine failed");
        }
    }

    /// Final ETD update `u_hat = exp_full * u_hat + dt * (b1 n1 + b23 (n2 + n3) +
    /// b4 n4)`, per component, in place on device.
    fn final_update(&mut self, dt: f64) {
        for c in 0..3 {
            let (u_hat, stage_n) = (&mut self.u_hat, &self.stage_n);
            self.backend
                .etd_final::<f64>(
                    &self.etd_exp_full,
                    &self.etd_b1,
                    &self.etd_b23,
                    &self.etd_b4,
                    dt,
                    &stage_n[0][c],
                    &stage_n[1][c],
                    &stage_n[2][c],
                    &stage_n[3][c],
                    &mut u_hat[c],
                )
                .expect("ResidentSolver: final update failed");
        }
    }

    /// Resident CFL timestep, reproducing the CPU `compute_cfl_dt_static` scheme
    /// WITHOUT a per-step host pull of the velocity field.
    ///
    /// The CPU reference computes `dt = safety * min(dx / max||u||, dx^2 / nu,
    /// 0.1)` from the N-grid physical velocity. This does the same on device:
    /// for each component it copies `u_hat[c]` into the CFL scratch (the cuFFT
    /// C2R may overwrite its input, so the primary state must be protected),
    /// inverse-transforms the scratch into the FRONT `real_len` reals of the
    /// padded `u_phys[c]` buffer with the N-grid plan, and applies the deferred
    /// `1 / real_len` cuFFT normalisation. A device max-reduction over those
    /// `real_len` points then yields `max||u||^2` with NO host transfer of the
    /// field (only the tiny <=256-partial tail of the reduction is pulled, as in
    /// `sum_norm_sq`). The step therefore stays resident.
    ///
    /// `safety` and `nu` match the CPU solver (`cfl_safety = 0.5`).
    pub fn cfl_dt(&mut self, safety: f64, nu: f64) -> f64 {
        let inv_n = 1.0 / (self.real_len as f64);
        for c in 0..3 {
            // Snapshot u_hat[c] into the CFL scratch (C2R is destructive).
            self.backend
                .copy_cplx::<f64>(&self.u_hat[c], &mut self.cfl_scratch)
                .expect("ResidentSolver: CFL copy failed");
            // Inverse transform (N-grid plan) scratch -> front real_len of u_phys[c].
            let d_cplx = self.backend.cplx_device_ptr(&self.cfl_scratch);
            let d_real = self.backend.real_device_ptr(&self.u_phys[c]);
            // SAFETY: both pointers come from resident buffers in the active
            // context; the N-grid plan matches (nx, ny, nz); cfl_scratch is
            // 2*cplx_len f64 and u_phys[c] is preal_len >= real_len f64 (the
            // N-grid inverse writes exactly real_len reals into the front); the
            // plan is bound to the backend stream; buffers outlive the call.
            unsafe {
                self.fft.c2r_3d_dev(d_cplx, d_real);
            }
            // The cuFFT inverse is unnormalised; the deferred 1/N is folded into
            // the squared-speed result below (scaling each real by 1/N scales the
            // squared speed by 1/N^2, and a positive scale commutes with max).
        }
        // Find max||u||^2 over the unnormalised front real_len reals of u_phys,
        // then apply the (1/N)^2 normalisation to the squared speed.
        let [u0, u1, u2] = &self.u_phys;
        let max_sq_unnorm = self.backend.max_speed_sq::<f64>(u0, u1, u2, self.real_len);
        let u_max = (max_sq_unnorm * inv_n * inv_n).sqrt();

        let dx = self.grid.dx();
        let advective = if u_max > 1e-30 { dx / u_max } else { 1.0 };
        let viscous = if nu > 1e-30 {
            dx * dx / nu
        } else {
            f64::INFINITY
        };
        safety * advective.min(viscous).min(0.1)
    }

    /// Current viscosity the solver was built with.
    pub fn nu(&self) -> f64 {
        self.nu
    }

    /// Downloads all three spectral velocity components into `host[0..3]`
    /// (diagnostics; three transfers). Each slice must be `cplx_len` long.
    pub fn download_u_hat_all(&self, host: &mut [Vec<Complex<f64>>; 3]) {
        for c in 0..3 {
            self.backend
                .download_cplx::<f64>(&self.u_hat[c], &mut host[c]);
        }
    }
}

/// Which physical field a pad/inverse step targets.
#[derive(Clone, Copy)]
enum FieldKind {
    /// The velocity field (padded into `u_phys`).
    Velocity,
    /// The vorticity field (padded into `omega_phys`).
    Vorticity,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear::compute_nonlinear;
    use ndarray::Array3;
    use vonkarman_fft::NdrustfftBackend;

    /// Builds a CUDA backend on device 0, or returns `None` (printing why) when
    /// no usable GPU is present, so the test SKIPS rather than fails in CI.
    fn backend_or_skip() -> Option<CudaBackend> {
        match CudaBackend::new(0) {
            Ok(be) => Some(be),
            Err(e) => {
                eprintln!("skipping GPU resident test: CudaBackend::new(0) failed: {e}");
                None
            }
        }
    }

    /// Deterministic pseudo-random, roughly divergence-free spectral velocity
    /// field on the `N` grid: build a random field, then Leray-project it (so the
    /// nonlinear input resembles a real solver state). Returns the three
    /// components as `Array3` (C-order = `[ix, iy, iz]`).
    fn random_div_free(ops: &SpectralOps<f64>, grid: &GridSpec) -> [Array3<Complex<f64>>; 3] {
        let (snx, sny, snz) = grid.spectral_shape();
        let shape = (snx, sny, snz);
        let mut u: [Array3<Complex<f64>>; 3] = [
            Array3::zeros(shape),
            Array3::zeros(shape),
            Array3::zeros(shape),
        ];
        let mut t = 0.123_f64;
        for c in 0..3 {
            for v in u[c].iter_mut() {
                *v = Complex::new(t.sin() * 1.3 + 0.2, (t * 1.7).cos() * 0.9 - 0.1);
                t += 0.037;
            }
            // Zero the k=0 mean so the field is genuinely fluctuating.
            u[c][[0, 0, 0]] = Complex::new(0.0, 0.0);
        }
        ops.leray_project(&mut u);
        u
    }

    /// Flatten an `Array3<Complex<f64>>` (C-order) into a host `Vec`, the flat
    /// `[ix, iy, iz]` order the resident solver uploads.
    fn flat(a: &Array3<Complex<f64>>) -> Vec<Complex<f64>> {
        a.iter().copied().collect()
    }

    #[test]
    fn resident_nonlinear_matches_cpu_to_1e_12() {
        let Some(backend) = backend_or_skip() else {
            return;
        };

        let n = 32;
        let nu = 0.01;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);
        let ops = SpectralOps::<f64>::new(&grid);

        // Shared random, divergence-free input.
        let u_hat = random_div_free(&ops, &grid);
        let u_hat_flat = [flat(&u_hat[0]), flat(&u_hat[1]), flat(&u_hat[2])];

        // CPU reference (the ground truth).
        let cpu_fft = NdrustfftBackend::new(grid.nx, grid.ny, grid.nz);
        let pg = grid.padded_3half();
        let cpu_fft_padded = NdrustfftBackend::new(pg.nx, pg.ny, pg.nz);
        let (snx, sny, snz) = grid.spectral_shape();
        let shape = (snx, sny, snz);
        let mut n_cpu: [Array3<Complex<f64>>; 3] = [
            Array3::zeros(shape),
            Array3::zeros(shape),
            Array3::zeros(shape),
        ];
        compute_nonlinear(&ops, &cpu_fft, &cpu_fft_padded, &grid, &u_hat, &mut n_cpu);

        // Resident GPU solver from the SAME input. Bracket the transfer counter.
        let mut solver = ResidentSolver::new(backend, grid, nu, &u_hat_flat);
        let setup_transfers = solver.backend().transfer_count();

        // Run the resident nonlinear: it must do ZERO host<->device transfers.
        solver.nonlinear();
        let after_nl = solver.backend().transfer_count();
        assert_eq!(
            after_nl, setup_transfers,
            "resident nonlinear must do zero host<->device transfers; \
             counter went {setup_transfers} -> {after_nl}"
        );

        // Download the GPU result (stage 0) for comparison; these are the only
        // transfers after setup.
        let cplx_len = solver.cplx_len();
        let mut max_rel = 0.0_f64;
        for c in 0..3 {
            let mut host = vec![Complex::new(0.0_f64, 0.0); cplx_len];
            solver.download_stage(0, c, &mut host);
            // Compare against the CPU reference (C-order flatten).
            let cpu_flat = flat(&n_cpu[c]);
            // Relative error in the l2 sense per component, plus a per-element
            // check, to a tight 1e-12.
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for (g, r) in host.iter().zip(cpu_flat.iter()) {
                let d = (g - r).norm();
                num += d * d;
                den += r.norm_sqr();
            }
            let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
            max_rel = max_rel.max(rel);
        }
        // Exactly three downloads (one per component) since setup.
        let after_dl = solver.backend().transfer_count();
        assert_eq!(
            after_dl,
            setup_transfers + 3,
            "expected exactly 3 diagnostic downloads after the nonlinear"
        );

        eprintln!("resident nonlinear vs CPU: max relative l2 error = {max_rel:e}");
        assert!(
            max_rel < 1e-12,
            "resident nonlinear relative error {max_rel:e} exceeds 1e-12"
        );

        // Document the resident-buffer budget at the target grids (no GPU work,
        // pure arithmetic on the buffer counts; 6 padded real buffers now, since
        // the cross product is in place over omega_phys). The cuFFT work areas
        // are reported separately by the resident run via cufftGetSize.
        for &nn in &[128usize, 256] {
            let g = GridSpec::cubic(nn, 2.0 * std::f64::consts::PI);
            let (gsnx, gsny, gsnz) = g.spectral_shape();
            let pg = g.padded_3half();
            let cplx = gsnx * gsny * gsnz;
            let pcplx = pg.nx * pg.ny * (pg.nz / 2 + 1);
            let preal = pg.nx * pg.ny * pg.nz;
            let bytes =
                (22 * 2 * cplx + 2 * pcplx + 6 * preal + 11 * cplx) * std::mem::size_of::<f64>();
            eprintln!(
                "resident buffer-only estimate at N={nn}: {:.3} GiB (excl. cuFFT work areas)",
                bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
        // And the live full estimate for the constructed N=32 solver, which DOES
        // include the cuFFT work areas (cufftGetSize) and plan scratch.
        eprintln!(
            "resident full peak_memory_bytes at N={n} (incl. cuFFT): {:.4} GiB",
            solver.peak_memory_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        let (free, total) = solver.mem_get_info();
        eprintln!(
            "cudaMemGetInfo at N={n}: free {:.3} GiB / total {:.3} GiB",
            free as f64 / (1024.0 * 1024.0 * 1024.0),
            total as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    /// The aliasing-safe `cross_product_inplace` (outputs aliasing the omega
    /// inputs) must be BIT-IDENTICAL to the disjoint `cross_product` on the same
    /// data. Runs both on the GPU and asserts every f64 matches exactly.
    #[test]
    fn cross_product_inplace_bit_identical_to_disjoint() {
        let Some(backend) = backend_or_skip() else {
            return;
        };

        // Deterministic pseudo-random padded physical fields.
        let n = 6000;
        let make = |seed: f64| -> Vec<f64> {
            (0..n)
                .map(|i| (seed + 0.017 * i as f64).sin() * 1.9 - 0.4)
                .collect()
        };
        let ux = make(0.2);
        let uy = make(1.1);
        let uz = make(2.5);
        let ox = make(3.3);
        let oy = make(4.7);
        let oz = make(5.9);

        let dux = backend.upload_real::<f64>(&ux);
        let duy = backend.upload_real::<f64>(&uy);
        let duz = backend.upload_real::<f64>(&uz);

        // Disjoint reference: fresh omega buffers, fresh cross outputs.
        let dox = backend.upload_real::<f64>(&ox);
        let doy = backend.upload_real::<f64>(&oy);
        let doz = backend.upload_real::<f64>(&oz);
        let mut rcx = backend.alloc_real::<f64>(n);
        let mut rcy = backend.alloc_real::<f64>(n);
        let mut rcz = backend.alloc_real::<f64>(n);
        backend
            .cross_product::<f64>(
                &dux, &duy, &duz, &dox, &doy, &doz, &mut rcx, &mut rcy, &mut rcz,
            )
            .expect("disjoint cross launch failed");
        let ref_cx = backend.download_real::<f64>(&rcx);
        let ref_cy = backend.download_real::<f64>(&rcy);
        let ref_cz = backend.download_real::<f64>(&rcz);

        // In-place: outputs alias the omega buffers (cx==ox, cy==oy, cz==oz).
        // We must alias the same buffers for both the o-inputs and c-outputs, so
        // use raw pointers (the kernel's read-before-write makes this sound).
        let mut aox = backend.upload_real::<f64>(&ox);
        let mut aoy = backend.upload_real::<f64>(&oy);
        let mut aoz = backend.upload_real::<f64>(&oz);
        {
            // SAFETY: the kernel reads all six inputs before any write and each
            // thread owns a unique index, so reading aox/aoy/aoz through shared
            // refs while writing them through the &mut is sound.
            let i_ox = &aox as *const RealBuf<f64>;
            let i_oy = &aoy as *const RealBuf<f64>;
            let i_oz = &aoz as *const RealBuf<f64>;
            let (ri_ox, ri_oy, ri_oz) = unsafe { (&*i_ox, &*i_oy, &*i_oz) };
            backend
                .cross_product_inplace_aliased::<f64>(
                    &dux, &duy, &duz, ri_ox, ri_oy, ri_oz, &mut aox, &mut aoy, &mut aoz,
                )
                .expect("in-place cross launch failed");
        }
        let ip_cx = backend.download_real::<f64>(&aox);
        let ip_cy = backend.download_real::<f64>(&aoy);
        let ip_cz = backend.download_real::<f64>(&aoz);

        for i in 0..n {
            assert_eq!(
                ref_cx[i].to_bits(),
                ip_cx[i].to_bits(),
                "cx bit mismatch at {i}: disjoint {} vs in-place {}",
                ref_cx[i],
                ip_cx[i]
            );
            assert_eq!(
                ref_cy[i].to_bits(),
                ip_cy[i].to_bits(),
                "cy bit mismatch at {i}"
            );
            assert_eq!(
                ref_cz[i].to_bits(),
                ip_cz[i].to_bits(),
                "cz bit mismatch at {i}"
            );
        }
        eprintln!("cross_product_inplace is bit-identical to disjoint over {n} points");
    }

    #[test]
    fn resident_etd_rk4_step_matches_cpu_to_1e_12() {
        use crate::Periodic3D;
        use crate::ic::IcType;
        use vonkarman_core::domain::Domain;
        use vonkarman_fft::BackendMode;

        let Some(backend) = backend_or_skip() else {
            return;
        };

        let n = 32;
        let nu = 0.01;
        let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);

        // CPU reference solver from the Taylor-Green IC. Its construction sets dt
        // from CFL and builds the matching ETD coefficients; the first
        // Domain::step keeps that dt (the adaptive check sees no change on step
        // 0) and performs exactly one etd_rk4_step. We read u_hat and dt at
        // construction, build the resident solver from the SAME state and dt,
        // then step both once and compare.
        let mut cpu = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
        let dt = cpu.dt();

        // Snapshot the construction-time spectral velocity (C-order flatten) for
        // the resident solver's initial state.
        let u0 = cpu.u_hat();
        let u_hat_flat = [flat(&u0[0]), flat(&u0[1]), flat(&u0[2])];

        let mut solver = ResidentSolver::new(backend, grid, nu, &u_hat_flat);

        // One CPU step (single ETD-RK4 step at the construction dt).
        cpu.step();
        // Confirm the CPU kept the construction dt (no adaptive recompute on the
        // first step), so the GPU step at `dt` is the matching single step.
        assert_eq!(cpu.dt(), dt, "CPU adaptively changed dt on step 0");

        // One resident GPU step at the SAME dt. Bracket the transfer counter: the
        // ETD coefficient upload happens inside (7 buffers) on the first dt, and
        // the step itself must do no other transfers.
        let before = solver.backend().transfer_count();
        solver.step(dt);
        let after_step = solver.backend().transfer_count();
        // The only transfers during step() are the 7 ETD coefficient uploads
        // (first dt). The physics (curl/pad/scale/cuFFT/cross/truncate/leray/etd)
        // moves nothing across the bus.
        assert_eq!(
            after_step - before,
            7,
            "resident step did unexpected transfers: {before} -> {after_step} \
             (expected exactly the 7 one-off ETD coefficient uploads)"
        );

        // Compare the stepped GPU u_hat against the CPU reference to 1e-12.
        let cpu_stepped = cpu.u_hat();
        let cplx_len = solver.cplx_len();
        let mut max_rel = 0.0_f64;
        for c in 0..3 {
            let mut host = vec![Complex::new(0.0_f64, 0.0); cplx_len];
            solver.download_u_hat(c, &mut host);
            let cpu_flat = flat(&cpu_stepped[c]);
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for (g, r) in host.iter().zip(cpu_flat.iter()) {
                let d = (g - r).norm();
                num += d * d;
                den += r.norm_sqr();
            }
            let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
            max_rel = max_rel.max(rel);
        }
        // The diagnostic downloads (3) are the only further transfers.
        let after_dl = solver.backend().transfer_count();
        assert_eq!(
            after_dl - after_step,
            3,
            "expected exactly 3 diagnostic downloads after the step"
        );

        eprintln!("resident ETD-RK4 step vs CPU: max relative l2 error = {max_rel:e}");
        assert!(
            max_rel < 1e-12,
            "resident ETD-RK4 step relative error {max_rel:e} exceeds 1e-12"
        );

        // Re-stepping at the SAME dt must NOT re-upload the ETD coefficients
        // (cached), so a second step does zero transfers of its own.
        let before2 = solver.backend().transfer_count();
        solver.step(dt);
        let after2 = solver.backend().transfer_count();
        assert_eq!(
            after2, before2,
            "a second step at the same dt must reuse cached ETD coefficients \
             (zero transfers); counter went {before2} -> {after2}"
        );
    }

    /// Wall-clock timing harness for the resident ETD-RK4 step (ignored by
    /// default; run with `--ignored --nocapture`).
    ///
    /// Constructs the resident solver from a Taylor-Green state at N = 128 and
    /// N = 256, times a fixed number of resident `step` calls (after a short
    /// warm-up), and reports steps per second at each size. This is the
    /// before/after measurement for the stream-ordered launch change: the
    /// numbers are reported exactly as measured.
    #[test]
    #[ignore = "GPU timing harness; run explicitly with --ignored --nocapture"]
    fn resident_step_timing() {
        use crate::Periodic3D;
        use crate::ic::IcType;
        use vonkarman_core::domain::Domain;
        use vonkarman_fft::BackendMode;

        for &(n, steps) in &[(128usize, 50usize), (256usize, 20usize)] {
            let Some(backend) = backend_or_skip() else {
                return;
            };

            let nu = 0.01;
            let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);

            // Take an initial Taylor-Green spectral state and its CFL dt from a
            // CPU solver, then build the resident solver from the same state.
            let cpu = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
            let dt = cpu.dt();
            let u0 = cpu.u_hat();
            let u_hat_flat = [flat(&u0[0]), flat(&u0[1]), flat(&u0[2])];
            let mut solver = ResidentSolver::new(backend, grid, nu, &u_hat_flat);

            // Warm-up (also triggers the one-off ETD coefficient upload at this
            // dt) so the timed loop measures only steady-state stepping.
            let warmup = 5;
            for _ in 0..warmup {
                solver.step(dt);
            }

            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                solver.step(dt);
            }
            let elapsed = t0.elapsed();
            let secs = elapsed.as_secs_f64();
            let per_step_ms = 1.0e3 * secs / steps as f64;
            let steps_per_s = steps as f64 / secs;
            eprintln!(
                "resident step timing N={n}: {steps} steps in {secs:.4} s \
                 => {per_step_ms:.3} ms/step, {steps_per_s:.3} steps/s"
            );
        }
    }

    /// Three-way per-step wall-clock comparison with repeats (ignored by
    /// default; run with `--ignored --nocapture`).
    ///
    /// For each resolution and each backend (CPU, non-resident cuFFT with a host
    /// bounce per transform, and the device-resident solver), times an ETD-RK4
    /// step over several repeats after a warm-up, and reports mean, sample
    /// standard deviation, and minimum ms/step plus the resident speedups. The
    /// three solvers are built in separate scopes so their device allocations
    /// never co-reside (the resident solver alone uses about 6.4 GiB at N=256).
    /// Each repeat and the summary are printed in a `TIMING,` prefixed CSV line
    /// so the numbers can be collected verbatim. The CPU and cuFFT solvers use
    /// adaptive-dt stepping (a few extra inverse transforms for the CFL bound)
    /// while the resident solver is driven at a fixed dt; the dominant cost (the
    /// 36 padded transforms per step) is shared by all three.
    #[test]
    #[ignore = "GPU 3-way timing harness; run with --ignored --nocapture"]
    fn three_way_step_timing() {
        use crate::Periodic3D;
        use crate::ic::IcType;
        use vonkarman_core::domain::Domain;
        use vonkarman_fft::BackendMode;

        // Mean, sample standard deviation, and minimum of a set of timings.
        fn stats(v: &[f64]) -> (f64, f64, f64) {
            let n = v.len() as f64;
            let mean = v.iter().sum::<f64>() / n;
            let var = if v.len() > 1 {
                v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
            } else {
                0.0
            };
            let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
            (mean, var.sqrt(), min)
        }

        // Time `steps` per repeat over `reps` repeats on one Domain, after a
        // warm-up. Returns ms/step for each repeat.
        fn loops_domain<D: Domain<f64>>(
            mut d: D,
            warmup: usize,
            steps: usize,
            reps: usize,
        ) -> Vec<f64> {
            for _ in 0..warmup {
                d.step();
            }
            (0..reps)
                .map(|_| {
                    let t0 = std::time::Instant::now();
                    for _ in 0..steps {
                        d.step();
                    }
                    1.0e3 * t0.elapsed().as_secs_f64() / steps as f64
                })
                .collect()
        }

        // (N, gpu_steps, gpu_reps, base_steps, base_reps): the CPU and cuFFT
        // baselines are tens of seconds per step at large N, so they take fewer
        // steps and repeats than the resident path (the headline number).
        let configs = [
            (64usize, 50usize, 5usize, 20usize, 3usize),
            (128, 30, 5, 5, 3),
            (256, 20, 5, 2, 3),
        ];
        for &(n, gpu_steps, gpu_reps, base_steps, base_reps) in &configs {
            let Some(backend) = backend_or_skip() else {
                return;
            };
            let nu = 0.01;
            let grid = GridSpec::cubic(n, 2.0 * std::f64::consts::PI);

            let base = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
            let dt = base.dt();
            let u0 = base.u_hat();
            let u_hat_flat = [flat(&u0[0]), flat(&u0[1]), flat(&u0[2])];
            drop(base);

            let cpu = {
                let d = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cpu);
                loops_domain(d, 2, base_steps, base_reps)
            };
            let cufft = {
                let d = Periodic3D::new(grid, nu, IcType::TaylorGreen, BackendMode::Cufft);
                loops_domain(d, 2, base_steps, base_reps)
            };
            let resident = {
                let mut solver = ResidentSolver::new(backend, grid, nu, &u_hat_flat);
                for _ in 0..5 {
                    solver.step(dt);
                }
                (0..gpu_reps)
                    .map(|_| {
                        let t0 = std::time::Instant::now();
                        for _ in 0..gpu_steps {
                            solver.step(dt);
                        }
                        1.0e3 * t0.elapsed().as_secs_f64() / gpu_steps as f64
                    })
                    .collect::<Vec<_>>()
            };

            for (mode, v) in [
                ("cpu", &cpu),
                ("cufft_nonresident", &cufft),
                ("resident", &resident),
            ] {
                for (r, ms) in v.iter().enumerate() {
                    eprintln!("TIMING,{n},{mode},{r},{ms:.4}");
                }
            }

            let (cpu_m, cpu_s, cpu_min) = stats(&cpu);
            let (cuf_m, cuf_s, cuf_min) = stats(&cufft);
            let (res_m, res_s, res_min) = stats(&resident);
            eprintln!(
                "SUMMARY N={n}: CPU {cpu_m:.2}+/-{cpu_s:.2} (min {cpu_min:.2}) | \
                 cuFFT-nonresident {cuf_m:.2}+/-{cuf_s:.2} (min {cuf_min:.2}) | \
                 resident {res_m:.2}+/-{res_s:.2} (min {res_min:.2}) ms/step | \
                 resident-vs-CPU {:.2}x | resident-vs-cuFFT {:.2}x",
                cpu_m / res_m,
                cuf_m / res_m
            );
        }
    }
}
