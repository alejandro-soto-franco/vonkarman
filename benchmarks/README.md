# vonkarman Cross-Solver Benchmarks

Compares vonkarman against four reference DNS solvers on the Taylor-Green vortex at Re=1600.

## Reference Solvers

| Solver | Language | Source |
|--------|----------|--------|
| hit3d | Fortran 90 + MPI | cpraveen/hit3d |
| spectralDNS | Python | spectralDNS/spectralDNS |
| Dedalus | Python | DedalusProject/dedalus |
| TurboGenPY | Python | saadgroup/TurboGenPY (IC spectrum comparison only) |

## Usage

```bash
pip install -r requirements.txt
python run_all.py build    # build/install all solvers
python run_all.py run      # run canonical test case
python run_all.py compare  # numerical comparison
python run_all.py report   # generate figures + markdown report
python run_all.py all      # all of the above
```

## Canonical Test Case

Taylor-Green vortex, Re=1600 (nu=6.25e-4), N=128, domain [0, 2*pi]^3, t in [0, 10].
All solvers use 3/2 dealiasing where applicable.

## Reference Data

- `reference_data/brachet1983.csv`: digitised enstrophy from Brachet et al. (1983) JFM 130
- `reference_data/vanrees2011.csv`: digitised dissipation rate from van Rees et al. (2011) JCP 230
