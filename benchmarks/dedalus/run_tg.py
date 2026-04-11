#!/usr/bin/env python3
"""Taylor-Green vortex Re=1600 using Dedalus v3.

Dedalus has no built-in TG example, so we write the IVP from scratch
using three Fourier bases on [0, 2*pi]^3.
"""
import os
import sys
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(SCRIPT_DIR, ".venv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")


def main():
    N = 128
    nu = 6.25e-4
    T = 10.0
    dt = 0.001

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "dedalus.csv")

    # Activate venv
    if os.path.exists(VENV):
        activate = os.path.join(VENV, "bin", "activate_this.py")
        if os.path.exists(activate):
            exec(open(activate).read(), {"__file__": activate})

    try:
        import dedalus.public as d3
    except ImportError:
        print("Dedalus not installed. Run setup.sh first.", file=sys.stderr)
        sys.exit(1)

    # Domain
    Lx = Ly = Lz = 2 * np.pi
    coords = d3.CartesianCoordinates("x", "y", "z")
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords["x"], size=N, bounds=(0, Lx), dealias=3 / 2)
    ybasis = d3.RealFourier(coords["y"], size=N, bounds=(0, Ly), dealias=3 / 2)
    zbasis = d3.RealFourier(coords["z"], size=N, bounds=(0, Lz), dealias=3 / 2)

    # Fields
    u = dist.VectorField(coords, name="u", bases=(xbasis, ybasis, zbasis))
    p = dist.Field(name="p", bases=(xbasis, ybasis, zbasis))
    tau_p = dist.Field(name="tau_p")

    # Problem: incompressible NS
    problem = d3.IVP([u, p, tau_p], namespace=locals())
    problem.add_equation("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)")
    problem.add_equation("div(u) + tau_p = 0")
    problem.add_equation("integ(p) = 0")

    # Solver
    solver = problem.build_solver(d3.RK443)
    solver.stop_sim_time = T

    # Taylor-Green IC
    x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
    u["g"][0] = np.sin(x) * np.cos(y) * np.cos(z)
    u["g"][1] = -np.cos(x) * np.sin(y) * np.cos(z)
    u["g"][2] = 0.0

    # Diagnostics
    records = []

    print(f"Running Dedalus TG Re=1600 N={N}...")
    wall_start = time.time()

    while solver.proceed:
        solver.step(dt)
        if solver.iteration % 100 == 0:
            E = 0.5 * d3.Integrate(u @ u).evaluate()["g"].flat[0] / (Lx * Ly * Lz)
            t = solver.sim_time
            records.append((t, E, 0.0, 0.0, 0.0))
            if solver.iteration % 1000 == 0:
                print(f"  t={t:.3f}, E={E:.8e}")

    wall_time = time.time() - wall_start
    print(f"Done in {wall_time:.1f}s")

    data = np.array(records)
    np.savetxt(
        out_path,
        data,
        delimiter=",",
        header="t,E,Omega,epsilon,max_omega",
        comments="",
    )
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
