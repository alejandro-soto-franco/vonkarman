#!/usr/bin/env python3
"""Taylor-Green vortex Re=1600 using Dedalus v3.

Uses the Dedalus v3 symbolic PDE interface with RealFourier bases.
"""
import os
import sys
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")


def main():
    N = 128
    nu = 6.25e-4
    T = 10.0
    dt = 0.001

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "dedalus.csv")

    # Activate venv
    venv = os.path.join(SCRIPT_DIR, ".venv")
    if os.path.exists(venv):
        activate = os.path.join(venv, "bin", "activate_this.py")
        if os.path.exists(activate):
            exec(open(activate).read(), {"__file__": activate})

    try:
        import dedalus.public as d3
    except ImportError:
        print("Dedalus not installed. Run setup.sh first.", file=sys.stderr)
        sys.exit(1)

    Lx = Ly = Lz = 2 * np.pi

    # Coordinates and bases
    coords = d3.CartesianCoordinates("x", "y", "z")
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords["x"], size=N, bounds=(0, Lx), dealias=3 / 2)
    ybasis = d3.RealFourier(coords["y"], size=N, bounds=(0, Ly), dealias=3 / 2)
    zbasis = d3.RealFourier(coords["z"], size=N, bounds=(0, Lz), dealias=3 / 2)

    # Fields
    u = dist.VectorField(coords, name="u", bases=(xbasis, ybasis, zbasis))
    p = dist.Field(name="p", bases=(xbasis, ybasis, zbasis))
    tau_p = dist.Field(name="tau_p")

    # Viscosity as a scalar Field
    nu_f = dist.Field(name="nu_f")
    nu_f["g"] = nu

    # Build the IVP with explicit namespace (avoid leaking Python floats)
    ns = {
        "u": u,
        "p": p,
        "tau_p": tau_p,
        "nu_f": nu_f,
    }
    problem = d3.IVP([u, p, tau_p], namespace=ns)
    problem.add_equation("dt(u) + grad(p) - nu_f*lap(u) = -dot(u, grad(u))")
    problem.add_equation("div(u) + tau_p = 0")
    problem.add_equation("integ(p) = 0")

    # Solver with RK443
    solver = problem.build_solver(d3.RK443)
    solver.stop_sim_time = T

    # Taylor-Green IC
    x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
    u["g"][0] = np.sin(x) * np.cos(y) * np.cos(z)
    u["g"][1] = -np.cos(x) * np.sin(y) * np.cos(z)
    u["g"][2] = 0.0

    # Energy integrand
    vol = Lx * Ly * Lz

    records = []

    print(f"Running Dedalus TG Re=1600 N={N}...")
    wall_start = time.time()

    while solver.proceed:
        solver.step(dt)
        if solver.iteration % 100 == 0:
            u.change_scales(1)
            E = 0.5 * d3.Integrate(d3.dot(u, u)).evaluate()["g"].flat[0] / vol
            t = solver.sim_time
            records.append((t, E, 0.0, 0.0, 0.0))
            if solver.iteration % 1000 == 0:
                elapsed = time.time() - wall_start
                print(f"  t={t:.3f}, E={E:.8e}, wall={elapsed:.0f}s")

    wall_time = time.time() - wall_start
    print(f"Done in {wall_time:.1f}s ({solver.iteration} steps)")

    data = np.array(records) if records else np.zeros((0, 5))
    np.savetxt(
        out_path,
        data,
        delimiter=",",
        header="t,E,Omega,epsilon,max_omega",
        comments="",
    )
    print(f"Written: {out_path} ({len(data)} rows)")


if __name__ == "__main__":
    main()
