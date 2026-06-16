"""Generate the pareto-format timing CSV from the in-process 3-way harness.

The numbers below are the per-repeat ms/step measured by the
`three_way_step_timing` test in vonkarman-periodic/src/resident.rs, run in
release mode on the RTX 5060 (warm, fixed dt for the resident path, adaptive dt
for the CPU and non-resident cuFFT baselines). They are reproduced here verbatim
so the timing CSV can be regenerated without a rerun. One row per repeat, so the
variance is visible at plot time.
"""

import csv
import os

RESULTS = os.path.join(os.path.dirname(__file__), "results")

# resolution -> backend -> [ms/step per repeat]
DATA = {
    64: {
        "cpu": [490.7972, 477.2341, 481.3004],
        "cufft_nonresident": [331.3747, 320.4686, 322.2507],
        "resident_gpu": [17.2559, 17.2721, 17.2690, 17.2665, 17.2644],
    },
    128: {
        "cpu": [5455.5498, 6184.2107, 5463.9681],
        "cufft_nonresident": [2656.5719, 3335.1632, 2628.5041],
        "resident_gpu": [163.7838, 164.0683, 164.2889, 164.4473, 164.5053],
    },
    256: {
        "cpu": [43907.4234, 43871.1236, 43773.6105],
        "cufft_nonresident": [21286.7985, 21378.2110, 21419.3077],
        "resident_gpu": [1495.9570, 1501.4815, 1504.9452, 1509.3174, 1510.6184],
    },
}

NOTES = "per-step, warm, release, RTX 5060, f64"

out_path = os.path.join(RESULTS, "timing_vonkarman_3way.csv")
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "solver",
            "backend",
            "resolution",
            "repeat_index",
            "walltime_s",
            "steps",
            "steps_per_s",
            "end_time",
            "notes",
        ]
    )
    for n, backends in sorted(DATA.items()):
        for backend, reps in backends.items():
            for r, ms in enumerate(reps):
                walltime_s = ms / 1000.0
                steps_per_s = 1000.0 / ms
                w.writerow(
                    [
                        "vonkarman",
                        backend,
                        n,
                        r,
                        f"{walltime_s:.6f}",
                        1,
                        f"{steps_per_s:.4f}",
                        0.0,
                        NOTES,
                    ]
                )

print(f"wrote {out_path}")
