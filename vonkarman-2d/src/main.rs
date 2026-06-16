//! Co-rotating vortex merger with gold + rust-red passive dye, rendered to a
//! PNG sequence. Two same-sign vortices orbit, spiral in, and merge, winding
//! the two dyes into a double spiral.
//!
//!   vonkarman-2d [n] [steps] [stride] [outdir]
//!   # then: ffmpeg -framerate 30 -i outdir/f_%05d.png -pix_fmt yuv420p out.mp4

use std::time::Instant;
use vonkarman_2d::{Sim, Spectral2D, co_rotating_vortices, write_dye_png};

fn arg<T: std::str::FromStr>(i: usize, default: T) -> T {
    std::env::args().nth(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let n: usize = arg(1, 512);
    let steps: usize = arg(2, 9000);
    let stride: usize = arg(3, 45);
    let out: String = arg(4, "frames2d".to_string());
    std::fs::create_dir_all(&out).expect("create outdir");

    // Flow + numerics. dt is advective-CFL safe; viscosity is exact (IF-RK4).
    let dt = 1.5e-3 * (512.0 / n as f64); // finer grid -> smaller step
    let nu = 8.0e-4;
    let kappa = 6.0e-4;
    let (sep, core, circ) = (1.5, 0.5, 10.0);

    let spec = Spectral2D::new(n);
    let (wh, ch) = co_rotating_vortices(&spec, sep, core, circ);
    let mut sim = Sim::new(spec, wh, ch, dt, nu, kappa);

    println!("vonkarman-2d: n={n} steps={steps} stride={stride} dt={dt:.2e} -> {out}/");
    let t0 = Instant::now();
    let mut frame = 0;
    for step in 0..=steps {
        if step % stride == 0 {
            write_dye_png(&sim.dye(), &format!("{out}/f_{frame:05}.png"));
            frame += 1;
            if frame % 20 == 0 {
                println!("  frame {frame} (step {step}, {:.0?})", t0.elapsed());
            }
        }
        if step < steps {
            sim.step();
        }
    }
    println!("done: {frame} frames in {:.0?}", t0.elapsed());
}
