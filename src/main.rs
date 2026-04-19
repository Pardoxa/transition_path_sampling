pub mod particle;
pub mod transition_path_state;

use std::io::BufWriter;

use particle::*;
use transition_path_state::*;

fn main() {
    let ensemble = Ensemble::minimum_fig3();
    let energy = ensemble.hamiltonian();
    println!("{energy}");
    let file = std::fs::File::create("test.gp").unwrap();
    let buf = BufWriter::new(file);
    ensemble.write_gnuplot_positions(buf).unwrap();

    // let mut ensemble = Ensemble::new(1.1185 + 0.5);
    // let file = std::fs::File::create("test.dat").unwrap();
    // let mut buf = BufWriter::new(file);
    // for i in 0..1000000 {
    //     ensemble.velocity_verlet_step_by(0.001);
    //     if i % 100 == 0 {
    //         ensemble.write(&mut buf);
    //     }
    // }

    let region_a = TargetRegion {
        potential_energy: -12.53,
        allowed_deviation: 0.01,
    };
    let region_b = TargetRegion {
        potential_energy: -11.47,
        allowed_deviation: 0.01,
    };
    let scale_search_iterations = 30;
    let scale_tuning = tune_fig3_kick_scale_for_region_b(
        region_b,
        0.01,
        20_000,
        0.2,
        0.05,
        scale_search_iterations,
    )
    .expect("could not tune figure-3 kick scale");

    println!(
        "best kick scale = {}, best terminal distance to region B = {}",
        scale_tuning.best_scale, scale_tuning.best_distance
    );

    let impulse_sweeps = 2;
    let impulse_sweep_tuning = tune_particle_impulses_by_sweeps_for_region_b(
        scale_tuning.best_start,
        region_b,
        0.01,
        20_000,
        impulse_sweeps,
        0.08,
        0.10,
        2,
    )
    .expect("could not tune per-particle impulses");

    println!(
        "best post-sweep distance to region B = {} after {} sweeps",
        impulse_sweep_tuning.best_distance, impulse_sweep_tuning.sweeps_completed
    );

    let ensemble = impulse_sweep_tuning.best_start;
    let transition_state =
        TransitionPathState::try_init(ensemble, region_a, region_b, 0.01, 20_000, 48, 25)
            .expect("could not initialize transition path state");

    println!("start H = {}", transition_state.ensemble.hamiltonian());
    println!("start V = {}", transition_state.ensemble.potential_energy());

    let file = std::fs::File::create("test.dat").unwrap();
    let buf = BufWriter::new(file);
    transition_state.ensemble.write_positions(buf).unwrap();
}
