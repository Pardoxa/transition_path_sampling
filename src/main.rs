pub mod particle;
pub mod transition_path_state;

use std::io::BufWriter;
use std::num::NonZeroUsize;

use particle::*;
use transition_path_state::*;

fn main() {
    let region_a = TargetRegion {
        potential_energy: Ensemble::new_groundstate().potential_energy(),
        allowed_deviation: 0.01,
    };
    let region_b = TargetRegion {
        potential_energy: Ensemble::minimum_fig3().potential_energy(),
        allowed_deviation: 0.01,
    };
    let tuning_config = TransitionTuningConfig::builder()
        .region_b(region_b)
        .number_of_steps(NonZeroUsize::new(20_000).expect("non-zero steps"))
        .scale_search_iterations(NonZeroUsize::new(30).expect("non-zero iterations"))
        .impulse_sweeps(NonZeroUsize::new(2).expect("non-zero sweeps"))
        .build()
        .expect("could not build transition tuning config");

    let scale_tuning = tuning_config.tune_fig3_kick_scale_for_region_b();

    println!(
        "best kick scale = {}, best terminal distance to region B = {}",
        scale_tuning.best_scale, scale_tuning.best_distance
    );

    let impulse_sweep_tuning =
        tuning_config.tune_particle_impulses_by_sweeps_for_region_b(scale_tuning.best_start);

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
