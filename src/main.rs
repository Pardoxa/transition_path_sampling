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
    ensemble.write_gnuplot_positions(buf);

    // let mut ensemble = Ensemble::new(1.1185 + 0.5);
    // let file = std::fs::File::create("test.dat").unwrap();
    // let mut buf = BufWriter::new(file);
    // for i in 0..1000000 {
    //     ensemble.velocity_verlet_step_by(0.001);
    //     if i % 100 == 0 {
    //         ensemble.write(&mut buf);
    //     }
    // }

    // let ensemble = Ensemble::new_groundstate();
    // let region_a = TargetRegion {
    //     potential_energy: -12.53,
    //     allowed_deviation: 0.3,
    // };
    // let region_b = TargetRegion {
    //     potential_energy: -11.5,
    //     allowed_deviation: 0.3,
    // };
    // let transition_state =
    //     TransitionPathState::try_init(ensemble, region_a, region_b, 0.001, 10_000, 48)
    //         .expect("could not initialize transition path state");

    // println!("start H = {}", transition_state.ensemble.hamiltonian());
    // println!("start V = {}", transition_state.ensemble.potential_energy());

    // let file = std::fs::File::create("test.dat").unwrap();
    // let buf = BufWriter::new(file);
    // transition_state.ensemble.write_positions(buf).unwrap();
}
