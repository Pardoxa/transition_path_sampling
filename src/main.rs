pub mod particle;
pub mod transition_path_state;

use std::io::BufWriter;

use particle::*;

fn main() {
    // for i in 1..1000{
    //     let distance = 1.11 +  i as f64 / 10000.0;
    //     let ensemble = Ensemble::new(distance);
    //     let energy = ensemble.hamiltonian();
    //     println!("{distance} {energy}")
    // }

    // let mut ensemble = Ensemble::new(1.1185 + 0.5);
    // let file = std::fs::File::create("test.dat").unwrap();
    // let mut buf = BufWriter::new(file);
    // for i in 0..1000000 {
    //     ensemble.velocity_verlet_step_by(0.001);
    //     if i % 100 == 0 {
    //         ensemble.write(&mut buf);
    //     }
    // }

    let ensemble = Ensemble::minimum_fig3();
    let energy = ensemble.hamiltonian();
    println!("{energy}");

    let file = std::fs::File::create("test.dat").unwrap();
    let buf = BufWriter::new(file);
    ensemble.write_positions(buf).unwrap();
}
