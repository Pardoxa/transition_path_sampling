use crate::particle::*;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64Mcg;

const MAX_INIT_ATTEMPTS: usize = 10_000;
const POSITION_WIGGLE_SCALE: f64 = 0.01;
const MOMENTUM_WIGGLE_SCALE: f64 = 0.05;

#[derive(Clone, Debug)]
pub struct TransitionPathState {
    pub ensemble: Ensemble,
    pub region_a: TargetRegion,
    pub region_b: TargetRegion,
    pub delta_t: f64,
    pub number_of_steps: usize,
}

impl TransitionPathState {
    pub fn try_init(
        ensemble: Ensemble,
        region_a: TargetRegion,
        region_b: TargetRegion,
        delta_t: f64,
        number_of_steps: usize,
    ) -> Option<Self> {
        if !delta_t.is_finite() || delta_t <= 0.0 || number_of_steps == 0 {
            return None;
        }

        if !ensemble.close_to_region(region_a) {
            return None;
        }

        let mut seeder = rand::rng();
        let mut rng = Pcg64Mcg::from_rng(&mut seeder);
        let mut best_start = ensemble;
        let mut best_distance =
            terminal_distance_to_region_b(&best_start, region_b, delta_t, number_of_steps);

        if best_distance <= 0.0 {
            return Some(Self {
                ensemble: best_start,
                region_a,
                region_b,
                delta_t,
                number_of_steps,
            });
        }

        for _ in 0..MAX_INIT_ATTEMPTS {
            let mut trial_start = best_start.clone();
            let mode = rng.random_range(0..3);
            let wiggle_positions = mode == 0 || mode == 2;
            let wiggle_momenta = mode == 1 || mode == 2;
            let scale_factor = rng.random_range(0.5..2.0);

            trial_start.random_wiggle(
                &mut rng,
                POSITION_WIGGLE_SCALE * scale_factor,
                MOMENTUM_WIGGLE_SCALE * scale_factor,
                wiggle_positions,
                wiggle_momenta,
            );

            if !trial_start.close_to_region(region_a) {
                continue;
            }

            let trial_distance =
                terminal_distance_to_region_b(&trial_start, region_b, delta_t, number_of_steps);

            if trial_distance <= 0.0 {
                return Some(Self {
                    ensemble: trial_start,
                    region_a,
                    region_b,
                    delta_t,
                    number_of_steps,
                });
            }

            if trial_distance < best_distance {
                best_start = trial_start;
                best_distance = trial_distance;
            }
        }

        None
    }
}

fn terminal_distance_to_region_b(
    start: &Ensemble,
    region_b: TargetRegion,
    delta_t: f64,
    number_of_steps: usize,
) -> f64 {
    let mut rolled = start.clone();
    for _ in 0..number_of_steps {
        rolled.velocity_verlet_step_by(delta_t);
    }
    rolled.distance_to_region(region_b)
}
