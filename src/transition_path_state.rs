use crate::particle::*;
use rand::RngExt;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

const MAX_INIT_ATTEMPTS: usize = 10_000;
const RNG_SEED: u128 = 0x7A6D_1C4B_92EF_0042_1B3E_9D17_55AA_3101;
const POSITION_WIGGLE_SCALE: f64 = 0.03;
const MOMENTUM_WIGGLE_SCALE: f64 = 0.03;

#[derive(Clone, Debug)]
pub struct TransitionPathState {
    pub ensemble: Ensemble,
    pub region_a: TargetRegion,
    pub region_b: TargetRegion,
    pub delta_t: f64,
    pub number_of_steps: usize,
}

struct SearchOutcome {
    restart: usize,
    best_start: Ensemble,
    best_distance: f64,
    success_start: Option<Ensemble>,
    attempts_used: usize,
}

impl TransitionPathState {
    pub fn try_init(
        ensemble: Ensemble,
        mut region_a: TargetRegion,
        mut region_b: TargetRegion,
        delta_t: f64,
        number_of_steps: usize,
        number_of_restarts: usize,
        recursion: u16,
    ) -> Option<Self> {
        let factor = 1.2f64.powi(recursion as i32);
        let old_region_a_allowed_deviation = region_a.allowed_deviation;
        let old_region_b_allowed_deviation = region_b.allowed_deviation;
        //region_a.allowed_deviation *=  factor;
        region_b.allowed_deviation *= factor;
        dbg!(recursion);
        dbg!(region_a);
        dbg!(region_b);

        if !delta_t.is_finite() || delta_t <= 0.0 || number_of_steps == 0 || number_of_restarts == 0
        {
            return None;
        }

        if !ensemble.close_to_region(region_a) {
            return None;
        }

        let initial_start = ensemble;
        let mut global_best_start = initial_start.clone();
        let mut global_best_distance =
            terminal_distance_to_region_b(&global_best_start, region_b, delta_t, number_of_steps);

        println!("try_init: using fixed RNG seed {RNG_SEED}");
        println!(
            "try_init: initial best distance to region B = {}",
            global_best_distance
        );

        if global_best_distance <= 0.0 {
            region_a.allowed_deviation = old_region_a_allowed_deviation;
            region_b.allowed_deviation = old_region_b_allowed_deviation;
            println!("try_init: initial state already reaches region B");
            if recursion == 0 {
                return Some(Self {
                    ensemble: initial_start,
                    region_a,
                    region_b,
                    delta_t,
                    number_of_steps,
                });
            } else {
                return TransitionPathState::try_init(
                    global_best_start,
                    region_a,
                    region_b,
                    delta_t,
                    number_of_steps,
                    number_of_restarts,
                    recursion - 1,
                );
            }
        }

        let num_cpu_threads = std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1);
        println!(
            "try_init: running {} restarts in batches of {}",
            number_of_restarts, num_cpu_threads
        );

        let restart_indices: Vec<usize> = (1..=number_of_restarts).collect();
        for batch in restart_indices.chunks(num_cpu_threads) {
            let batch_first = *batch.first().expect("non-empty batch expected");
            let batch_last = *batch.last().expect("non-empty batch expected");
            println!("try_init: running batch {}..={}", batch_first, batch_last);

            let mut outcomes: Vec<_> = batch
                .par_iter()
                .map(|&restart| {
                    greedy_search_from_start(
                        initial_start.clone(),
                        region_a,
                        region_b,
                        delta_t,
                        number_of_steps,
                        restart,
                    )
                })
                .collect();

            outcomes.sort_by_key(|outcome| outcome.restart);

            for outcome in &outcomes {
                println!(
                    "try_init: restart {} best distance {} (attempts {})",
                    outcome.restart, outcome.best_distance, outcome.attempts_used
                );

                if outcome.best_distance < global_best_distance {
                    global_best_start = outcome.best_start.clone();
                    global_best_distance = outcome.best_distance;
                    println!(
                        "try_init: restart {} improved global best distance to {}",
                        outcome.restart, global_best_distance
                    );
                }
            }

            if let Some(success_outcome) = outcomes
                .iter()
                .find(|outcome| outcome.success_start.is_some())
            {
                let success_start = success_outcome
                    .success_start
                    .as_ref()
                    .expect("success outcome without state")
                    .clone();
                println!(
                    "try_init: success in restart {} after {} attempts",
                    success_outcome.restart, success_outcome.attempts_used
                );

                region_a.allowed_deviation = old_region_a_allowed_deviation;
                region_b.allowed_deviation = old_region_b_allowed_deviation;

                if recursion == 0 {
                    return Some(Self {
                        ensemble: success_start,
                        region_a,
                        region_b,
                        delta_t,
                        number_of_steps,
                    });
                } else {
                    return TransitionPathState::try_init(
                        success_start,
                        region_a,
                        region_b,
                        delta_t,
                        number_of_steps,
                        number_of_restarts,
                        recursion - 1,
                    );
                }
            }
        }

        let final_steps = number_of_steps.checked_mul(2)?;
        println!(
            "try_init: final attempt with global best and {} steps",
            final_steps
        );

        let final_distance =
            terminal_distance_to_region_b(&global_best_start, region_b, delta_t, final_steps);
        println!(
            "try_init: final attempt distance to region B = {}",
            final_distance
        );

        region_a.allowed_deviation = old_region_a_allowed_deviation;
        region_b.allowed_deviation = old_region_b_allowed_deviation;

        if final_distance <= 0.0 {
            if recursion == 0 {
                return Some(Self {
                    ensemble: global_best_start,
                    region_a,
                    region_b,
                    delta_t,
                    number_of_steps: final_steps,
                });
            } else {
                return TransitionPathState::try_init(
                    global_best_start,
                    region_a,
                    region_b,
                    delta_t,
                    number_of_steps,
                    number_of_restarts,
                    recursion - 1,
                );
            }
        }

        println!(
            "try_init: failed after {} restarts; best distance = {}",
            number_of_restarts, global_best_distance
        );

        None
    }
}

fn greedy_search_from_start(
    start: Ensemble,
    region_a: TargetRegion,
    region_b: TargetRegion,
    delta_t: f64,
    number_of_steps: usize,
    restart: usize,
) -> SearchOutcome {
    let restart_seed = restart_seed(restart);
    let mut rng = Pcg64Mcg::new(restart_seed);
    let mut best_start = start;
    let mut best_distance =
        terminal_distance_to_region_b(&best_start, region_b, delta_t, number_of_steps);

    if best_distance <= 0.0 {
        return SearchOutcome {
            restart,
            best_start: best_start.clone(),
            best_distance,
            success_start: Some(best_start),
            attempts_used: 0,
        };
    }

    for attempt_idx in 0..MAX_INIT_ATTEMPTS {
        let attempt = attempt_idx + 1;
        let mut trial_start = best_start.clone();
        let wiggle_mode = match rng.random_range(0..3) {
            0 => WiggleMode::Positions,
            1 => WiggleMode::Momenta,
            _ => WiggleMode::PositionsAndMomenta,
        };
        let scale_factor = rng.random_range(0.5..1.0);

        trial_start.random_wiggle(
            &mut rng,
            POSITION_WIGGLE_SCALE * scale_factor,
            MOMENTUM_WIGGLE_SCALE * scale_factor,
            wiggle_mode,
        );

        if !trial_start.close_to_region(region_a) {
            continue;
        }

        let trial_distance =
            terminal_distance_to_region_b(&trial_start, region_b, delta_t, number_of_steps);

        if trial_distance <= 0.0 {
            return SearchOutcome {
                restart,
                best_start: trial_start.clone(),
                best_distance: trial_distance,
                success_start: Some(trial_start),
                attempts_used: attempt,
            };
        }

        if trial_distance < best_distance {
            best_start = trial_start;
            best_distance = trial_distance;
        }
    }

    SearchOutcome {
        restart,
        best_start,
        best_distance,
        success_start: None,
        attempts_used: MAX_INIT_ATTEMPTS,
    }
}

fn restart_seed(restart: usize) -> u128 {
    const SEED_STEP: u128 = 0x9E37_79B9_7F4A_7C15_6A09_E667_F3BC_C909;
    RNG_SEED.wrapping_add((restart as u128).wrapping_mul(SEED_STEP))
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
