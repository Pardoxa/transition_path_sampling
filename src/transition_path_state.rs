use crate::particle::*;
use rand::RngExt;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use std::num::NonZeroUsize;

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

#[derive(Clone, Debug)]
pub struct KickScaleTuningResult {
    pub best_scale: f64,
    pub best_distance: f64,
    pub best_start: Ensemble,
}

#[derive(Clone, Debug)]
pub struct ParticleImpulseSweepTuningResult {
    pub best_distance: f64,
    pub best_start: Ensemble,
    pub sweeps_completed: usize,
}

#[derive(Clone, Debug)]
pub struct TransitionTuningConfig {
    region_b: TargetRegion,
    delta_t: f64,
    number_of_steps: NonZeroUsize,
    initial_scale: f64,
    initial_step: f64,
    scale_search_iterations: NonZeroUsize,
    impulse_sweeps: NonZeroUsize,
    angle_step: f64,
    length_step_fraction: f64,
    max_projection_repair_passes: usize,
}

#[derive(Clone, Debug)]
pub struct TransitionTuningConfigBuilder {
    region_b: TargetRegion,
    delta_t: f64,
    number_of_steps: NonZeroUsize,
    initial_scale: f64,
    initial_step: f64,
    scale_search_iterations: NonZeroUsize,
    impulse_sweeps: NonZeroUsize,
    angle_step: f64,
    length_step_fraction: f64,
    max_projection_repair_passes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TuningConfigErrorKind {
    RegionBPotentialEnergyNotFinite(f64),
    RegionBAllowedDeviationInvalid(f64),
    DeltaTInvalid(f64),
    InitialScaleInvalid(f64),
    InitialStepInvalid(f64),
    AngleStepInvalid(f64),
    LengthStepFractionInvalid(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TuningConfigBuildError {
    pub errors: Vec<TuningConfigErrorKind>,
}

impl std::fmt::Display for TuningConfigBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid transition tuning config: {:?}", self.errors)
    }
}

impl std::error::Error for TuningConfigBuildError {}

impl Default for TransitionTuningConfigBuilder {
    fn default() -> Self {
        Self {
            region_b: TargetRegion {
                potential_energy: -11.47,
                allowed_deviation: 0.01,
            },
            delta_t: 0.01,
            number_of_steps: NonZeroUsize::new(20_000).expect("non-zero default"),
            initial_scale: 0.2,
            initial_step: 0.05,
            scale_search_iterations: NonZeroUsize::new(30).expect("non-zero default"),
            impulse_sweeps: NonZeroUsize::new(2).expect("non-zero default"),
            angle_step: 0.08,
            length_step_fraction: 0.10,
            max_projection_repair_passes: 2,
        }
    }
}

impl TransitionTuningConfigBuilder {
    /// Creates a builder with defaults matching the current tuning setup.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn region_b(mut self, region_b: TargetRegion) -> Self {
        self.region_b = region_b;
        self
    }

    pub fn delta_t(mut self, delta_t: f64) -> Self {
        self.delta_t = delta_t;
        self
    }

    pub fn number_of_steps(mut self, number_of_steps: NonZeroUsize) -> Self {
        self.number_of_steps = number_of_steps;
        self
    }

    pub fn initial_scale(mut self, initial_scale: f64) -> Self {
        self.initial_scale = initial_scale;
        self
    }

    pub fn initial_step(mut self, initial_step: f64) -> Self {
        self.initial_step = initial_step;
        self
    }

    pub fn scale_search_iterations(mut self, scale_search_iterations: NonZeroUsize) -> Self {
        self.scale_search_iterations = scale_search_iterations;
        self
    }

    pub fn impulse_sweeps(mut self, impulse_sweeps: NonZeroUsize) -> Self {
        self.impulse_sweeps = impulse_sweeps;
        self
    }

    pub fn angle_step(mut self, angle_step: f64) -> Self {
        self.angle_step = angle_step;
        self
    }

    pub fn length_step_fraction(mut self, length_step_fraction: f64) -> Self {
        self.length_step_fraction = length_step_fraction;
        self
    }

    pub fn max_projection_repair_passes(mut self, max_projection_repair_passes: usize) -> Self {
        self.max_projection_repair_passes = max_projection_repair_passes;
        self
    }

    /// Validates all configured parameters and builds an immutable tuning
    /// configuration.
    pub fn build(self) -> Result<TransitionTuningConfig, TuningConfigBuildError> {
        let mut errors = Vec::new();

        if !self.region_b.potential_energy.is_finite() {
            errors.push(TuningConfigErrorKind::RegionBPotentialEnergyNotFinite(
                self.region_b.potential_energy,
            ));
        }

        if !self.region_b.allowed_deviation.is_finite() || self.region_b.allowed_deviation < 0.0 {
            errors.push(TuningConfigErrorKind::RegionBAllowedDeviationInvalid(
                self.region_b.allowed_deviation,
            ));
        }

        if !self.delta_t.is_finite() || self.delta_t <= 0.0 {
            errors.push(TuningConfigErrorKind::DeltaTInvalid(self.delta_t));
        }

        if !self.initial_scale.is_finite() || self.initial_scale < 0.0 {
            errors.push(TuningConfigErrorKind::InitialScaleInvalid(
                self.initial_scale,
            ));
        }

        if !self.initial_step.is_finite() || self.initial_step <= 0.0 {
            errors.push(TuningConfigErrorKind::InitialStepInvalid(self.initial_step));
        }

        if !self.angle_step.is_finite() || self.angle_step <= 0.0 {
            errors.push(TuningConfigErrorKind::AngleStepInvalid(self.angle_step));
        }

        if !self.length_step_fraction.is_finite() || self.length_step_fraction <= 0.0 {
            errors.push(TuningConfigErrorKind::LengthStepFractionInvalid(
                self.length_step_fraction,
            ));
        }

        if !errors.is_empty() {
            return Err(TuningConfigBuildError { errors });
        }

        Ok(TransitionTuningConfig {
            region_b: self.region_b,
            delta_t: self.delta_t,
            number_of_steps: self.number_of_steps,
            initial_scale: self.initial_scale,
            initial_step: self.initial_step,
            scale_search_iterations: self.scale_search_iterations,
            impulse_sweeps: self.impulse_sweeps,
            angle_step: self.angle_step,
            length_step_fraction: self.length_step_fraction,
            max_projection_repair_passes: self.max_projection_repair_passes,
        })
    }
}

impl TransitionTuningConfig {
    pub fn builder() -> TransitionTuningConfigBuilder {
        TransitionTuningConfigBuilder::new()
    }
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

/// Builds a deterministic RNG seed for a given restart index.
///
/// Different restarts get decorrelated seeds while still being reproducible.
fn restart_seed(restart: usize) -> u128 {
    const SEED_STEP: u128 = 0x9E37_79B9_7F4A_7C15_6A09_E667_F3BC_C909;
    RNG_SEED.wrapping_add((restart as u128).wrapping_mul(SEED_STEP))
}

/// Rolls a cloned ensemble for `number_of_steps` and returns its terminal
/// distance to region B.
pub(crate) fn terminal_distance_to_region_b(
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

#[derive(Clone, Debug)]
struct KickCandidate {
    index: usize,
    scale: f64,
    distance: f64,
    start: Ensemble,
}

impl TransitionTuningConfig {
    /// Tunes a single global kick scale for
    /// `Ensemble::new_groundstate_kicked_towards_fig3` using an adaptive
    /// 3-point search (`scale-step`, `scale`, `scale+step`).
    pub fn tune_fig3_kick_scale_for_region_b(&self) -> KickScaleTuningResult {
        let mut current_scale = self.initial_scale;
        let mut step = self.initial_step;

        let initial_start = Ensemble::new_groundstate_kicked_towards_fig3(current_scale);
        let mut best_distance = terminal_distance_to_region_b(
            &initial_start,
            self.region_b,
            self.delta_t,
            self.number_of_steps.get(),
        );
        let mut best_scale = current_scale;
        let mut best_start = initial_start;

        for _iteration_idx in 0..self.scale_search_iterations.get() {
            let scales = [
                (current_scale - step).max(0.0),
                current_scale,
                current_scale + step,
            ];

            let candidates: Vec<_> = scales
                .par_iter()
                .enumerate()
                .map(|(index, &scale)| {
                    let start = Ensemble::new_groundstate_kicked_towards_fig3(scale);
                    let distance = terminal_distance_to_region_b(
                        &start,
                        self.region_b,
                        self.delta_t,
                        self.number_of_steps.get(),
                    );
                    KickCandidate {
                        index,
                        scale,
                        distance,
                        start,
                    }
                })
                .collect();

            let winner = select_best_candidate(&candidates);

            #[cfg(feature = "kick_tuning_trace")]
            println!(
                "kick_tune iter={} scales=[{:.12}, {:.12}, {:.12}] dists=[{:.12}, {:.12}, {:.12}] best_scale={:.12} best_dist={:.12}",
                _iteration_idx,
                scales[0],
                scales[1],
                scales[2],
                candidates[0].distance,
                candidates[1].distance,
                candidates[2].distance,
                winner.scale,
                winner.distance
            );

            if winner.distance < best_distance {
                best_distance = winner.distance;
                best_scale = winner.scale;
                best_start = winner.start.clone();
            }

            if best_distance <= 0.0 {
                break;
            }

            if winner.index == 1 {
                step *= 0.5;
            } else {
                current_scale = winner.scale;
                step *= 1.3;
            }
        }

        KickScaleTuningResult {
            best_scale,
            best_distance,
            best_start,
        }
    }

    /// Refines a kicked starting ensemble by local impulse search over multiple
    /// sweeps.
    ///
    /// For each sweep, each particle is optimized in turn by trying small angle
    /// and length perturbations of its impulse. Candidate perturbations for one
    /// particle are evaluated in parallel and the best improving candidate is
    /// kept.
    ///
    /// Each candidate is projected to zero net momentum before scoring so the
    /// optimization objective matches the preferred physical constraint.
    ///
    /// After each full optimizer pass, a zero-net-momentum projection is
    /// checked. If that projection worsens the terminal distance to region B,
    /// another full pass is retried from the projected state, up to
    /// `max_projection_repair_passes` retries.
    pub fn tune_particle_impulses_by_sweeps_for_region_b(
        &self,
        start: Ensemble,
    ) -> ParticleImpulseSweepTuningResult {
        let mut current_pass_start = start;
        let mut total_sweeps_completed = 0usize;

        let initial_distance = terminal_distance_to_region_b(
            &current_pass_start,
            self.region_b,
            self.delta_t,
            self.number_of_steps.get(),
        );
        let mut best_unconstrained_start = current_pass_start.clone();
        let mut best_unconstrained_distance = initial_distance;

        let mut best_zero_momentum_candidate: Option<Ensemble> = None;
        let mut best_zero_momentum_distance = f64::INFINITY;

        for pass_idx in 0..=self.max_projection_repair_passes {
            let unconstrained_pass_result =
                run_particle_impulse_sweeps_unconstrained(current_pass_start, self, pass_idx);
            total_sweeps_completed += unconstrained_pass_result.sweeps_completed;

            if unconstrained_pass_result.best_distance + 1.0e-12 < best_unconstrained_distance {
                best_unconstrained_distance = unconstrained_pass_result.best_distance;
                best_unconstrained_start = unconstrained_pass_result.best_start.clone();
            }

            let mut zero_momentum_candidate = unconstrained_pass_result.best_start.clone();
            enforce_zero_net_momentum(&mut zero_momentum_candidate);
            let zero_momentum_candidate_distance = terminal_distance_to_region_b(
                &zero_momentum_candidate,
                self.region_b,
                self.delta_t,
                self.number_of_steps.get(),
            );

            if zero_momentum_candidate_distance + 1.0e-12 < best_zero_momentum_distance {
                best_zero_momentum_distance = zero_momentum_candidate_distance;
                best_zero_momentum_candidate = Some(zero_momentum_candidate.clone());
            }

            #[cfg(feature = "kick_tuning_trace")]
            println!(
                "particle_tune pass={} unconstrained={:.12} projected_zero_momentum={:.12}",
                pass_idx, unconstrained_pass_result.best_distance, zero_momentum_candidate_distance
            );

            if zero_momentum_candidate_distance <= unconstrained_pass_result.best_distance + 1.0e-12
            {
                break;
            }

            if pass_idx == self.max_projection_repair_passes {
                break;
            }

            current_pass_start = zero_momentum_candidate;
        }

        let (selected_start, selected_distance) =
            if let Some(zero_momentum_candidate) = best_zero_momentum_candidate {
                if best_zero_momentum_distance <= best_unconstrained_distance + 1.0e-12 {
                    #[cfg(feature = "kick_tuning_trace")]
                    println!(
                        "particle_tune final=zero_momentum distance={:.12}",
                        best_zero_momentum_distance
                    );
                    (zero_momentum_candidate, best_zero_momentum_distance)
                } else {
                    #[cfg(feature = "kick_tuning_trace")]
                    println!(
                        "particle_tune final=unconstrained_fallback distance={:.12}",
                        best_unconstrained_distance
                    );
                    (best_unconstrained_start, best_unconstrained_distance)
                }
            } else {
                (best_unconstrained_start, best_unconstrained_distance)
            };

        ParticleImpulseSweepTuningResult {
            best_distance: selected_distance,
            best_start: selected_start,
            sweeps_completed: total_sweeps_completed,
        }
    }
}

/// Runs one unconstrained particle-impulse sweep pass.
///
/// This optimizer updates one particle at a time and keeps only improving
/// candidates. Candidate states are projected to zero net momentum before
/// scoring so the sweep objective matches the physical preference.
fn run_particle_impulse_sweeps_unconstrained(
    start: Ensemble,
    config: &TransitionTuningConfig,
    _pass_idx: usize,
) -> ParticleImpulseSweepTuningResult {
    let mut working = start;
    let mut best_distance = terminal_distance_to_region_b(
        &working,
        config.region_b,
        config.delta_t,
        config.number_of_steps.get(),
    );
    let mut sweeps_completed = 0usize;

    for _sweep_idx in 0..config.impulse_sweeps.get() {
        if best_distance <= 0.0 {
            break;
        }

        let mut improved_in_sweep = false;

        #[cfg(feature = "kick_tuning_trace")]
        println!(
            "particle_tune pass={} sweep={} start_distance={:.12}",
            _pass_idx, _sweep_idx, best_distance
        );

        for particle_idx in 0..working.particles.len() {
            let particle = working.particles[particle_idx];
            let base_angle = particle.p_y.atan2(particle.p_x);
            let base_len = (particle.p_x * particle.p_x + particle.p_y * particle.p_y).sqrt();
            let length_delta = (base_len * config.length_step_fraction).max(1.0e-4);

            let angle_offsets = [-2.0, -1.0, 0.0, 1.0, 2.0];
            let length_offsets = [-length_delta, 0.0, length_delta];

            let candidate_params: Vec<_> = angle_offsets
                .iter()
                .flat_map(|&angle_factor| {
                    length_offsets.iter().map(move |&length_offset| {
                        (base_angle + angle_factor * config.angle_step, length_offset)
                    })
                })
                .collect();

            let best_candidate = candidate_params
                .par_iter()
                .map(|&(candidate_angle, length_offset)| {
                    let mut candidate = working.clone();
                    let candidate_len = (base_len + length_offset).max(0.0);
                    candidate.particles[particle_idx].p_x = candidate_len * candidate_angle.cos();
                    candidate.particles[particle_idx].p_y = candidate_len * candidate_angle.sin();
                    enforce_zero_net_momentum(&mut candidate);
                    let candidate_distance = terminal_distance_to_region_b(
                        &candidate,
                        config.region_b,
                        config.delta_t,
                        config.number_of_steps.get(),
                    );
                    (candidate_distance, candidate)
                })
                .reduce_with(|best, trial| {
                    if trial.0 + 1.0e-12 < best.0 {
                        trial
                    } else {
                        best
                    }
                })
                .expect("candidate list should not be empty");

            if best_candidate.0 + 1.0e-12 < best_distance {
                working = best_candidate.1;
                best_distance = best_candidate.0;
                improved_in_sweep = true;

                #[cfg(feature = "kick_tuning_trace")]
                println!(
                    "particle_tune pass={} sweep={} particle={} improved_distance={:.12}",
                    _pass_idx, _sweep_idx, particle_idx, best_distance
                );

                if best_distance <= 0.0 {
                    break;
                }
            }
        }

        if !improved_in_sweep {
            break;
        }

        sweeps_completed += 1;

        #[cfg(feature = "kick_tuning_trace")]
        println!(
            "particle_tune pass={} sweep={} end_distance={:.12}",
            _pass_idx, _sweep_idx, best_distance
        );
    }

    ParticleImpulseSweepTuningResult {
        best_distance,
        best_start: working,
        sweeps_completed,
    }
}

/// Selects the best candidate by distance with deterministic tie-breaking.
fn select_best_candidate(candidates: &[KickCandidate]) -> &KickCandidate {
    let mut best = candidates
        .first()
        .expect("select_best_candidate requires non-empty candidates");

    for candidate in candidates.iter().skip(1) {
        if candidate.distance + 1.0e-12 < best.distance {
            best = candidate;
            continue;
        }

        if (candidate.distance - best.distance).abs() <= 1.0e-12
            && candidate_priority(candidate.index) < candidate_priority(best.index)
        {
            best = candidate;
        }
    }

    best
}

/// Returns a deterministic tie-break priority for the 3-point scale search.
///
/// When candidate distances are numerically equal, we prefer the center scale
/// (`index == 1`) so the tuner shrinks step size instead of oscillating.
fn candidate_priority(index: usize) -> u8 {
    match index {
        1 => 0,
        0 => 1,
        2 => 2,
        _ => 3,
    }
}

/// Removes net linear momentum by subtracting the mean impulse vector from all
/// particles.
fn enforce_zero_net_momentum(ensemble: &mut Ensemble) {
    if ensemble.particles.is_empty() {
        return;
    }

    let count = ensemble.particles.len() as f64;
    let (sum_px, sum_py) = ensemble
        .particles
        .iter()
        .fold((0.0, 0.0), |(acc_x, acc_y), particle| {
            (acc_x + particle.p_x, acc_y + particle.p_y)
        });

    let mean_px = sum_px / count;
    let mean_py = sum_py / count;

    if mean_px.abs() <= 1.0e-14 && mean_py.abs() <= 1.0e-14 {
        return;
    }

    for particle in ensemble.particles.iter_mut() {
        particle.p_x -= mean_px;
        particle.p_y -= mean_py;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[test]
    fn kick_scale_tuning_does_not_worsen_result() {
        let initial_scale = 0.2;
        let config = TransitionTuningConfig::builder()
            .number_of_steps(NonZeroUsize::new(500).expect("non-zero"))
            .scale_search_iterations(NonZeroUsize::new(3).expect("non-zero"))
            .build()
            .expect("valid config");

        let baseline_start = Ensemble::new_groundstate_kicked_towards_fig3(initial_scale);
        let baseline_distance = terminal_distance_to_region_b(
            &baseline_start,
            config.region_b,
            config.delta_t,
            config.number_of_steps.get(),
        );

        let result = config.tune_fig3_kick_scale_for_region_b();

        assert!(
            result.best_distance <= baseline_distance + 1.0e-12,
            "tuning worsened result: baseline={} tuned={}",
            baseline_distance,
            result.best_distance
        );
    }

    #[test]
    fn particle_impulse_sweep_tuning_does_not_worsen_result() {
        let config = TransitionTuningConfig::builder()
            .number_of_steps(NonZeroUsize::new(500).expect("non-zero"))
            .impulse_sweeps(NonZeroUsize::new(2).expect("non-zero"))
            .build()
            .expect("valid config");

        let tuned_scale_start = Ensemble::new_groundstate_kicked_towards_fig3(0.8);
        let baseline_distance = terminal_distance_to_region_b(
            &tuned_scale_start,
            config.region_b,
            config.delta_t,
            config.number_of_steps.get(),
        );

        let result = config.tune_particle_impulses_by_sweeps_for_region_b(tuned_scale_start);

        assert!(
            result.best_distance <= baseline_distance + 1.0e-12,
            "sweep tuning worsened result: baseline={} tuned={}",
            baseline_distance,
            result.best_distance
        );
    }

    #[test]
    fn sweep_result_is_stable_under_final_zero_momentum_projection() {
        let config = TransitionTuningConfig::builder()
            .number_of_steps(NonZeroUsize::new(500).expect("non-zero"))
            .impulse_sweeps(NonZeroUsize::new(2).expect("non-zero"))
            .build()
            .expect("valid config");

        let start = Ensemble::new_groundstate_kicked_towards_fig3(0.8);
        let result = config.tune_particle_impulses_by_sweeps_for_region_b(start);

        let tuned_distance = terminal_distance_to_region_b(
            &result.best_start,
            config.region_b,
            config.delta_t,
            config.number_of_steps.get(),
        );

        let mut projected_again = result.best_start.clone();
        enforce_zero_net_momentum(&mut projected_again);
        let projected_again_distance = terminal_distance_to_region_b(
            &projected_again,
            config.region_b,
            config.delta_t,
            config.number_of_steps.get(),
        );

        assert!(
            projected_again_distance <= tuned_distance + 1.0e-12,
            "final zero-momentum projection worsened result: tuned={} projected={}",
            tuned_distance,
            projected_again_distance
        );
    }

    #[test]
    fn builder_rejects_invalid_float_parameters() {
        let err = TransitionTuningConfig::builder()
            .delta_t(0.0)
            .initial_step(-0.1)
            .length_step_fraction(0.0)
            .build()
            .expect_err("invalid config should fail");

        assert!(
            err.errors
                .contains(&TuningConfigErrorKind::DeltaTInvalid(0.0))
        );
        assert!(
            err.errors
                .contains(&TuningConfigErrorKind::InitialStepInvalid(-0.1))
        );
        assert!(
            err.errors
                .contains(&TuningConfigErrorKind::LengthStepFractionInvalid(0.0))
        );
    }
}
