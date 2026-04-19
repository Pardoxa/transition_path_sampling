// Bolhuis 1998
// Masse auf 1 gesetzt

use rand::RngExt;
use std::io::Write;

const RADIUS_NEAR_GROUNDSTATE: f64 = 1.1185;
const MIN_R_2: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Particle {
    pub x: f64,
    pub y: f64,
    pub p_x: f64,
    pub p_y: f64,
}

impl Particle {
    pub fn kinetic_energy(&self) -> f64 {
        (self.p_x * self.p_x + self.p_y * self.p_y) * 0.5
    }

    /// The potential energy resulting from the Lennard jones potential
    pub fn potential_energy(&self, other: &Self) -> f64 {
        let x = self.x - other.x;
        let y = self.y - other.y;
        let mut r_2 = x * x + y * y;
        if r_2 < MIN_R_2 {
            r_2 = MIN_R_2;
        }

        let r_6 = r_2 * r_2 * r_2;

        let r_12 = r_6 * r_6;

        4.0 * (r_12.recip() - r_6.recip())
    }

    pub fn force(&self, other: &Self) -> [f64; 2] {
        let x_derivative = force(self.x, other.x, self.y, other.y);
        let y_derivative = force(self.y, other.y, self.x, other.x);
        [x_derivative, y_derivative]
    }
}

fn force(x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
    // https://www.wolframalpha.com/input?i=d%2Fda+%281%2F%28%28%28a-x2%29%5E2%2B%28y1-y2%29%5E2%29%5E%283%29%29+-+1%2F%28%28%28a-x2%29%5E2%2B%28y1-y2%29%5E2%29%5E%286%29%29%29+
    let x_diff = x1 - x2;
    let x_diff_sq = x_diff * x_diff;
    let y_diff = y1 - y2;
    let y_diff_sq = y_diff * y_diff;
    let mut x_diff_sq_plus_y_diff_sq = x_diff_sq + y_diff_sq;
    if x_diff_sq_plus_y_diff_sq < MIN_R_2 {
        x_diff_sq_plus_y_diff_sq = MIN_R_2;
    }
    let x_diff_sq_plus_y_diff_sq_pow2 = x_diff_sq_plus_y_diff_sq * x_diff_sq_plus_y_diff_sq;
    let oben = 6.0 * x_diff * (2.0 - x_diff_sq_plus_y_diff_sq_pow2 * x_diff_sq_plus_y_diff_sq);
    let unten = x_diff_sq_plus_y_diff_sq_pow2
        * x_diff_sq_plus_y_diff_sq_pow2
        * x_diff_sq_plus_y_diff_sq_pow2
        * x_diff_sq_plus_y_diff_sq;
    4.0 * oben / unten
}

#[derive(Clone, Debug)]
pub struct Ensemble {
    pub particles: Vec<Particle>,
    current_time: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WiggleMode {
    Positions,
    Momenta,
    PositionsAndMomenta,
}

impl Ensemble {
    pub fn minimum_fig2() -> Self {
        let r = RADIUS_NEAR_GROUNDSTATE;
        let mut particles = Vec::with_capacity(7);
        // center particle
        particles.push(Particle {
            x: 0.0,
            y: 0.0,
            p_x: 0.0,
            p_y: 0.0,
        });
        let angle_triangle = std::f64::consts::FRAC_PI_6;
        particles.push(Particle {
            x: 0.0,
            y: angle_triangle.cos() * r * 2.0,
            p_x: 0.0,
            p_y: 0.0,
        });
        let step = std::f64::consts::FRAC_PI_3;
        particles.extend((0..5).map(|i| {
            let theta = (i as f64) * step;
            Particle {
                x: r * theta.cos(),
                y: r * theta.sin(),
                p_x: 0.0,
                p_y: 0.0,
            }
        }));
        let mut ensemble = Ensemble {
            particles,
            current_time: 0.0,
        };
        ensemble.find_local_minima(100);
        ensemble.current_time = 0.0;
        ensemble
    }

    pub fn minimum_fig3() -> Self {
        let r = RADIUS_NEAR_GROUNDSTATE;
        let mut particles = Vec::with_capacity(7);
        // center particle
        particles.push(Particle {
            x: 0.0,
            y: 0.0,
            p_x: 0.0,
            p_y: 0.0,
        });
        let angle_triangle = std::f64::consts::FRAC_PI_6;
        particles.push(Particle {
            x: angle_triangle.sin() * r * 3.0,
            y: angle_triangle.cos() * r,
            p_x: 0.0,
            p_y: 0.0,
        });
        let step = std::f64::consts::FRAC_PI_3;
        particles.extend((0..5).map(|i| {
            let theta = (i as f64) * step;
            Particle {
                x: r * theta.cos(),
                y: r * theta.sin(),
                p_x: 0.0,
                p_y: 0.0,
            }
        }));
        let mut ensemble = Ensemble {
            particles,
            current_time: 0.0,
        };
        ensemble.find_local_minima(1000);
        ensemble.current_time = 0.0;
        ensemble
    }

    pub fn new_groundstate() -> Self {
        let mut particles = Vec::with_capacity(7);
        // center particle
        particles.push(Particle {
            x: 0.0,
            y: 0.0,
            p_x: 0.0,
            p_y: 0.0,
        });
        let step = std::f64::consts::FRAC_PI_3;
        particles.extend((0..6).map(|i| {
            let theta = (i as f64) * step;
            Particle {
                x: RADIUS_NEAR_GROUNDSTATE * theta.cos(),
                y: RADIUS_NEAR_GROUNDSTATE * theta.sin(),
                p_x: 0.0,
                p_y: 0.0,
            }
        }));
        let mut ensemble = Ensemble {
            particles,
            current_time: 0.0,
        };
        ensemble.find_local_minima(100);
        ensemble.current_time = 0.0;
        ensemble
    }

    /// Returns a groundstate ensemble whose impulses are kicked toward the
    /// figure-3 geometry, scaled by `kick_scale`.
    ///
    /// Particle positions are exactly the same as `new_groundstate`; only
    /// `p_x/p_y` are modified.
    pub fn new_groundstate_kicked_towards_fig3(kick_scale: f64) -> Self {
        let mut ensemble = Self::new_groundstate();
        if ensemble.particles.is_empty() {
            return ensemble;
        }

        let kick_vectors = groundstate_to_fig3_kick_vectors(&ensemble.particles);
        for (particle, [dx, dy]) in ensemble.particles.iter_mut().zip(kick_vectors.into_iter()) {
            particle.p_x = dx * kick_scale;
            particle.p_y = dy * kick_scale;
        }

        ensemble
    }

    // Angenommen ich bin in der Nähe eines Minimums.
    // Dann sollte ich durch: Bewegungsgleichung ein bisschen verfolgen
    // + impuls abschneiden näher zum Minimum kommen.
    // Und das iteriere ich einfach ein paar mal
    fn find_local_minima(&mut self, iterations: usize) {
        for _ in 0..iterations {
            for _ in 0..100 {
                self.velocity_verlet_step_by(0.001);
            }
            self.set_impuls_0();
        }
    }

    pub fn set_impuls_0(&mut self) {
        self.particles.iter_mut().for_each(|particle| {
            particle.p_x = 0.0;
            particle.p_y = 0.0;
        });
    }

    pub fn hamiltonian(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    pub fn kinetic_energy(&self) -> f64 {
        self.particles.iter().map(Particle::kinetic_energy).sum()
    }

    pub fn potential_energy(&self) -> f64 {
        let mut particle_slice = self.particles.as_slice();
        let mut energy_sum = 0.0;
        while let Some((first_particle, rest)) = particle_slice.split_first() {
            energy_sum += rest
                .iter()
                .map(|other_particle| first_particle.potential_energy(other_particle))
                .sum::<f64>();
            particle_slice = rest;
        }

        energy_sum
    }

    fn forces(&self) -> Vec<[f64; 2]> {
        // Forces als summe der forces
        let impuls_ableitung: Vec<_> = (0..self.particles.len())
            .map(|i| {
                let left = &self.particles[0..i];
                let this = &self.particles[i];
                let right = self.particles.get(i + 1..);
                let right_iter = right.iter().flat_map(|&particles| particles);

                let mut derivative = [0.0; 2];

                for particle in left.iter().chain(right_iter) {
                    derivative
                        .iter_mut()
                        .zip(this.force(particle))
                        .for_each(|(a, b)| *a += b);
                }

                derivative
            })
            .collect();

        impuls_ableitung
    }

    // Nun velocity verlet: https://www.thp.uni-koeln.de/trebst/PracticalCourse/molecular_dynamics/molecular_dynamics.pdf
    // https://www.algorithm-archive.org/contents/verlet_integration/verlet_integration.html
    pub fn velocity_verlet_step_by(&mut self, delta_time: f64) {
        // Nun habe ich alle ableitungen.
        let current_forces: Vec<_> = self.forces();

        let delta_time_2 = delta_time * 0.5;
        let delta_time_sq_div_2 = delta_time * delta_time * 0.5;

        // update location
        self.particles
            .iter_mut()
            .zip(current_forces.iter())
            .for_each(|(particle, &[fx, fy])| {
                particle.x += delta_time * particle.p_x + delta_time_sq_div_2 * fx;
                particle.y += delta_time * particle.p_y + delta_time_sq_div_2 * fy;
            });

        let future_forces = self.forces();

        // Update impulses
        self.particles
            .iter_mut()
            .zip(current_forces)
            .zip(future_forces)
            .for_each(|((particle, current_force), future_force)| {
                particle.p_x += delta_time_2 * (current_force[0] + future_force[0]);
                particle.p_y += delta_time_2 * (current_force[1] + future_force[1]);
            });

        self.current_time += delta_time;
    }

    pub fn write<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        write!(writer, "{} ", self.current_time)?;
        for particle in self.particles.iter() {
            write!(writer, "{} {} ", particle.x, particle.y)?;
        }
        writeln!(writer)
    }

    pub fn write_positions<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        for particle in self.particles.iter() {
            write!(writer, "{} {} ", particle.x, particle.y)?;
        }
        writeln!(writer)
    }

    pub fn write_gnuplot_positions<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        writeln!(writer, "set t x11")?;
        writeln!(writer, "set xlabel 'x'")?;
        writeln!(writer, "set ylabel 'y'")?;

        writeln!(writer, "$data << EOF")?;

        for particle in self.particles.iter() {
            writeln!(writer, "{} {}", particle.x, particle.y)?;
        }
        writeln!(writer, "EOF")?;
        writeln!(writer, "p $data")
    }

    pub fn random_wiggle<R: RngExt + ?Sized>(
        &mut self,
        rng: &mut R,
        position_scale: f64,
        momentum_scale: f64,
        mode: WiggleMode,
    ) {
        if self.particles.is_empty() {
            return;
        }

        let index = rng.random_range(0..self.particles.len());
        let particle = &mut self.particles[index];

        match mode {
            WiggleMode::Positions => {
                particle.x += rng.random_range(-position_scale..position_scale);
                particle.y += rng.random_range(-position_scale..position_scale);
            }
            WiggleMode::Momenta => {
                particle.p_x += rng.random_range(-momentum_scale..momentum_scale);
                particle.p_y += rng.random_range(-momentum_scale..momentum_scale);
            }
            WiggleMode::PositionsAndMomenta => {
                particle.x += rng.random_range(-position_scale..position_scale);
                particle.y += rng.random_range(-position_scale..position_scale);
                particle.p_x += rng.random_range(-momentum_scale..momentum_scale);
                particle.p_y += rng.random_range(-momentum_scale..momentum_scale);
            }
        }
    }

    pub fn distance_to_region(&self, target_region: TargetRegion) -> f64 {
        let deviation = (self.potential_energy() - target_region.potential_energy).abs();
        (deviation - target_region.allowed_deviation).max(0.0)
    }

    pub fn close_to_region(&self, target_region: TargetRegion) -> bool {
        self.distance_to_region(target_region) <= 0.0
    }

    pub fn distance_to_hamiltonian_region(&self, target_region: TargetRegion) -> f64 {
        let deviation = (self.hamiltonian() - target_region.potential_energy).abs();
        (deviation - target_region.allowed_deviation).max(0.0)
    }

    pub fn set_momentum_to_region(
        &mut self,
        target_region: TargetRegion,
    ) -> Result<(), MomentumRegionError> {
        if !target_region.allowed_deviation.is_finite()
            || !target_region.potential_energy.is_finite()
            || target_region.allowed_deviation < 0.0
        {
            return Err(MomentumRegionError::InvalidTargetRegion);
        }

        let potential_energy = self.potential_energy();
        let min_total_energy = target_region.potential_energy - target_region.allowed_deviation;
        let max_total_energy = target_region.potential_energy + target_region.allowed_deviation;

        if potential_energy > max_total_energy {
            return Err(MomentumRegionError::PotentialTooHigh {
                potential_energy,
                max_total_energy,
            });
        }

        self.set_impuls_0();

        if potential_energy < min_total_energy {
            let needed_kinetic_energy = min_total_energy - potential_energy;
            let first_particle = self
                .particles
                .first_mut()
                .ok_or(MomentumRegionError::NoParticles)?;
            first_particle.p_x = (2.0 * needed_kinetic_energy).sqrt();
        }

        if self.distance_to_hamiltonian_region(target_region) <= 0.0 {
            Ok(())
        } else {
            Err(MomentumRegionError::NumericalMiss)
        }
    }
}

/// Builds per-particle kick vectors from the groundstate geometry toward the
/// figure-3 geometry using a minimum-cost one-to-one assignment.
///
/// The returned vectors are zero-mean so the total momentum is zero after
/// applying them with a common scalar.
fn groundstate_to_fig3_kick_vectors(groundstate_particles: &[Particle]) -> Vec<[f64; 2]> {
    let fig3 = Ensemble::minimum_fig3();
    let assignment = best_particle_assignment(groundstate_particles, &fig3.particles);

    let mut vectors: Vec<[f64; 2]> = groundstate_particles
        .iter()
        .enumerate()
        .map(|(source_idx, source)| {
            let target = &fig3.particles[assignment[source_idx]];
            [target.x - source.x, target.y - source.y]
        })
        .collect();

    let count = vectors.len() as f64;
    let [sum_x, sum_y] = vectors
        .iter()
        .fold([0.0, 0.0], |[acc_x, acc_y], [x, y]| [acc_x + x, acc_y + y]);
    let mean_x = sum_x / count;
    let mean_y = sum_y / count;

    vectors.iter_mut().for_each(|[x, y]| {
        *x -= mean_x;
        *y -= mean_y;
    });

    vectors
}

/// Computes total squared-distance assignment cost between two particle sets.
fn assignment_cost(from: &[Particle], to: &[Particle], assignment: &[usize]) -> f64 {
    from.iter()
        .zip(assignment.iter())
        .map(|(source, &target_idx)| {
            let target = &to[target_idx];
            let dx = source.x - target.x;
            let dy = source.y - target.y;
            dx * dx + dy * dy
        })
        .sum()
}

/// Finds the minimum-cost one-to-one particle assignment by brute-force
/// permutation search.
fn best_particle_assignment(from: &[Particle], to: &[Particle]) -> Vec<usize> {
    assert_eq!(from.len(), to.len());

    let mut current_assignment: Vec<usize> = (0..from.len()).collect();
    let mut best_assignment = current_assignment.clone();
    let mut best_cost = assignment_cost(from, to, &current_assignment);

    while next_permutation(&mut current_assignment) {
        let current_cost = assignment_cost(from, to, &current_assignment);
        if current_cost < best_cost {
            best_cost = current_cost;
            best_assignment = current_assignment.clone();
        }
    }

    best_assignment
}

/// Advances to the next lexicographic permutation in-place.
///
/// Returns `false` when the input was the last permutation.
fn next_permutation(permutation: &mut [usize]) -> bool {
    if permutation.len() < 2 {
        return false;
    }

    let mut pivot = permutation.len() - 2;
    loop {
        if permutation[pivot] < permutation[pivot + 1] {
            break;
        }
        if pivot == 0 {
            permutation.reverse();
            return false;
        }
        pivot -= 1;
    }

    let mut successor = permutation.len() - 1;
    while permutation[successor] <= permutation[pivot] {
        successor -= 1;
    }

    permutation.swap(pivot, successor);
    permutation[pivot + 1..].reverse();
    true
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MomentumRegionError {
    InvalidTargetRegion,
    PotentialTooHigh {
        potential_energy: f64,
        max_total_energy: f64,
    },
    NoParticles,
    NumericalMiss,
}

#[derive(Clone, Copy, Debug)]
pub struct TargetRegion {
    pub potential_energy: f64,
    pub allowed_deviation: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f64, y: f64) -> Particle {
        Particle {
            x,
            y,
            p_x: 0.0,
            p_y: 0.0,
        }
    }

    #[test]
    fn brute_force_assignment_finds_exact_permutation_match() {
        let source = vec![
            point(0.0, 0.0),
            point(2.0, 1.0),
            point(-1.5, 3.0),
            point(4.2, -2.0),
            point(-3.0, 1.3),
            point(0.7, -4.1),
            point(5.5, 2.7),
        ];

        let permutation = [3usize, 0, 6, 2, 5, 1, 4];
        let target: Vec<_> = permutation.iter().map(|&idx| source[idx]).collect();

        let assignment = best_particle_assignment(&source, &target);
        let total_cost = assignment_cost(&source, &target, &assignment);

        assert!(
            total_cost <= 1.0e-12,
            "expected near-zero cost, got {total_cost}"
        );

        for (source_idx, &target_idx) in assignment.iter().enumerate() {
            let src = source[source_idx];
            let dst = target[target_idx];
            assert!((src.x - dst.x).abs() <= 1.0e-12);
            assert!((src.y - dst.y).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn kicked_groundstate_keeps_positions_and_zero_net_momentum() {
        let ground = Ensemble::new_groundstate();
        let kicked = Ensemble::new_groundstate_kicked_towards_fig3(0.2);

        for (a, b) in ground.particles.iter().zip(kicked.particles.iter()) {
            assert!((a.x - b.x).abs() <= 1.0e-12);
            assert!((a.y - b.y).abs() <= 1.0e-12);
        }

        let [sum_px, sum_py] = kicked
            .particles
            .iter()
            .fold([0.0, 0.0], |[acc_x, acc_y], p| {
                [acc_x + p.p_x, acc_y + p.p_y]
            });

        assert!(sum_px.abs() <= 1.0e-12, "sum_px not near zero: {sum_px}");
        assert!(sum_py.abs() <= 1.0e-12, "sum_py not near zero: {sum_py}");
    }
}
