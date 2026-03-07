// Bolhuis 1998
// Masse auf 1 gesetzt

use std::io::Write;

const RADIUS_NEAR_GROUNDSTATE: f64 = 1.1185;
const MIN_R_2: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Particle {
    x: f64,
    y: f64,
    p_x: f64,
    p_y: f64,
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
    particles: Vec<Particle>,
    current_time: f64,
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
            x: -angle_triangle.sin() * r * 3.0,
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
}
