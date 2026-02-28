use std::io::{BufWriter, Write};

fn main() {
    // for i in 1..1000{
    //     let distance = 1.11 +  i as f64 / 10000.0;
    //     let ensemble = Ensemble::new(distance);
    //     let energy = ensemble.hamiltonian();
    //     println!("{distance} {energy}")
    // }

    let mut ensemble = Ensemble::new(1.1185+0.5);
    let file = std::fs::File::create("test.dat").unwrap();
    let mut buf = BufWriter::new(file);
    for i in 0..1000000{
        ensemble.velocity_verlet_step_by(0.001);
        if i % 100 == 0 {
            ensemble.write(&mut buf);
        }
    }
}


// Bolhuis 1998
// Masse auf 1 gesetzt

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Particle{
    x: f64,
    y: f64,
    p_x: f64,
    p_y: f64,
}

impl Particle
{

    pub fn kinetic_energy(&self) -> f64
    {
        let impuls = (self.p_x * self.p_x + self.p_y * self.p_y).sqrt();
        impuls * impuls
    }

    /// The potential energy resulting from the Lennard jones potential
    pub fn potential_energy(&self, other: &Self) -> f64
    {
        let x = self.x - other.x;
        let y = self.y - other.y;
        let r_2 = x*x + y*y;

        let recip_r_2 = r_2.recip();

        let recip_r_6 = recip_r_2 * recip_r_2 * recip_r_2;
        let recip_r_12 = recip_r_6 * recip_r_6;

        4.0 * (recip_r_12 - recip_r_6)
    }

    pub fn deriviative(&self, other: &Self) -> [f64; 2]
    {
        let x_derivative = derivative(self.x, other.x, self.y, other.y);
        let y_derivative = derivative(self.y, other.y, self.x, other.x);
        [x_derivative, y_derivative]
    }
}

fn derivative(x1: f64, x2: f64, y1: f64, y2: f64) -> f64
{
    // https://www.wolframalpha.com/input?i=d%2Fda+%281%2F%28%28%28a-x2%29%5E2%2B%28y1-y2%29%5E2%29%5E%283%29%29+-+1%2F%28%28%28a-x2%29%5E2%2B%28y1-y2%29%5E2%29%5E%286%29%29%29+
    let x_diff = x1 - x2;
    let x_diff_sq = x_diff * x_diff;
    let y_diff = y1 - y2;
    let y_diff_sq = y_diff * y_diff;
    let x_diff_sq_plus_y_diff_sq = x_diff_sq + y_diff_sq;
    let x_diff_sq_plus_y_diff_sq_pow2 = x_diff_sq_plus_y_diff_sq * x_diff_sq_plus_y_diff_sq;
    let oben = 6.0*x_diff * (2.0 -x_diff_sq_plus_y_diff_sq_pow2 * x_diff_sq_plus_y_diff_sq);
    let unten = x_diff_sq_plus_y_diff_sq_pow2 * x_diff_sq_plus_y_diff_sq_pow2 * x_diff_sq_plus_y_diff_sq_pow2 * x_diff_sq_plus_y_diff_sq;
    4.0 * oben / unten
}

#[derive(Clone, Debug)]
pub struct Ensemble{
    particles: Vec<Particle>,
    current_time: f64
}

impl Ensemble{
     pub fn new(r: f64) -> Self {
        let mut particles = Vec::with_capacity(7);
        // center particle
        particles.push(Particle {
            x: 0.0,
            y: 0.0,
            p_x: 0.0,
            p_y: 0.0,
        });
        let two_pi = 2.0 * std::f64::consts::PI;
        let step = two_pi / 6.0;
        particles.extend((0..6).map(|i| {
            let theta = (i as f64) * step;
            Particle {
                x: r * theta.cos(),
                y: r * theta.sin(),
                p_x: 0.0,
                p_y: 0.0,
            }
        }));
        Ensemble { particles, current_time: 0.0 }
    }

    pub fn hamiltonian(&self) -> f64
    {
        self.kinetic_energy() + self.potential_energy()
    }

    pub fn kinetic_energy(&self) -> f64
    {
        self.particles.iter().map(Particle::kinetic_energy).sum()
    }

    pub fn potential_energy(&self) -> f64
    {
        let mut particle_slice = self.particles.as_slice();
        let mut energy_sum = 0.0;
        while let Some((first_particle, rest)) = particle_slice.split_first() {
            energy_sum += rest.iter()
                .map(
                    |other_particle|
                    {
                        first_particle.potential_energy(other_particle)
                    }
                ).sum::<f64>();
            particle_slice = rest;
        }

        energy_sum
    }

    // Ableitung Ort
    pub fn dr_dt(&self) -> impl Iterator<Item = f64>
    {
        self.particles
            .iter()
            .flat_map(
                |particle|
                {
                    [particle.p_x, particle.p_y]
                }
            )
    }

    fn forces(&self) -> Vec<f64>
    {
        // Ableitung der impulse als summe aller paar-ableitungen berechnen
        let mut impuls_ableitung: Vec<_> = (0..self.particles.len())
            .flat_map(
                |i|
                {
                    let left = &self.particles[0..i];
                    let this = &self.particles[i];
                    let right = self
                        .particles
                        .get(i+1..);
                    let right_iter = right
                        .iter()
                        .flat_map(|&particles| particles);

                    let mut derivative = [0.0; 2];

                    for particle in left.iter().chain(right_iter) {
                        let d = this.deriviative(particle);
                        derivative[0] += d[0];
                        derivative[1] += d[1];
                    }

                    derivative
                }
            ).collect();
        // Noch mit -1 multiplizieren, weil dp = - dH/dr (Hamilton)
        // Anmerkung: Irgendwo anders scheine ich einen Vorzeichenfehler zu haben, darum habe ich diese Zeile
        // auskommentiert. Ist vermutlich sogar effizienter XD 
        //impuls_ableitung.iter_mut().for_each(|entry| *entry = -*entry);

        impuls_ableitung
    }

    // Nun velocity verlet: https://www.thp.uni-koeln.de/trebst/PracticalCourse/molecular_dynamics/molecular_dynamics.pdf
    fn velocity_verlet_step_by(&mut self, delta_time: f64)
    {
        // Nun habe ich alle ableitungen. 
        let current_forces: Vec<_> = self.forces();

        let delta_time_sq_div_2 = delta_time * delta_time * 0.5;

        // update location
        self.particles
            .iter_mut()
            .zip(current_forces.chunks_exact(2))
            .for_each(
                |(particle, force)|
                {
                    particle.x += delta_time * particle.p_x + delta_time_sq_div_2 * force[0];
                    particle.y += delta_time * particle.p_y + delta_time_sq_div_2 * force[1];
                }
            );

        let future_forces = self.forces();

        // Update impulses
        self.particles
            .iter_mut()
            .zip(current_forces.chunks_exact(2))
            .zip(future_forces.chunks_exact(2))
            .for_each(
                |((particle, current_force), future_force)|
                {
                    particle.p_x += delta_time_sq_div_2 * (current_force[0] + future_force[0]);
                    particle.p_y += delta_time_sq_div_2 * (current_force[1] + future_force[1]);
                }
            );

        self.current_time += delta_time;

    }


    pub fn write<W: Write>(&self, mut writer: W)
    {
        write!(writer, "{} ", self.current_time).unwrap();
        for particle in self.particles.iter(){
            write!(writer, "{} {} ", particle.x, particle.y).unwrap();
        }
        writeln!(writer).unwrap()
    }
}