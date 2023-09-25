use std::time::{Duration, Instant};

use float_ord::FloatOrd;
use k::{joint::Range, SerialChain};
use nalgebra::{DMatrix, DVector, Isometry3};
use nlopt::{approximate_gradient, Nlopt, ObjFn};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

mod config;
mod math;
mod py;

pub use config::*;
use math::*;

#[derive(Clone)]
pub struct Robot {
    serial_chain: SerialChain<f64>,
}

impl Robot {
    pub fn new(serial_chain: SerialChain<f64>) -> Self {
        Self { serial_chain }
    }
}

impl Robot {
    pub fn fk(&self, q: &[f64]) -> Isometry3<f64> {
        self.serial_chain.set_joint_positions_unchecked(q);
        self.serial_chain.end_transform()
    }

    pub fn jacobian_local(&self, q: &[f64]) -> DMatrix<f64> {
        let t_n = self.fk(q); // this updates the joint positions

        let mut m = k::jacobian(&self.serial_chain);

        // K computes a Jacobian J(q) in Pinocchio's terms as
        // LOCAL_WORLD_ALIGNED.  Because we compute the compute the right
        // Jacobian of log SE(3) J_r(X), we prefer to work in body frame.
        // Convert J(q) into the local body frame (Pinocchio calls this LOCAL
        // frame).
        let w_inv = t_n.rotation.inverse();
        for mut col in m.column_iter_mut() {
            let mut linear = col.fixed_rows_mut::<3>(0);
            let linear_w = w_inv * &linear;
            linear.copy_from(&linear_w);

            let mut angular = col.fixed_rows_mut::<3>(3);
            let angular_w = w_inv * &angular;
            angular.copy_from(&angular_w);
        }

        m
    }

    pub fn bounds(&self) -> (DVector<f64>, DVector<f64>) {
        let unlimited = Range::new(f64::NEG_INFINITY, f64::INFINITY);

        let (lb, ub) = self
            .serial_chain
            .iter_joints()
            .map(|j| j.limits.unwrap_or(unlimited))
            .map(|l| (l.min, l.max))
            .unzip();

        (DVector::from_vec(lb), DVector::from_vec(ub))
    }

    pub fn random_configuration(&self, rng: &mut impl rand::Rng) -> Vec<f64> {
        let (lb, ub) = self.bounds();
        let mut q = vec![0.0; self.serial_chain.dof()];
        for i in 0..6 {
            q[i] = rng.gen_range(lb[i]..=ub[i])
        }

        q
    }

    #[allow(non_snake_case)] // math symbols :)
    pub fn ee_error_grad(
        &self,
        tfm_target: &Isometry3<f64>,
        tfm_actual: &Isometry3<f64>, // = self.fk(q), saves a recomputation
        q: &[f64],
        grad_mode: GradientMode,
    ) -> DVector<f64> {
        match grad_mode {
            GradientMode::Analytical => {
                let tfm_error = tfm_target.inv_mul(tfm_actual);

                let Jq = self.jacobian_local(q);
                let Jlogr = se3::right_jacobian(tfm_error);
                let J = (Jlogr * Jq).transpose(); // Jq' * Jr' = (Jr * Jq)'

                2.0 * J * se3::log(tfm_error)
            }
            GradientMode::Numerical => {
                let mut g = [0.0; 6];
                approximate_gradient(
                    q,
                    |x: &[f64]| {
                        let tfm_actual = self.fk(x);
                        se3::log(tfm_target.inverse() * tfm_actual).norm()
                    },
                    &mut g,
                );
                DVector::from_row_slice(&g)
            }
        }
    }
}

pub fn objective(
    x: &[f64],
    grad: Option<&mut [f64]>,
    (tfm_target, robot): &mut (Isometry3<f64>, Robot),
) -> f64 {
    let tfm_actual = robot.fk(x);
    let tfm_error = tfm_target.inverse() * tfm_actual;

    if let Some(g) = grad {
        let grad = robot.ee_error_grad(tfm_target, &tfm_actual, x, GradientMode::Analytical);
        g.copy_from_slice(grad.as_slice());
    }

    se3::log(tfm_error).norm_squared()
}

pub fn solve(
    robot: &Robot,
    config: &SolverConfig,
    tfm_target: &Isometry3<f64>,
    x0: Vec<f64>,
) -> (Option<Vec<f64>>, f64) {
    let (lb, ub) = robot.bounds();

    let max_time = Duration::from_secs_f64(config.max_time);
    let start_time = Instant::now();

    // TODO: We can probably use the `rayon::ThreadPoolBuilder` to keep a
    // thread-local Nlopt instance, instead of recreating it for every
    // iteration.

    const RNG_SEED: u64 = 42;
    const MAX_ITERATIONS: u64 = 1000;

    let solution_stream = (0..MAX_ITERATIONS)
        .into_par_iter()
        .map(|i| {
            Nlopt::<Box<dyn ObjFn<()>>, ()>::srand_seed(Some(RNG_SEED));

            let mut rng = ChaCha8Rng::seed_from_u64(RNG_SEED);
            rng.set_stream(i);

            let mut opt = Nlopt::new(
                nlopt::Algorithm::Slsqp,
                robot.serial_chain.dof(),
                objective,
                nlopt::Target::Minimize,
                (*tfm_target, robot.clone()),
            );
            opt.set_xtol_abs1(config.xtol_abs).unwrap();
            opt.set_lower_bounds(lb.as_slice()).unwrap();
            opt.set_upper_bounds(ub.as_slice()).unwrap();
            opt.set_maxtime(config.max_time).unwrap();

            // The first attempt gets the initial seed provided by the caller.
            // All other attempts start at some random point.
            let mut x = if i == 0 {
                x0.clone()
            } else {
                robot.random_configuration(&mut rng)
            };

            let res = opt.optimize(&mut x);

            if let Ok((_, cost)) = res {
                (Some(x), cost)
            } else {
                (None, f64::INFINITY)
            }
        })
        .take_any_while(|_| (Instant::now() - start_time) < max_time);

    match config.solution_mode {
        SolutionMode::Quality => {
            // Continue solving until the timeout is reached and take the best of all
            // solutions.
            solution_stream
                .min_by_key(|&(_, score)| FloatOrd(score))
                .unwrap()
        }
        SolutionMode::Speed => {
            // Take the first solution which satisfies the tolerance.
            solution_stream
                .find_any(|&(_, score)| score < 1e-4)
                .unwrap()
        }
    }
}
