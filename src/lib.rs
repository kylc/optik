use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use float_ord::FloatOrd;
use k::{joint::Range, SerialChain};
use nalgebra::{DMatrix, DVector, Isometry3};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use slsqp_sys::*;

mod config;
mod math;
mod py;

pub use config::*;
use math::*;

#[derive(Clone)]
pub struct Robot {
    chain: SerialChain<f64>,
}

impl Robot {
    pub fn new(chain: SerialChain<f64>) -> Self {
        Self { chain }
    }
}

impl Robot {
    pub fn fk(&self, q: &[f64]) -> Isometry3<f64> {
        self.chain.set_joint_positions_unchecked(q);
        self.chain.end_transform()
    }

    pub fn jacobian_local(&self, q: &[f64]) -> DMatrix<f64> {
        let t_n = self.fk(q); // this updates the joint positions

        let mut m = k::jacobian(&self.chain);

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

    pub fn joint_limits(&self) -> (DVector<f64>, DVector<f64>) {
        let unlimited = Range::new(f64::NEG_INFINITY, f64::INFINITY);

        let (lb, ub) = self
            .chain
            .iter_joints()
            .map(|j| j.limits.unwrap_or(unlimited))
            .map(|l| (l.min, l.max))
            .unzip();

        (DVector::from_vec(lb), DVector::from_vec(ub))
    }

    pub fn random_configuration(&self, rng: &mut impl rand::Rng) -> Vec<f64> {
        let (lb, ub) = self.joint_limits();
        (0..self.chain.dof())
            .map(|i| rng.gen_range(lb[i]..=ub[i]))
            .collect()
    }

    pub fn ik(
        &self,
        config: &SolverConfig,
        tfm_target: &Isometry3<f64>,
        x0: Vec<f64>,
    ) -> Option<(Vec<f64>, f64)> {
        let (lb, ub) = self.joint_limits();

        // Compute the time at which the user-specified timeout will expire. We
        // will ensure that no solve threads continue iterating beyond this
        // time.
        let start_time = Instant::now();
        let end_time = start_time + Duration::from_secs_f64(config.max_time);

        // Fix a global RNG seed, which is used to compute sub-seeds for each thread.
        const RNG_SEED: u64 = 42;

        // In SolutionMode::Speed, when one thread finds a solution which
        // satisfies the tolerances, it will immediately tell all of the other
        // threads to exit.
        let should_exit = Arc::new(AtomicBool::new(false));

        // Build a parallel stream of solutions from which we can choose how to
        // draw a final result.
        let solution_stream = (0..u64::MAX)
            .into_par_iter()
            .map(|i| {
                let mut rng = ChaCha8Rng::seed_from_u64(RNG_SEED);
                rng.set_stream(i);

                let args = ObjectiveArgs {
                    robot: self.clone(),
                    config: config.clone(),
                    tfm_target: *tfm_target,
                };

                // The first attempt gets the initial seed provided by the caller.
                // All other attempts start at some random point.
                let mut x = if i == 0 {
                    x0.clone()
                } else {
                    self.random_configuration(&mut rng)
                };

                let mut solver = SlsqpSolver::new(x.len());
                solver.set_ftol(config.ftol_abs);
                solver.set_dxtol(config.xtol_abs);
                solver.set_lb(lb.as_slice());
                solver.set_ub(ub.as_slice());

                // Iterate the soler until any of:
                // - The solver converges within the tolerance
                // - Another thread signals that it has converged
                // - The timeout expires
                while solver.iterate(
                    &mut x,
                    |x| objective(x, &args),
                    |x, g| objective_grad(x, g, &args),
                ) == IterationResult::Continue
                    && !should_exit.load(Ordering::Relaxed)
                    && Instant::now() < end_time
                {}

                // TODO: Don't re-evaluate the objective function here. It was
                // already done in the last iteration of the solver.
                let f = objective(&x, &args);
                if f < config.ftol_abs {
                    // Short-circuit any other threads before we return for a
                    // modest performance improvement.
                    if config.solution_mode == SolutionMode::Speed {
                        should_exit.store(true, Ordering::Relaxed);
                    }

                    Some((x, f))
                } else {
                    None
                }
            })
            .take_any_while(|_| Instant::now() < end_time)
            .flatten();

        match config.solution_mode {
            SolutionMode::Quality => {
                // Continue solving until the timeout is reached and take the best of all
                // solutions.
                solution_stream.min_by_key(|&(_, obj)| FloatOrd(obj))
            }
            SolutionMode::Speed => {
                // Take the first solution which satisfies the tolerance.
                solution_stream.find_any(|&(_, obj)| obj < config.ftol_abs)
            }
        }
    }
}

#[derive(Clone)]
pub struct ObjectiveArgs {
    pub robot: Robot,
    pub config: SolverConfig,
    pub tfm_target: Isometry3<f64>,
}

pub fn objective(x: &[f64], args: &ObjectiveArgs) -> f64 {
    let tfm_actual = args.robot.fk(x);
    let tfm_target = args.tfm_target;
    let tfm_error = tfm_target.inverse() * tfm_actual;

    se3::log(tfm_error).norm_squared()
}

pub fn objective_grad(x: &[f64], g: &mut [f64], args: &ObjectiveArgs) {
    let robot = &args.robot;
    let tfm_actual = &args.robot.fk(x);
    let tfm_target = &args.tfm_target;

    match args.config.gradient_mode {
        GradientMode::Analytical => {
            let tfm_error = tfm_target.inv_mul(tfm_actual);

            let j_q = robot.jacobian_local(x);
            let j_log_se3_r = se3::right_jacobian(tfm_error);
            let j = (j_log_se3_r * j_q).transpose(); // Jq' * Jr' = (Jr * Jq)'

            let q = 2.0 * j * se3::log(tfm_error);
            g.copy_from_slice(q.as_slice());
        }
        GradientMode::Numerical => {
            let n = x.len();
            let mut x0 = x.to_vec();
            let eps = f64::EPSILON.powf(1.0 / 3.0);
            for i in 0..n {
                let x0i = x0[i];
                x0[i] = x0i - eps;
                let fl = objective(&x0, args);
                x0[i] = x0i + eps;
                let fh = objective(&x0, args);
                g[i] = (fh - fl) / (2.0 * eps);
                x0[i] = x0i;
            }
        }
    }
}
