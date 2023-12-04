use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use float_ord::FloatOrd;
use k::{joint::Range, Chain, SerialChain};
use nalgebra::{DMatrix, DVector, DVectorSlice, Isometry3};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::{
    prelude::{IntoParallelIterator, ParallelIterator},
    ThreadPoolBuilder,
};
use slsqp_sys::*;

mod config;
pub mod math;

pub use config::*;
use math::*;

pub fn set_parallelism(n: usize) {
    ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .unwrap()
}

#[derive(Clone)]
pub struct Robot {
    pub chain: SerialChain<f64>,
}

impl Robot {
    pub fn new(chain: SerialChain<f64>) -> Self {
        Self { chain }
    }

    pub fn from_urdf_file(path: impl AsRef<Path>, base_link: &str, ee_link: &str) -> Self {
        let urdf = urdf_rs::read_file(path).expect("error parsing URDF file!");
        Robot::from_urdf(&urdf, base_link, ee_link)
    }

    pub fn from_urdf_str(urdf: &str, base_link: &str, ee_link: &str) -> Self {
        let urdf = urdf_rs::read_from_string(urdf).expect("error parsing URDF file!");
        Robot::from_urdf(&urdf, base_link, ee_link)
    }

    pub fn from_urdf(urdf: &urdf_rs::Robot, base_link: &str, ee_link: &str) -> Self {
        let chain = Chain::<f64>::from(urdf);

        let base_link = chain
            .find_link(base_link)
            .unwrap_or_else(|| panic!("link '{}' does not exist!", base_link));
        let ee_link = chain
            .find_link(ee_link)
            .unwrap_or_else(|| panic!("link '{}' does not exist!", ee_link));

        let serial = SerialChain::from_end_to_root(ee_link, base_link);
        Robot::new(serial)
    }

    pub fn jacobian_local(&self, q: &[f64]) -> DMatrix<f64> {
        let t_n = self.fk(q); // this updates the joint positions

        let mut m = k::jacobian(&self.chain);

        // K computes a Jacobian J(q) in Pinocchio's terms as
        // LOCAL_WORLD_ALIGNED.  Because we compute the right Jacobian of log
        // SE(3) J_r(X), we prefer to work in body frame.  Convert J(q) into the
        // local body frame (Pinocchio calls this LOCAL frame).
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

    pub fn num_positions(&self) -> usize {
        self.chain.dof()
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

    pub fn fk(&self, q: &[f64]) -> Isometry3<f64> {
        self.chain.set_joint_positions_unchecked(q);
        self.chain.end_transform()
    }

    pub fn ik(
        &self,
        config: &SolverConfig,
        tfm_target: &Isometry3<f64>,
        mut x0: Vec<f64>,
    ) -> Option<(Vec<f64>, f64)> {
        // Project into the joint limits.
        let (lb, ub) = self.joint_limits();
        for i in 0..self.num_positions() {
            x0[i] = x0[i].clamp(lb[i], ub[i]);
        }

        // Fix a global RNG seed, which is used to compute sub-seeds for each thread.
        const RNG_SEED: u64 = 42;

        // Compute the time at which the user-specified timeout will expire. We
        // will ensure that no solve threads continue iterating beyond this
        // time. If no max time is specified then run until the retry count is
        // exhausted.
        let start_time = Instant::now();
        let is_timed_out = || {
            let elapsed_time = Instant::now().duration_since(start_time).as_secs_f64();
            config.max_time > 0.0 && elapsed_time > config.max_time
        };

        // In SolutionMode::Speed, when one thread finds a solution which
        // satisfies the tolerances, it will immediately tell all of the other
        // threads to exit.
        let should_exit = Arc::new(AtomicBool::new(false));

        // If a maximum number of restarts is specified then we limit to that.
        // Otherwise, limit to a huge number.
        let max_restarts = if config.max_restarts > 0 {
            config.max_restarts
        } else {
            u64::MAX
        };

        // Build a parallel stream of solutions from which we can choose how to
        // draw a final result.
        let solution_stream = (0..max_restarts)
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
                solver.set_tol_f(config.tol_f);
                solver.set_tol_df(config.tol_df);
                solver.set_tol_dx(config.tol_dx);
                solver.set_lb(lb.as_slice());
                solver.set_ub(ub.as_slice());

                // Bookkeeping for stopping criteria.
                let mut x_prev = x.clone();
                let mut f_prev = objective(&x, &args);

                // Iterate the soler until any of:
                // - The solver converges within the tolerance
                // - Another thread signals that it has converged
                // - The timeout expires
                while !should_exit.load(Ordering::Relaxed) && !is_timed_out() {
                    match solver.iterate(
                        &mut x,
                        |x| objective(x, &args),
                        |x, g| objective_grad(x, g, &args),
                    ) {
                        IterationResult::Continue => {
                            x_prev.copy_from_slice(&x);
                            f_prev = solver.cost();

                            continue;
                        }
                        IterationResult::Converged => {
                            // The SLSQP solver can report convergence
                            // regardless of whether our tol_f, tol_df, nor
                            // tol_dx conditions are met. If we verify that this
                            // solution does meet the criteria then it can be
                            // returned.
                            let df = solver.cost() - f_prev;
                            let dx = DVector::from_row_slice(&x) - DVector::from_row_slice(&x_prev);

                            if solver.cost().abs() < config.tol_f
                                || (config.tol_df > 0.0 && df.abs() < config.tol_df)
                                || (config.tol_dx > 0.0 && dx.norm().abs() < config.tol_dx)
                            {
                                // Short-circuit any other threads for a modest
                                // performance increase.
                                if config.solution_mode == SolutionMode::Speed {
                                    should_exit.store(true, Ordering::Relaxed);
                                }

                                return Some((x, solver.cost()));
                            }

                            return None;
                        }
                        IterationResult::Error => return None,
                    }
                }

                None
            })
            .take_any_while(|_| !is_timed_out())
            .flatten();

        match config.solution_mode {
            SolutionMode::Quality => {
                // Continue solving until the timeout is reached and take the
                // best of all solutions. In this case, the cost of a given
                // solution is computed as its distance from the seed.
                solution_stream.min_by_key(|(x, _)| {
                    let x = DVectorSlice::from_slice(x, self.num_positions());
                    let x0 = DVectorSlice::from_slice(&x0, self.num_positions());

                    FloatOrd(x.metric_distance(&x0))
                })
            }
            SolutionMode::Speed => {
                // Take the first solution which satisfies the tolerance.
                solution_stream.find_any(|_| true)
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

pub fn objective(q: &[f64], args: &ObjectiveArgs) -> f64 {
    // Compute the pose error w.r.t. the actual pose:
    //
    //   X_AT = X_WA^1 * X_WT
    let tfm_actual = args.robot.fk(q);
    let tfm_target = args.tfm_target;
    let tfm_error = tfm_target.inv_mul(&tfm_actual);

    // Minimize the sqaure Euclidean distance of the log pose error. We choose
    // to use the square distance due to its smoothness.
    se3::log(&tfm_error).norm_squared()
}

/// Compute the gradient `g` w.r.t. the local parameterization.
pub fn objective_grad(q: &[f64], g: &mut [f64], args: &ObjectiveArgs) {
    let robot = &args.robot;
    let tfm_actual = &args.robot.fk(q);
    let tfm_target = &args.tfm_target;

    match args.config.gradient_mode {
        GradientMode::Analytical => {
            // Pose error is computed as in the objective function.
            let tfm_error = tfm_target.inv_mul(tfm_actual);

            // We compute the Jacobian of our task w.r.t. the joint angles.
            //
            // - Jqdot: the local (body-coordinate) frame Jacobian of X w.r.t. q
            // - Jlog6: the right (body-coordinate) Jacobian of log6(X)
            // - Jtask: the Jacobian of our pose tracking task
            //
            //   Jtask(q) = Jlog6(X) * J(q)
            let j_qdot = robot.jacobian_local(q);
            let j_log6 = se3::right_jacobian(&tfm_error);
            let j_task = j_log6 * j_qdot;

            // We must compute the objective function gradient:
            //
            //   ∇h = [ ∂h/∂x1  ...  ∂h/∂xn ]
            //
            // Given the pose task Jacobian of the form:
            //
            //   J = ⎡ ∂f1/∂x1 ... ∂f1/∂xn ⎤
            //       ⎢    .           .    ⎥
            //       ⎢    .           .    ⎥
            //       ⎣ ∂fm/∂x1 ... ∂fm/∂xn ⎦
            //
            // Apply the chain rule to compute the derivative:
            //
            //   g = log(X)
            //   f = || g ||^2
            //   h' = (f' ∘ g) * g' = (2.0 * log6(X)) * J
            let fdot_g = 2.0 * se3::log(&tfm_error).transpose();
            let grad_h = fdot_g * j_task;

            g.copy_from_slice(grad_h.as_slice());
        }
        GradientMode::Numerical => {
            // Central finite difference:
            //
            //   ∇h = (f(x + Δx) - f(x - Δx)) / Δx
            let n = q.len();
            let mut x0 = q.to_vec();
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
