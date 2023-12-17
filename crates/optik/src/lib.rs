use std::{
    cell::RefCell,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use nalgebra::{DVector, DVectorView, Isometry3, Matrix6xX};
use ordered_float::OrderedFloat;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::{
    prelude::{IntoParallelIterator, ParallelIterator},
    ThreadPoolBuilder,
};

mod config;
pub mod kinematics;
pub mod math;
pub mod objective;

pub use config::*;
use kinematics::*;
use objective::*;
use slsqp_sys::{IterationResult, SlsqpSolver};

pub fn set_parallelism(n: usize) {
    ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .unwrap()
}

#[derive(Clone)]
pub struct Robot {
    chain: KinematicChain,
}

impl Robot {
    pub fn new(chain: KinematicChain) -> Self {
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
        let chain = KinematicChain::from_urdf(urdf, base_link, ee_link);
        Robot::new(chain)
    }
}

impl Robot {
    pub fn num_positions(&self) -> usize {
        self.chain.nq()
    }

    pub fn joint_limits(&self) -> (DVector<f64>, DVector<f64>) {
        let (lb, ub) = self.chain.joints.iter().map(|j| j.limits).unzip();

        (DVector::from_vec(lb), DVector::from_vec(ub))
    }

    pub fn random_configuration(&self, rng: &mut impl rand::Rng) -> Vec<f64> {
        let (lb, ub) = self.joint_limits();
        (0..self.num_positions())
            .map(|i| rng.gen_range(lb[i]..=ub[i]))
            .collect()
    }

    pub fn joint_jacobian(&self, fk: &ForwardKinematics) -> Matrix6xX<f64> {
        self.chain.joint_jacobian(fk)
    }

    pub fn fk(&self, q: &[f64]) -> Isometry3<f64> {
        *self.chain.forward_kinematics(q).ee_tfm()
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
        let solution_stream = (0..max_restarts).into_par_iter().flat_map(|i| {
            // Fix a global RNG seed, which is used to compute sub-seeds for each thread.
            const RNG_SEED: u64 = 42;

            let mut rng = ChaCha8Rng::seed_from_u64(RNG_SEED);
            rng.set_stream(i);

            let cache = RefCell::new(KinematicsCache::default());
            let args = ObjectiveArgs {
                robot: self,
                config,
                tfm_target,
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
            let mut f_prev = objective(&x, &args, &mut cache.borrow_mut());

            // Iterate the solver until any of:
            // - The solver converges within the active convergence criteria
            // - Another thread arrives at a solution (in speed mode)
            // - The specified timeout expires
            while !should_exit.load(Ordering::Relaxed) && !is_timed_out() {
                match solver.iterate(
                    &mut x,
                    |x| objective(x, &args, &mut cache.borrow_mut()),
                    |x, g| objective_grad(x, g, &args, &mut cache.borrow_mut()),
                ) {
                    IterationResult::Continue => {
                        x_prev.copy_from_slice(&x);
                        f_prev = solver.cost();

                        continue;
                    }
                    IterationResult::Converged => {
                        // The SLSQP solver can report convergence
                        // regardless of whether our tol_f, tol_df, or
                        // tol_dx conditions are met. If we verify that this
                        // solution does meet the criteria then it can be
                        // returned.
                        //
                        // This is due to an internal `accuracy` parameter,
                        // which reports convergence if the change in
                        // objective function falls below it.
                        let df = solver.cost() - f_prev;
                        let dx = DVectorView::from_slice(&x, x.len())
                            - DVectorView::from_slice(&x_prev, x_prev.len());

                        // Report convergence if _any_ of the active
                        // convergence criteria are met.
                        if solver.cost().abs() < config.tol_f
                            || (config.tol_df > 0.0 && df.abs() < config.tol_df)
                            || (config.tol_dx > 0.0 && dx.norm().abs() < config.tol_dx)
                        {
                            // Short-circuit any other threads for a modest
                            // performance increase.
                            //
                            // In `Speed` mode, if the current thread has
                            // converged on a satisfactory solution then we
                            // don't care what the others are going to
                            // produce.
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
        });

        match config.solution_mode {
            SolutionMode::Quality => {
                // Continue solving until the timeout is reached and take the
                // best of all solutions. In this case, the cost of a given
                // solution is computed as its distance from the seed.
                solution_stream.min_by_key(|(x, _)| {
                    let x = DVectorView::from_slice(x, x.len());
                    let x0 = DVectorView::from_slice(&x0, x0.len());

                    OrderedFloat(x.metric_distance(&x0))
                })
            }
            SolutionMode::Speed => {
                // Take the first solution which satisfies the tolerance.
                solution_stream.find_any(|_| true)
            }
        }
    }
}
