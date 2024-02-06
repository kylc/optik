use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use nalgebra::{DVectorView, Isometry3, Matrix6xX};
use nlopt::{Algorithm, Nlopt, SuccessState, Target};
use ordered_float::OrderedFloat;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::{
    prelude::{IntoParallelIterator, ParallelIterator},
    ThreadPool, ThreadPoolBuilder,
};

mod config;
pub mod kinematics;
pub mod math;
pub mod objective;

pub use config::*;
use kinematics::*;
use objective::*;

pub struct Robot {
    chain: KinematicChain,
    thread_pool: ThreadPool,
}

impl Robot {
    pub fn new(chain: KinematicChain) -> Self {
        Self {
            chain,
            thread_pool: ThreadPoolBuilder::default().build().unwrap(),
        }
    }

    pub fn from_urdf(urdf: &urdf_rs::Robot, base_link: &str, ee_link: &str) -> Self {
        let chain = KinematicChain::from_urdf(urdf, base_link, ee_link);
        Robot::new(chain)
    }

    pub fn from_urdf_file(path: impl AsRef<Path>, base_link: &str, ee_link: &str) -> Self {
        let urdf = urdf_rs::read_file(path).expect("error parsing URDF file!");
        Robot::from_urdf(&urdf, base_link, ee_link)
    }

    pub fn from_urdf_str(urdf: &str, base_link: &str, ee_link: &str) -> Self {
        let urdf = urdf_rs::read_from_string(urdf).expect("error parsing URDF file!");
        Robot::from_urdf(&urdf, base_link, ee_link)
    }
}

impl Robot {
    pub fn set_parallelism(&mut self, n: usize) {
        // Don't rebuild the thread pool (an expensive operation) unless this
        // value has actually changed.
        if n != self.thread_pool.current_num_threads() {
            self.thread_pool = ThreadPoolBuilder::new().num_threads(n).build().unwrap()
        }
    }

    pub fn num_positions(&self) -> usize {
        self.chain.num_positions()
    }

    pub fn joint_limits(&self) -> (Vec<f64>, Vec<f64>) {
        self.chain
            .joints
            .iter()
            .flat_map(|j| j.limits.clone())
            .unzip()
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

    pub fn fk(&self, q: &[f64]) -> ForwardKinematics {
        self.chain.forward_kinematics(q)
    }

    pub fn ik(
        &self,
        config: &SolverConfig,
        tfm_target: &Isometry3<f64>,
        x0: Vec<f64>,
    ) -> Option<(Vec<f64>, f64)> {
        // Complain if the provided seed is out side the joint limits. The
        // solver may be able to handle this, but it seems likely that there is
        // a bug in the user's program if this occurs and we should notify them.
        let (lb, ub) = self.joint_limits();
        if x0.iter().enumerate().any(|(i, q)| *q < lb[i] || *q > ub[i]) {
            panic!("seed joint position outside of joint limits")
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

        // HEURISTIC: Establish an artifical stopping criteria if the
        // optimization seems to have stalled. This case will not be considered
        // a successful result, but it will free up this thread to restart from
        // another seed.
        let tol_df = if config.tol_df > 0.0 {
            config.tol_df
        } else {
            // Stop when the gradient is a few orders of magnitude smaller than
            // the requested precision. When this happens, the optimization is
            // most likely stuck in a local minimum.
            //
            // We can't make this too big w.r.t. tol_f or we will exit in cases
            // that are still on track to converge to a valid solution.
            1e-3 * config.tol_f
        };

        // Build a parallel stream of solutions from which we can choose how to
        // draw a final result.
        self.thread_pool.install(|| {
            let solution_stream = (0..max_restarts)
                .into_par_iter()
                .panic_fuse()
                .map(|i| {
                    let mut optimizer = Nlopt::new(
                        Algorithm::Slsqp,
                        self.num_positions(),
                        |x: &[f64], grad: Option<&mut [f64]>, fk: &mut ForwardKinematics| {
                            // Early-exit if we are out of time, or if another
                            // thread has already found a satisfactory solution.
                            if is_timed_out() || should_exit.load(Ordering::Relaxed) {
                                return None;
                            }

                            // Share kinematic results between the gradient and
                            // objective function evaluations, when possible.
                            self.chain.forward_kinematics_mut(x, fk);

                            // Compute the gradient only if it was requested by
                            // the optimizer.
                            if let Some(g) = grad {
                                objective_grad(self, tfm_target, fk, g);
                            }

                            // Always compute the objective value.
                            Some(objective(self, tfm_target, fk))
                        },
                        Target::Minimize,
                        // Cache the forward kinematic container within each
                        // thread to avoid re-allocating memory each objective
                        // iteration.
                        ForwardKinematics::default(),
                    );

                    optimizer.set_stopval(config.tol_f).unwrap();
                    optimizer.set_ftol_abs(tol_df).unwrap();
                    optimizer.set_xtol_abs1(config.tol_dx).unwrap();
                    optimizer.set_lower_bounds(lb.as_slice()).unwrap();
                    optimizer.set_upper_bounds(ub.as_slice()).unwrap();

                    // LBFGS memory size is chosen empirically. This value seems
                    // to give the best performance for ~6 DoF robot arms.
                    const LBFGS_STORAGE_SIZE: usize = 10;
                    optimizer
                        .set_vector_storage(Some(LBFGS_STORAGE_SIZE))
                        .unwrap();

                    // Fix a global RNG seed, which is used to compute sub-seeds
                    // for each thread.
                    const RNG_SEED: u64 = 42;
                    let mut rng = ChaCha8Rng::seed_from_u64(RNG_SEED);
                    rng.set_stream(i);

                    // The first attempt gets the initial seed provided by the
                    // caller.  All other attempts start at some random point.
                    let mut x = if i == 0 {
                        x0.clone()
                    } else {
                        self.random_configuration(&mut rng)
                    };

                    if let Ok((r, c)) = optimizer.optimize(&mut x) {
                        // Make sure that we exited for the right reasons. For
                        // example, NLopt considers a timeout to be a success
                        // but we treat it as a failure.
                        let success = (config.tol_f >= 0.
                            && matches!(r, SuccessState::StopValReached))
                            || (config.tol_df >= 0. && matches!(r, SuccessState::FtolReached))
                            || (config.tol_dx >= 0. && matches!(r, SuccessState::XtolReached));

                        if success {
                            if config.solution_mode == SolutionMode::Speed {
                                should_exit.store(true, Ordering::Relaxed);
                            }

                            return Some((x, c));
                        }
                    }

                    None
                })
                // Stop issuing new solve requests once we run out of time.
                .take_any_while(|_| !is_timed_out())
                // Ignore failed solve attemps.
                .flatten();

            match config.solution_mode {
                SolutionMode::Quality => {
                    // Continue solving until the timeout is reached and take
                    // the best of all solutions. In this case, the cost of a
                    // given solution is computed as its distance from the seed.
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
        })
    }
}
