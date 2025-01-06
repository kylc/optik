use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use clarabel::{
    algebra::{BlockConcatenate, CscMatrix},
    solver::{
        DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus,
        SupportedConeT::{self, NonnegativeConeT, ZeroConeT},
    },
};
use nalgebra::{DVectorView, Isometry3, Matrix6xX, Vector3, Vector6, stack};
use nlopt::{Algorithm, Nlopt, SuccessState, Target};
use ordered_float::OrderedFloat;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::{
    ThreadPool, ThreadPoolBuilder,
    prelude::{IntoParallelIterator, ParallelIterator},
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
            .map(|i| rng.random_range(lb[i]..=ub[i]))
            .collect()
    }

    pub fn joint_jacobian(&self, fk: &ForwardKinematics) -> Matrix6xX<f64> {
        self.chain.joint_jacobian(fk)
    }

    pub fn fk(&self, q: &[f64], ee_offset: &Isometry3<f64>) -> ForwardKinematics {
        self.chain.forward_kinematics(q, ee_offset)
    }

    /// Given a desired end-effector frame velocity Vᴱ, attempt to solve for the
    /// corresponding joint velocity v = [Jᴱ(q)]⁻¹Vᴱ. However, because the robot
    /// may be joint velocity limited, an exact solution may not be possible.
    ///
    /// We therefore formulate the following optimization problem (as in [1]):
    ///
    ///   max_{vₙ, α} α
    ///     s.t.
    ///       # A scaling factor for how much to move in the desired direction.
    ///       # We attempt to maximize this in the interval (0, 1), i.e. move as
    ///       # far as we can.
    ///       0 ≤ α ≤ 1
    ///       # Apply symmetric jointspace velocity limits.
    ///       -vmax ≤ vₙ ≤ vmax
    ///       # Movement is only allowed in the direction of the desired
    ///       # end-effector frame velocity.
    ///       Jᴱ(q)vₙ = αVᴱ
    ///
    /// If a solution is found, returns a tuple of the alpha scaling factor and
    /// the resulting joint velocities. Otherwise, returns None.
    ///
    /// [1]: https://manipulation.csail.mit.edu/pick.html#section6
    pub fn diff_ik(
        &self,
        x0: Vec<f64>,
        V_WE: &Vector6<f64>,
        v_max: &[f64],
        ee_offset: &Isometry3<f64>,
    ) -> Option<(f64, Vec<f64>)> {
        let n = self.num_positions();

        // Appends the following constraint:
        // 0 ≤ α ≤ 1.
        let append_alpha_constraints =
            |A: &mut CscMatrix<f64>, b: &mut Vec<f64>, K: &mut Vec<SupportedConeT<f64>>| {
                let mut A_ext = CscMatrix::zeros((2, A.n));

                // α ≤ 1.0
                // α + s = 1.0, s ≥ 0
                A_ext.set_entry((0, n), 1.0);
                b.push(1.0);
                K.push(NonnegativeConeT(1));

                // α >= 0
                // -α + s = 0.0, s ≥ 0
                A_ext.set_entry((1, n), -1.0);
                b.push(0.0);
                K.push(NonnegativeConeT(1));

                *A = CscMatrix::vcat(A, &A_ext).unwrap();
            };

        // Appends the following constraint:
        // -vₘₐₓ ≤ vₙ ≤ vₘₐₓ
        let append_velocity_constraints =
            |A: &mut CscMatrix<f64>, b: &mut Vec<f64>, K: &mut Vec<SupportedConeT<f64>>| {
                let mut A_ext = CscMatrix::zeros((2 * n, A.n));

                for i in 0..n {
                    // vᵢ ≤ vₘₐₓ
                    // vᵢ + s = vₘₐₓ, s ≥ 0
                    A_ext.set_entry((i * 2, i), 1.0);
                    b.push(v_max[i]);
                    K.push(NonnegativeConeT(1));

                    // vᵢ ≥ -vₘₐₓ
                    // -vᵢ + s = vₘₐₓ, s ≥ 0
                    A_ext.set_entry((i * 2 + 1, i), -1.0);
                    b.push(v_max[i]);
                    K.push(NonnegativeConeT(1));
                }

                *A = CscMatrix::vcat(A, &A_ext).unwrap();
            };

        // Appends the following constraint:
        // Jᴱ(q)vₙ = αVᴱ
        let append_cartesian_constraints =
            |A: &mut CscMatrix<f64>, b: &mut Vec<f64>, K: &mut Vec<SupportedConeT<f64>>| {
                let fk = self.fk(&x0, ee_offset);

                // Rotate the local frame end-effector Jacobian into the world frame to
                // match the input velocity frame.
                let R_WE = fk.ee_tfm().rotation.to_rotation_matrix();
                let JEq = self.joint_jacobian(&fk);
                let JEq_W = stack![
                     R_WE * JEq.fixed_rows::<3>(0);
                     R_WE * JEq.fixed_rows::<3>(3)
                ];

                // Jᴱ(q)vₙ = αVᴱ
                let A_ext = CscMatrix::from(stack![JEq_W, -V_WE].row_iter());
                b.extend(vec![0.0; n]);
                K.push(ZeroConeT(n));

                *A = CscMatrix::vcat(A, &A_ext).unwrap();
            };

        // Solve
        //   min 0.5 xᵀPx + qᵀx
        //     s.t.
        //       Ax + s = b
        //       s ∈ K
        let P = CscMatrix::zeros((n + 1, n + 1));
        let mut q = vec![0.0; n + 1];
        q[n] = -100.0; // Reward large alpha values.

        let mut A = CscMatrix::zeros((0, n + 1));
        let mut b = vec![];
        let mut K = vec![];

        append_alpha_constraints(&mut A, &mut b, &mut K);
        append_velocity_constraints(&mut A, &mut b, &mut K);
        append_cartesian_constraints(&mut A, &mut b, &mut K);

        let mut solver = DefaultSolver::new(
            &P,
            &q,
            &A,
            &b,
            &K,
            DefaultSettingsBuilder::default()
                .verbose(false)
                .build()
                .unwrap(),
        )
        .expect("solver initialization failed");
        solver.solve();

        let solution = solver.solution;
        match solution.status {
            SolverStatus::Solved => {
                let x = solution.x[0..n].to_vec();
                let alpha = solution.x[n];
                Some((alpha, x))
            }
            _ => None,
        }
    }

    pub fn ik(
        &self,
        config: &SolverConfig,
        tfm_target: &Isometry3<f64>,
        x0: Vec<f64>,
        ee_offset: &Isometry3<f64>,
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
                            self.chain.forward_kinematics_mut(x, ee_offset, fk);

                            // Compute the gradient only if it was requested by
                            // the optimizer.
                            if let Some(g) = grad {
                                objective_grad(
                                    self,
                                    tfm_target,
                                    fk,
                                    g,
                                    Vector3::from_row_slice(&config.linear_weight),
                                    Vector3::from_row_slice(&config.angular_weight),
                                );
                            }

                            // Always compute the objective value.
                            Some(objective(
                                self,
                                tfm_target,
                                fk,
                                Vector3::from_row_slice(&config.linear_weight),
                                Vector3::from_row_slice(&config.angular_weight),
                            ))
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
