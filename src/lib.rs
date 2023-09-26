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

    pub fn ik(
        &self,
        config: &SolverConfig,
        tfm_target: &Isometry3<f64>,
        x0: Vec<f64>,
    ) -> (Option<Vec<f64>>, f64) {
        let (lb, ub) = self.joint_limits();

        let max_time = Duration::from_secs_f64(config.max_time);
        let start_time = Instant::now();

        // TODO: We can probably use the `rayon::ThreadPoolBuilder` to keep a
        // thread-local Nlopt instance, instead of recreating it for every
        // iteration.

        const RNG_SEED: u64 = 42;

        let should_exit = Arc::new(AtomicBool::new(false));
        let solution_stream = (0..u64::MAX)
            .into_par_iter()
            .map(|i| {
                Nlopt::<Box<dyn ObjFn<()>>, ()>::srand_seed(Some(RNG_SEED));

                let mut rng = ChaCha8Rng::seed_from_u64(RNG_SEED);
                rng.set_stream(i);

                let args = ObjectiveArgs {
                    robot: self.clone(),
                    config: config.clone(),
                    tfm_target: tfm_target.clone(),
                    should_exit: Arc::clone(&should_exit),
                };

                let mut opt = Nlopt::new(
                    nlopt::Algorithm::Slsqp,
                    self.chain.dof(),
                    objective,
                    nlopt::Target::Minimize,
                    args,
                );
                opt.set_ftol_abs(config.ftol_abs).unwrap();
                opt.set_xtol_abs1(config.xtol_abs).unwrap();
                opt.set_lower_bounds(lb.as_slice()).unwrap();
                opt.set_upper_bounds(ub.as_slice()).unwrap();
                opt.set_maxtime(config.max_time).unwrap();

                // The first attempt gets the initial seed provided by the caller.
                // All other attempts start at some random point.
                let mut x = if i == 0 {
                    x0.clone()
                } else {
                    self.random_configuration(&mut rng)
                };

                let res = opt.optimize(&mut x);

                if let Ok((_, cost)) = res {
                    // Short-circuit any other threads before we return for a
                    // modest performance improvement.
                    if config.solution_mode == SolutionMode::Speed && cost < config.ftol_abs {
                        should_exit.store(true, Ordering::SeqCst);
                    }

                    (Some(x), cost)
                } else {
                    (None, f64::INFINITY)
                }
            })
            .take_any_while(|_| (Instant::now() - start_time) < max_time);

        // TODO: Don't unwrap: we may not have a soultion at all, in which case
        // we should return None.
        // TODO: Hardcoded score conditions below need to be configurable (based
        // on `ftol`?)
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
                    .find_any(|&(_, score)| score < config.ftol_abs)
                    .unwrap()
            }
        }
    }
}

pub struct ObjectiveArgs {
    pub robot: Robot,
    pub config: SolverConfig,
    pub tfm_target: Isometry3<f64>,
    pub should_exit: Arc<AtomicBool>,
}

pub fn objective(x: &[f64], grad: Option<&mut [f64]>, args: &mut ObjectiveArgs) -> f64 {
    // A poor substitude to calling Nlopt::force_stop, but the usage of that API
    // is almost impossible. Return an objective and gradient we know will cause
    // the optimizer to exit immediately -- because it thinks it is done!
    //
    // Just make sure we don't actually interpret this as a real solution.
    if args.should_exit.load(Ordering::Relaxed) {
        if let Some(g) = grad {
            g.fill(0.0)
        };
        return 0.0;
    }

    let robot = &args.robot;
    let tfm_actual = args.robot.fk(x);
    let tfm_target = args.tfm_target;
    let tfm_error = tfm_target.inverse() * tfm_actual;

    if let Some(g) = grad {
        let grad = robot.ee_error_grad(&tfm_target, &tfm_actual, x, args.config.gradient_mode);
        g.copy_from_slice(grad.as_slice());
    }

    se3::log(tfm_error).norm_squared()
}
