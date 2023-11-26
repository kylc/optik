use approx::assert_abs_diff_eq;
use nalgebra::Vector6;
use optik::{
    kinematics::KinematicsCache,
    objective::{objective_grad, ObjectiveArgs},
    GradientMode, Robot, SolverConfig,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

const TEST_MODEL_STR: &str = include_str!("data/ur3e.urdf");

#[test]
fn test_gradient_analytical_vs_numerical() {
    let robot = Robot::from_urdf_str(TEST_MODEL_STR, "ur_base_link", "ur_ee_link");

    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..100 {
        let x0: Vector6<f64> = rng.gen();
        let tfm_target = rng.gen();

        // Analytical gradient
        let args = ObjectiveArgs {
            robot: &robot,
            config: &SolverConfig {
                gradient_mode: GradientMode::Analytical,
                ..Default::default()
            },
            tfm_target: &tfm_target,
        };

        let mut g_a = Vector6::zeros();
        objective_grad(
            x0.as_slice(),
            g_a.as_mut_slice(),
            &args,
            &mut KinematicsCache::default(),
        );

        // Numerical gradient
        let args = ObjectiveArgs {
            robot: &robot,
            config: &SolverConfig {
                gradient_mode: GradientMode::Numerical,
                ..Default::default()
            },
            tfm_target: &tfm_target,
        };

        let mut g_n = Vector6::zeros();
        objective_grad(
            x0.as_slice(),
            g_n.as_mut_slice(),
            &args,
            &mut KinematicsCache::default(),
        );

        assert_abs_diff_eq!(g_a, g_n, epsilon = 1e-6);
    }
}
