use approx::assert_abs_diff_eq;
use nalgebra::{DVector, Vector6};
use optik::{Robot, SolutionMode, SolverConfig};
use rand::{rngs::StdRng, Rng, SeedableRng};

const TEST_MODEL_STR: &str = include_str!("data/ur3e.urdf");

#[test]
fn test_determinism() {
    let robot = Robot::from_urdf_str(TEST_MODEL_STR, "ur_base_link", "ur_ee_link");

    let mut rng = StdRng::seed_from_u64(42);
    let x_target: Vector6<f64> = rng.gen();
    let tfm_target = robot.fk(x_target.as_slice());

    let config = SolverConfig {
        solution_mode: SolutionMode::Quality,
        max_time: 0.0,
        max_restarts: 10,
        ..Default::default()
    };
    let x_sol = DVector::from(
        robot
            .ik(&config, &tfm_target, vec![0.0; robot.num_positions()])
            .unwrap()
            .0,
    );

    // `robot.ik()` should return the same solution every time.
    for _ in 0..10 {
        let x_sol_i = DVector::from(
            robot
                .ik(&config, &tfm_target, vec![0.0; robot.num_positions()])
                .unwrap()
                .0,
        );

        assert_abs_diff_eq!(x_sol, x_sol_i, epsilon = 1e-6);
    }
}
