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

#[test]
fn test_solution_forward_backward() {
    let robot = Robot::from_urdf_str(TEST_MODEL_STR, "ur_base_link", "ur_ee_link");

    let mut rng = StdRng::seed_from_u64(42);

    let config = SolverConfig {
        solution_mode: SolutionMode::Speed,
        tol_f: 1e-12,
        ..Default::default()
    };

    for _ in 0..100 {
        let x_target: Vector6<f64> = rng.gen();
        let tfm_target = robot.fk(x_target.as_slice());

        let x_sol = DVector::from(
            robot
                .ik(&config, &tfm_target, vec![0.0; robot.num_positions()])
                .unwrap()
                .0,
        );
        let tfm_sol = robot.fk(x_sol.as_slice());

        // NOTE: epsilon here isn't exactly the same as the epsilon we pass into
        // the solver since we're not comparing within the manifold here. I've
        // chosen a reasonable epsilon here that passes the test with the given
        // solver tolerance.
        assert_abs_diff_eq!(tfm_target, tfm_sol, epsilon = 1e-6);
    }
}
