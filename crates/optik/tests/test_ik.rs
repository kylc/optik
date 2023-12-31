use std::time::Instant;

use approx::assert_abs_diff_eq;
use nalgebra::{DVector, Isometry3, Vector6};
use optik::{Robot, SolutionMode, SolverConfig};
use rand::{rngs::StdRng, Rng, SeedableRng};

const TEST_MODEL_STR: &str = include_str!("data/ur3e.urdf");

#[test]
#[should_panic(expected = "joint limits")]
fn test_invalid_seed() {
    let robot = Robot::from_urdf_str(TEST_MODEL_STR, "ur_base_link", "ur_ee_link");
    let config = SolverConfig::default();
    let tfm_target = Isometry3::identity();

    let (_, ub) = robot.joint_limits();
    let mut x0 = vec![0.0; 6];
    x0[4] = ub[4] + 1.0;

    robot.ik(&config, &tfm_target, x0);
}

#[test]
fn test_stopping_maxtime() {
    const MAX_TIME: f64 = 0.05;

    let robot = Robot::from_urdf_str(TEST_MODEL_STR, "ur_base_link", "ur_ee_link");
    let tfm_target = Isometry3::translation(100.0, 100.0, 100.0); // impossible goal
    let x0 = vec![0.0; 6];

    let config = SolverConfig {
        max_time: MAX_TIME,
        ..Default::default()
    };

    let start = Instant::now();
    robot.ik(&config, &tfm_target, x0);
    let end = Instant::now();
    let duration = end - start;

    assert_abs_diff_eq!(duration.as_secs_f64(), MAX_TIME, epsilon = 1e-1);
}

#[test]
fn test_determinism() {
    let mut robot = Robot::from_urdf_str(TEST_MODEL_STR, "ur_base_link", "ur_ee_link");
    robot.set_parallelism(1);

    let mut rng = StdRng::seed_from_u64(42);
    let x_target: Vector6<f64> = rng.gen();
    let tfm_target = robot.fk(x_target.as_slice()).ee_tfm();

    let config = SolverConfig {
        max_time: 0.0,
        max_restarts: 25,
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
        max_time: 0.0,
        max_restarts: 25,
        ..Default::default()
    };

    for _ in 0..10 {
        let x_target: Vector6<f64> = rng.gen();
        let tfm_target = robot.fk(x_target.as_slice()).ee_tfm();

        let x_sol = DVector::from(
            robot
                .ik(&config, &tfm_target, vec![0.0; robot.num_positions()])
                .unwrap()
                .0,
        );
        let tfm_sol = robot.fk(x_sol.as_slice()).ee_tfm();

        // NOTE: epsilon here isn't exactly the same as the epsilon we pass into
        // the solver since we're not comparing within the manifold here. I've
        // chosen a reasonable epsilon here that passes the test with the given
        // solver tolerance.
        assert_abs_diff_eq!(tfm_target, tfm_sol, epsilon = 1e-6);
    }
}

#[test]
fn test_solution_quality() {
    let robot = Robot::from_urdf_str(TEST_MODEL_STR, "ur_base_link", "ur_ee_link");

    let mut rng = StdRng::seed_from_u64(42);

    let config_speed = SolverConfig {
        solution_mode: SolutionMode::Speed,
        max_time: 0.0,
        max_restarts: 15,
        ..Default::default()
    };
    let config_quality = SolverConfig {
        solution_mode: SolutionMode::Quality,
        ..config_speed
    };

    for _ in 0..20 {
        let x0 = vec![0.0; robot.num_positions()];
        let x_target: Vector6<f64> = rng.gen();
        let tfm_target = robot.fk(x_target.as_slice()).ee_tfm();

        let sol_speed = DVector::from(robot.ik(&config_speed, &tfm_target, x0.clone()).unwrap().0);
        let sol_quality = DVector::from(
            robot
                .ik(&config_quality, &tfm_target, x0.clone())
                .unwrap()
                .0,
        );

        let x0 = DVector::from_row_slice(x0.as_slice());
        assert!(sol_quality.metric_distance(&x0) <= sol_speed.metric_distance(&x0));
    }
}
