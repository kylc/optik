use approx::assert_abs_diff_eq;
use nalgebra::{DVector, Vector6};
use optik::{
    objective::{objective, objective_grad},
    Robot,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

const TEST_MODEL_STR: &str = include_str!("data/ur3e.urdf");

fn finite_difference<F>(f: F, x: &[f64]) -> DVector<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let mut g = vec![0.0; n];
    let mut x0 = x.to_vec();
    let eps = f64::EPSILON.powf(1.0 / 3.0);
    for i in 0..n {
        let x0i = x0[i];
        x0[i] = x0i - eps;
        // let fkl = args.robot.fk(&x0);
        let fl = f(&x0);
        x0[i] = x0i + eps;
        // let fkh = args.robot.fk(&x0);
        let fh = f(&x0);
        g[i] = (fh - fl) / (2.0 * eps);
        x0[i] = x0i;
    }

    DVector::from_row_slice(&g)
}

#[test]
fn test_gradient_analytical_vs_numerical() {
    let robot = Robot::from_urdf_str(TEST_MODEL_STR, "ur_base_link", "ur_ee_link");

    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..100 {
        let x0: Vector6<f64> = rng.gen();
        let tfm_target = rng.gen();
        let fk = robot.fk(x0.as_slice());

        // Analytical gradient
        let mut g_a = Vector6::zeros();
        objective_grad(&robot, &tfm_target, &fk, g_a.as_mut_slice());

        // Numerical gradient
        let g_n = finite_difference(
            |x| {
                let fk = robot.fk(x);
                objective(&robot, &tfm_target, &fk)
            },
            x0.as_slice(),
        );

        assert_abs_diff_eq!(g_a.as_slice(), g_n.as_slice(), epsilon = 1e-6);
    }
}
