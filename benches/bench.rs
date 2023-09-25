use criterion::{black_box, criterion_group, criterion_main, Criterion};
use k::Chain;
use nalgebra::Isometry3;
use rand::Rng;

use optik::*;

const BENCHMARK_MODEL_PATH: &str = "models/ur3e.urdf";
const BENCHMARK_MODEL_EE_NAME: &str = "ur_ee_link";

fn load_benchmark_model() -> Robot {
    let chain = Chain::<f64>::from_urdf_file(BENCHMARK_MODEL_PATH).unwrap();
    let serial = k::SerialChain::from_end(chain.find_link(BENCHMARK_MODEL_EE_NAME).unwrap());

    Robot::new(serial)
}

fn bench_jacobian(c: &mut Criterion) {
    let robot = load_benchmark_model();
    let q = robot.random_configuration(&mut rand::thread_rng());

    c.bench_function("jacobian", |b| {
        b.iter(|| robot.jacobian_local(black_box(&q)))
    });
}

fn bench_gradient(c: &mut Criterion) {
    let robot = load_benchmark_model();

    let q = robot.random_configuration(&mut rand::thread_rng());
    let tfm_target = rand::thread_rng().gen();
    let tfm_actual = robot.fk(&q);

    c.bench_function("gradient_analytical", |b| {
        b.iter(|| robot.ee_error_grad(&tfm_target, &tfm_actual, &q, GradientMode::Analytical))
    });

    c.bench_function("gradient_numerical", |b| {
        b.iter(|| robot.ee_error_grad(&tfm_target, &tfm_actual, &q, GradientMode::Numerical))
    });
}

fn bench_objective(c: &mut Criterion) {
    let robot = load_benchmark_model();

    let q = robot.random_configuration(&mut rand::thread_rng());
    let target_pose = Isometry3::identity();
    let mut grad = vec![0.0; 6];
    let mut user_data = (target_pose, robot);
    c.bench_function("objective", |b| {
        b.iter(|| {
            objective(&q, Some(&mut grad), &mut user_data);
        })
    });
}

fn bench_ik(c: &mut Criterion) {
    let robot = load_benchmark_model();
    let config = SolverConfig::default();

    let x0 = vec![0.1, 0.2, 0.0, 0.3, -0.2, -1.1];
    let tfm_target = robot.fk(&vec![-0.1, -0.2, 0.0, -0.3, 0.2, 1.1]);

    c.bench_function("ik", |b| {
        b.iter(|| solve(&robot, &config, &tfm_target, x0.clone()))
    });
}

criterion_group!(
    benches,
    bench_jacobian,
    bench_gradient,
    bench_objective,
    bench_ik
);
criterion_main!(benches);
