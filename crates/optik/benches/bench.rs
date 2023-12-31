use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::Isometry3;

use optik::*;

const BENCH_MODEL_STR: &str = include_str!("../tests/data/ur3e.urdf");

fn load_benchmark_model() -> Robot {
    Robot::from_urdf_str(BENCH_MODEL_STR, "ur_base_link", "ur_ee_link")
}

fn bench_gradient(c: &mut Criterion) {
    use optik::objective::objective_grad;

    let robot = load_benchmark_model();

    let q = robot.random_configuration(&mut rand::thread_rng());
    let mut g = vec![0.0; q.len()];
    let tfm_target = Isometry3::identity();

    c.bench_function("gradient", |b| {
        let fk = robot.fk(&q);
        b.iter(|| objective_grad(&robot, &tfm_target, &fk, &mut g))
    });
}

fn bench_objective(c: &mut Criterion) {
    use optik::objective::objective;

    let robot = load_benchmark_model();

    let q = robot.random_configuration(&mut rand::thread_rng());
    let fk = robot.fk(&q);
    let tfm_target = Isometry3::identity();
    c.bench_function("objective", |b| {
        b.iter(|| objective(&robot, &tfm_target, &fk))
    });
}

fn bench_fk(c: &mut Criterion) {
    let robot = load_benchmark_model();
    let x0 = vec![0.1, 0.2, 0.0, 0.3, -0.2, -1.1];

    c.bench_function("fk", |b| b.iter(|| robot.fk(&x0)));
}

fn bench_joint_jacobian(c: &mut Criterion) {
    let robot = load_benchmark_model();
    let x0 = vec![0.1, 0.2, 0.0, 0.3, -0.2, -1.1];

    let fk = robot.fk(&x0);
    c.bench_function("joint_jacobian", |b| b.iter(|| robot.joint_jacobian(&fk)));
}

fn bench_ik(c: &mut Criterion) {
    let robot = load_benchmark_model();
    let config = SolverConfig::default();

    let x0 = vec![0.1, 0.2, 0.0, 0.3, -0.2, -1.1];
    let tfm_target = robot.fk(&[-0.1, -0.2, 0.0, -0.3, 0.2, 1.1]).ee_tfm();

    c.bench_function("ik", |b| {
        b.iter(|| robot.ik(&config, black_box(&tfm_target), x0.clone()))
    });
}

criterion_group!(
    benches,
    bench_gradient,
    bench_objective,
    bench_fk,
    bench_joint_jacobian,
    bench_ik
);
criterion_main!(benches);
