use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::Isometry3;

use optik::*;

const BENCH_MODEL_STR: &str = include_str!("../tests/data/ur3e.urdf");

fn load_benchmark_model() -> Robot {
    Robot::from_urdf_str(BENCH_MODEL_STR, "ur_base_link", "ur_ee_link")
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
    let mut g = vec![0.0; q.len()];
    let tfm_target = Isometry3::identity();
    let args = ObjectiveArgs {
        robot,
        config: SolverConfig::default(),
        tfm_target,
    };

    c.bench_function("gradient_analytical", |b| {
        let args = ObjectiveArgs {
            config: SolverConfig {
                gradient_mode: GradientMode::Analytical,
                ..args.config
            },
            ..args.clone()
        };
        b.iter(|| objective_grad(black_box(&q), &mut g, &args))
    });

    c.bench_function("gradient_numerical", |b| {
        let args = ObjectiveArgs {
            config: SolverConfig {
                gradient_mode: GradientMode::Numerical,
                ..args.config
            },
            ..args.clone()
        };
        b.iter(|| objective_grad(black_box(&q), &mut g, &args))
    });
}

fn bench_objective(c: &mut Criterion) {
    let robot = load_benchmark_model();

    let q = robot.random_configuration(&mut rand::thread_rng());
    let tfm_target = Isometry3::identity();
    let args = ObjectiveArgs {
        robot,
        config: SolverConfig::default(),
        tfm_target,
    };
    c.bench_function("objective", |b| b.iter(|| objective(black_box(&q), &args)));
}

fn bench_ik(c: &mut Criterion) {
    let robot = load_benchmark_model();
    let config = SolverConfig::default();

    let x0 = vec![0.1, 0.2, 0.0, 0.3, -0.2, -1.1];
    let tfm_target = robot.fk(&[-0.1, -0.2, 0.0, -0.3, 0.2, 1.1]);

    c.bench_function("ik", |b| {
        b.iter(|| robot.ik(&config, black_box(&tfm_target), x0.clone()))
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
