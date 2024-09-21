use std::{process, time::Instant};

use nalgebra::Isometry3;
use optik::*;
use rand::{rngs::StdRng, SeedableRng};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let robot = if let Some([urdf_path, base_name, ee_name]) = args.get(1..4) {
        Robot::from_urdf_file(urdf_path, base_name, ee_name)
    } else {
        eprintln!("Usage: example <my_robot.urdf> <base_link> <ee_link>");
        process::exit(1);
    };

    let config = SolverConfig::default();

    let mut rng = StdRng::seed_from_u64(42);
    let n = 10000;

    let mut n_success = 0;
    let mut t_tot = 0;
    for _ in 0..n {
        let x0 = robot.random_configuration(&mut rng);
        let q_target = robot.random_configuration(&mut rng);
        let target_ee_pose = robot.fk(&q_target, &Isometry3::identity()).ee_tfm();

        let t0 = Instant::now();
        if let Some((_, c)) = robot.ik(&config, &target_ee_pose, x0, &Isometry3::identity()) {
            let tf = Instant::now();
            t_tot += (tf - t0).as_micros();
            n_success += 1;

            println!("Solve time: {:4.}us (to: {:.1e})", (tf - t0).as_micros(), c);
        }
    }

    println!("Average time: {:.0}us", t_tot as f64 / n_success as f64);
    println!(
        "Success rate: {:.1}%",
        100.0 * (n_success as f64 / n as f64)
    );
}
