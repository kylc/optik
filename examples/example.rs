use std::time::Instant;

use optik::*;
use rand::{rngs::StdRng, SeedableRng};

fn main() {
    let urdf_path = std::env::args().nth(1).unwrap();
    let base_name = std::env::args().nth(2).unwrap();
    let ee_name = std::env::args().nth(3).unwrap();

    let chain = k::Chain::<f64>::from_urdf_file(urdf_path).unwrap();
    let serial = k::SerialChain::from_end_to_root(
        chain.find_link(&ee_name).unwrap(),
        chain.find_link(&base_name).unwrap(),
    );
    let robot = Robot::new(serial);

    let config = SolverConfig::default();

    let mut rng = StdRng::seed_from_u64(42);
    let n = 10000;
    let mut t_tot = 0;
    for _ in 0..n {
        let q_target = robot.random_configuration(&mut rng);
        let target_ee_pose = robot.fk(&q_target);

        let t0 = Instant::now();
        let x0 = robot.random_configuration(&mut rng);
        if let Some((_, c)) = robot.ik(&config, &target_ee_pose, x0) {
            let tf = Instant::now();
            t_tot += (tf - t0).as_micros();

            println!("Total time: {:4.}us (to: {:.1e})", (tf - t0).as_micros(), c,);
        }
    }

    println!("Average time: {:.0}us", t_tot as f64 / n as f64);
}
