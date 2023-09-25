use std::time::Instant;

use optik::*;
use rand::{rngs::StdRng, SeedableRng};

fn main() {
    let urdf_path = std::env::args().nth(1).unwrap();
    let ee_name = std::env::args().nth(2).unwrap();

    let chain = k::Chain::<f64>::from_urdf_file(urdf_path).unwrap();
    let serial = k::SerialChain::from_end(chain.find_link(&ee_name).unwrap());
    let robot = Robot::new(serial);

    let config = SolverConfig::default();

    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..10000 {
        let q_target = robot.random_configuration(&mut rng);
        let target_ee_pose = robot.fk(&q_target);

        let t0 = Instant::now();
        let x0 = robot.random_configuration(&mut rng);
        solve(&robot, &config, &target_ee_pose, x0);
        let tf = Instant::now();

        println!(
            "Total time: {:4.}us (to: {:.4}, {} tries)",
            (tf - t0).as_micros(),
            0,
            0
        );
    }
}
