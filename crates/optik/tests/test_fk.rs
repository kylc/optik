use approx::assert_abs_diff_eq;
use nalgebra::Isometry3;
use optik::Robot;

const TEST_MODEL_STR: &str = include_str!("data/ur3e.urdf");

macro_rules! fetch_data {
    ($path: literal) => {
        serde_json::from_str(include_str!($path)).unwrap()
    };
}

#[test]
fn test_fk() {
    let inputs: Vec<Vec<f64>> = fetch_data!("data/test_fk_inputs.json");
    let outputs: Vec<Isometry3<f64>> = fetch_data!("data/test_fk_outputs.json");

    let robot = Robot::from_urdf_str(TEST_MODEL_STR, "ur_base_link", "ur_ee_link");
    for (input, output) in inputs.into_iter().zip(outputs) {
        assert_abs_diff_eq!(robot.fk(&input).ee_tfm(), output, epsilon = 1e-6);
    }
}
