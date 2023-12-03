use approx::assert_abs_diff_eq;
use nalgebra::{Isometry3, Matrix3, Matrix6, Vector3, Vector6};
use optik::math;

// Various test cases for math operations. Ground truth is taken from equivalent
// function calls in Pinocchio, and compared against our results.

macro_rules! fetch_data {
    ($path: literal) => {
        serde_json::from_str(include_str!($path)).unwrap()
    };
}

#[test]
fn test_so3_log() {
    let inputs: Vec<Isometry3<f64>> = fetch_data!("data/test_math_inputs.json");
    let outputs: Vec<Vector3<f64>> = fetch_data!("data/test_math_outputs_so3_log.json");

    for (input, output) in inputs.into_iter().zip(outputs) {
        assert_abs_diff_eq!(math::so3::log(&input.rotation), output, epsilon = 1e-6);
    }
}

#[test]
fn test_so3_right_jacobian() {
    let inputs: Vec<Isometry3<f64>> = fetch_data!("data/test_math_inputs.json");
    let outputs: Vec<Matrix3<f64>> = fetch_data!("data/test_math_outputs_so3_right_jacobian.json");

    for (input, output) in inputs.into_iter().zip(outputs) {
        let so3_log = math::so3::log(&input.rotation);
        assert_abs_diff_eq!(math::so3::right_jacobian(&so3_log), output, epsilon = 1e-6);
    }
}

#[test]
fn test_se3_log() {
    let inputs: Vec<Isometry3<f64>> = fetch_data!("data/test_math_inputs.json");
    let outputs: Vec<Vector6<f64>> = fetch_data!("data/test_math_outputs_se3_log.json");

    for (input, output) in inputs.into_iter().zip(outputs) {
        assert_abs_diff_eq!(math::se3::log(&input), output, epsilon = 1e-6);
    }
}

#[test]
fn test_se3_right_jacobian() {
    let inputs: Vec<Isometry3<f64>> = fetch_data!("data/test_math_inputs.json");
    let outputs: Vec<Matrix6<f64>> = fetch_data!("data/test_math_outputs_se3_right_jacobian.json");

    for (input, output) in inputs.into_iter().zip(outputs) {
        assert_abs_diff_eq!(math::se3::right_jacobian(&input), output, epsilon = 1e-6);
    }
}
