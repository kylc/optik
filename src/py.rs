use nalgebra::{Isometry3, Matrix4, Translation3, UnitQuaternion};

use pyo3::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::{config::SolverConfig, solve, GradientMode, Robot, SolutionMode};

#[pymethods]
impl SolverConfig {
    #[new]
    #[pyo3(signature=(gradient_mode=GradientMode::Analytical,
                      solution_mode=SolutionMode::Speed,
                      max_time=0.1,
                      xtol_abs=1e-5))]
    fn py_new(
        gradient_mode: GradientMode,
        solution_mode: SolutionMode,
        max_time: f64,
        xtol_abs: f64,
    ) -> Self {
        SolverConfig {
            gradient_mode,
            solution_mode,
            max_time,
            xtol_abs,
        }
    }
}

#[pyclass]
#[pyo3(name = "Robot")]
struct PyRobot(Robot);

#[pyclass]
#[pyo3(name = "Isometry3")]
struct PyIsometry3(Isometry3<f64>);

#[pyfunction]
fn set_parallelism(n: usize) {
    ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .unwrap()
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn load_model(path: &str, base_link: &str, ee_link: &str) -> PyRobot {
    let chain = k::Chain::<f64>::from_urdf_file(path).unwrap();

    let base_link = chain
        .find_link(base_link)
        .unwrap_or_else(|| panic!("link '{}' does not exist!", base_link));
    let ee_link = chain
        .find_link(ee_link)
        .unwrap_or_else(|| panic!("link '{}' does not exist!", ee_link));

    let serial = k::SerialChain::from_end_to_root(ee_link, base_link);
    PyRobot(Robot::new(serial))
}

#[pyfunction]
fn random_pose(robot: &PyRobot) -> PyIsometry3 {
    let robot = &robot.0;
    let q_target = robot.random_configuration(&mut rand::thread_rng());
    let target_ee_pose = robot.fk(&q_target);

    PyIsometry3(target_ee_pose)
}

#[pyfunction]
fn parse_pose(v: Vec<Vec<f64>>) -> PyIsometry3 {
    let mut m = Matrix4::zeros();
    for i in 0..4 {
        for j in 0..4 {
            m[(i, j)] = v[i][j];
        }
    }

    let t = Translation3::from(m.fixed_slice::<3, 1>(0, 3).into_owned());
    let r = UnitQuaternion::from_matrix(&m.fixed_slice::<3, 3>(0, 0).into_owned());

    PyIsometry3(Isometry3::from_parts(t, r))
}

#[pyfunction]
#[pyo3(name = "solve")]
fn py_solve(
    robot: &PyRobot,
    config: &SolverConfig,
    target: &PyIsometry3,
    x0: Vec<f64>,
) -> (Option<Vec<f64>>, f64) {
    solve(&robot.0, config, &target.0, x0)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn optik(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRobot>()?;
    m.add_class::<GradientMode>()?;
    m.add_class::<SolutionMode>()?;
    m.add_class::<SolverConfig>()?;
    m.add_function(wrap_pyfunction!(set_parallelism, m)?)?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(random_pose, m)?)?;
    m.add_function(wrap_pyfunction!(parse_pose, m)?)?;
    m.add_function(wrap_pyfunction!(py_solve, m)?)?;
    Ok(())
}
