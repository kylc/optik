use std::ops::Deref;

use nalgebra::{Isometry3, Matrix4, Translation3, UnitQuaternion};

use pyo3::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::{config::SolverConfig, GradientMode, Robot, SolutionMode};

#[pyfunction]
fn set_parallelism(n: usize) {
    ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .unwrap()
}

#[pymethods]
impl SolverConfig {
    #[new]
    #[pyo3(signature=(gradient_mode=GradientMode::Analytical,
                      solution_mode=SolutionMode::Speed,
                      max_time=0.1,
                      xtol_abs=1e-10,
                      ftol_abs=1e-5))]
    fn py_new(
        gradient_mode: GradientMode,
        solution_mode: SolutionMode,
        max_time: f64,
        xtol_abs: f64,
        ftol_abs: f64,
    ) -> Self {
        SolverConfig {
            gradient_mode,
            solution_mode,
            max_time,
            xtol_abs,
            ftol_abs,
        }
    }
}

#[pyclass]
#[pyo3(name = "Robot")]
pub struct PyRobot(Robot);

impl Deref for PyRobot {
    type Target = Robot;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyRobot {
    #[staticmethod]
    fn from_urdf_file(path: &str, base_link: &str, ee_link: &str) -> PyRobot {
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

    fn joint_limits(&self) -> (Vec<f64>, Vec<f64>) {
        let robot = &self.0;

        (
            robot.joint_limits().0.as_slice().to_vec(),
            robot.joint_limits().1.as_slice().to_vec(),
        )
    }

    fn fk(&self, x: Vec<f64>) -> Vec<Vec<f64>> {
        let robot = &self.0;

        assert_eq!(x.len(), robot.chain.dof());

        let ee_pose = robot.fk(&x);
        ee_pose
            .to_matrix()
            .row_iter()
            .map(|row| row.iter().copied().collect())
            .collect()
    }

    fn ik(
        &self,
        config: &SolverConfig,
        target: Vec<Vec<f64>>,
        x0: Vec<f64>,
    ) -> (Option<Vec<f64>>, f64) {
        let robot = &self.0;

        assert_eq!(x0.len(), robot.chain.dof());

        fn parse_pose(v: Vec<Vec<f64>>) -> Isometry3<f64> {
            let m = Matrix4::from_iterator(v.into_iter().flatten()).transpose();

            let t = Translation3::from(m.fixed_slice::<3, 1>(0, 3).into_owned());
            let r = UnitQuaternion::from_matrix(&m.fixed_slice::<3, 3>(0, 0).into_owned());

            Isometry3::from_parts(t, r)
        }

        let target = parse_pose(target);
        robot.ik(config, &target, x0)
    }
}

#[pymodule]
fn optik(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRobot>()?;
    m.add_class::<GradientMode>()?;
    m.add_class::<SolutionMode>()?;
    m.add_class::<SolverConfig>()?;
    m.add_function(wrap_pyfunction!(set_parallelism, m)?)?;
    Ok(())
}
