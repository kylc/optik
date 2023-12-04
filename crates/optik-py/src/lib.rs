use std::ops::Deref;

use optik::{GradientMode, Robot, SolutionMode, SolverConfig};

use nalgebra::{Isometry3, Matrix4, Rotation3, Translation3, UnitQuaternion};
use pyo3::prelude::*;

#[pyfunction]
fn set_parallelism(n: usize) {
    optik::set_parallelism(n)
}

#[pyclass]
#[pyo3(name = "SolverConfig")]
struct PySolverConfig(SolverConfig);

#[pymethods]
impl PySolverConfig {
    #[new]
    #[pyo3(signature=(gradient_mode="analytical",
                      solution_mode="speed",
                      max_time=0.1,
                      max_restarts=u64::MAX,
                      tol_f=1e-6,
                      tol_dx=-1.0,
                      tol_df=-1.0))]
    fn py_new(
        gradient_mode: &str,
        solution_mode: &str,
        max_time: f64,
        max_restarts: u64,
        tol_f: f64,
        tol_dx: f64,
        tol_df: f64,
    ) -> Self {
        let gradient_mode = match gradient_mode {
            "analytical" => GradientMode::Analytical,
            "numerical" => GradientMode::Numerical,
            _ => panic!("invalid gradient mode"),
        };

        let solution_mode = match solution_mode {
            "speed" => SolutionMode::Speed,
            "quality" => SolutionMode::Quality,
            _ => panic!("invalid solution mode"),
        };

        if max_time == 0.0 && max_restarts == 0 {
            panic!("no time or restart limit applied -- solver would run forever")
        }

        PySolverConfig(SolverConfig {
            gradient_mode,
            solution_mode,
            max_time,
            max_restarts,
            tol_dx,
            tol_f,
            tol_df,
        })
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
        PyRobot(Robot::from_urdf_file(path, base_link, ee_link))
    }

    pub fn num_positions(&self) -> usize {
        self.0.num_positions()
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

        assert_eq!(x.len(), robot.num_positions());

        let ee_pose = robot.fk(&x);
        ee_pose
            .to_matrix()
            .row_iter()
            .map(|row| row.iter().copied().collect())
            .collect()
    }

    fn ik(
        &self,
        config: &PySolverConfig,
        target: Vec<Vec<f64>>,
        x0: Vec<f64>,
    ) -> Option<(Vec<f64>, f64)> {
        let robot = &self.0;

        assert_eq!(x0.len(), self.num_positions());

        fn parse_pose(v: Vec<Vec<f64>>) -> Isometry3<f64> {
            let m = Matrix4::from_iterator(v.into_iter().flatten()).transpose();

            let t = Translation3::from(m.fixed_slice::<3, 1>(0, 3).into_owned());

            // Some rotational gymnastics here to work around an nalgebra bug
            // (fixed in 0.31) which fails on direct conversion from Matrix3 ->
            // UnitQuaternion for inputs that are already a rotation matrix.
            // https://github.com/dimforge/nalgebra/pull/1101
            let g = m.fixed_slice::<3, 3>(0, 0).into_owned();
            let r = Rotation3::from_matrix_unchecked(g);
            let q = UnitQuaternion::from_rotation_matrix(&r);

            Isometry3::from_parts(t, q)
        }

        let target = parse_pose(target);
        robot.ik(&config.0, &target, x0)
    }
}

#[pymodule()]
#[pyo3(name = "optik")]
fn optik_extension(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRobot>()?;
    m.add_class::<PySolverConfig>()?;
    m.add_function(wrap_pyfunction!(set_parallelism, m)?)?;
    Ok(())
}
