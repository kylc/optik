#![allow(non_local_definitions)] // silence warning from pyo3::pymethods macro

use optik::{Robot, SolverConfig};

use nalgebra::{
    Isometry3, Matrix3, Matrix4, Quaternion, Rotation3, Translation3, UnitQuaternion, UnitVector3,
    Vector3,
};
use pyo3::prelude::*;

fn parse_pose(v: Option<Vec<Vec<f64>>>) -> Isometry3<f64> {
    if let Some(v) = v {
        let matrix = Matrix4::from_iterator(v.into_iter().flatten()).transpose();
        let isometry_opt: Option<Isometry3<f64>> = nalgebra::try_convert(matrix);
        match isometry_opt {
            Some(isometry) => return isometry,
            None => {
                // normalize the rotation part of the matrix
                let translation_vec: Vector3<f64> = matrix.fixed_view::<3, 1>(0, 3).into();
                let translation: Translation3<f64> = Translation3::from(translation_vec);
                let rotation_matrix: Matrix3<f64> = matrix.fixed_view::<3, 3>(0, 0).into();
                let mut rotation: Rotation3<f64> =
                    Rotation3::from_matrix_unchecked(rotation_matrix);
                rotation.renormalize();
                return Isometry3::from_parts(translation, rotation.into());
            }
        }
    } else {
        Isometry3::identity()
    }
}

fn pose_to_isometry(v: Option<Vec<f64>>) -> Isometry3<f64> {
    // v is already [px, py, pz, qw, qx, qy, qz]

    if let Some(v) = v {
        Isometry3::from_parts(
            Translation3::new(v[0], v[1], v[2]),
            UnitQuaternion::from_quaternion(Quaternion::new(v[3], v[4], v[5], v[6])),
        )
    } else {
        Isometry3::identity()
    }
}

#[pyclass]
#[pyo3(name = "SolverConfig")]
struct PySolverConfig(SolverConfig);

#[pymethods]
impl PySolverConfig {
    #[new]
    #[pyo3(signature=(solution_mode="speed",
                      max_time=0.1,
                      max_restarts=u64::MAX,
                      tol_f=1e-6,
                      tol_df=-1.0,
                      tol_dx=-1.0,
                      linear_weight=[1.0, 1.0, 1.0],
                      angular_weight=[1.0, 1.0, 1.0]))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        solution_mode: &str,
        max_time: f64,
        max_restarts: u64,
        tol_f: f64,
        tol_df: f64,
        tol_dx: f64,
        linear_weight: [f64; 3],
        angular_weight: [f64; 3],
    ) -> Self {
        let solution_mode = solution_mode.parse().expect("invalid solution mode");

        if max_time == 0.0 && max_restarts == 0 {
            panic!("no time or restart limit applied -- solver would run forever")
        }

        PySolverConfig(SolverConfig {
            solution_mode,
            max_time,
            max_restarts,
            tol_f,
            tol_df,
            tol_dx,
            linear_weight,
            angular_weight,
        })
    }
}

#[pyclass]
#[pyo3(name = "Robot")]
pub struct PyRobot(Robot);

#[pymethods]
impl PyRobot {
    #[staticmethod]
    fn from_urdf_file(path: &str, base_link: &str, ee_link: &str) -> PyRobot {
        PyRobot(Robot::from_urdf_file(path, base_link, ee_link))
    }

    pub fn set_parallelism(&mut self, n: usize) {
        self.0.set_parallelism(n);
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

    #[pyo3(signature=(x, ee_offset=None))]
    fn joint_jacobian(&self, x: Vec<f64>, ee_offset: Option<Vec<Vec<f64>>>) -> Vec<Vec<f64>> {
        let robot = &self.0;

        assert_eq!(x.len(), robot.num_positions(), "len(x0) != num_positions");

        let fk = robot.fk(&x, &parse_pose(ee_offset));
        let jac = robot.joint_jacobian(&fk);
        jac.row_iter()
            .map(|row| row.iter().copied().collect())
            .collect()
    }

    #[pyo3(signature=(x, ee_offset=None))]
    fn fk(&self, x: Vec<f64>, ee_offset: Option<Vec<Vec<f64>>>) -> Vec<Vec<f64>> {
        let robot = &self.0;

        assert_eq!(x.len(), robot.num_positions(), "len(x0) != num_positions");

        let ee_pose = robot.fk(&x, &parse_pose(ee_offset)).ee_tfm();
        ee_pose
            .to_matrix()
            .row_iter()
            .map(|row| row.iter().copied().collect())
            .collect()
    }

    #[pyo3(signature=(x, ee_offset=None))]
    fn fk_medra(&self, x: Vec<f64>, ee_offset: Option<Vec<f64>>) -> Vec<f64> {
        // Evaluate forward kinematics. ee_offset and return type are
        //   pose + quaternion vector [px, py, pz, qw, qx, qy, qz].
        let robot = &self.0;

        assert_eq!(x.len(), robot.num_positions(), "len(x0) != num_positions");

        let ee_pose = robot.fk(&x, &pose_to_isometry(ee_offset)).ee_tfm();

        let translation = ee_pose.translation.vector;
        let rotation = ee_pose.rotation;
        let quat = rotation.quaternion();

        vec![
            translation.x,
            translation.y,
            translation.z,
            quat.w,
            quat.i,
            quat.j,
            quat.k,
        ]
    }

    #[pyo3(signature=(config, target, x0, ee_offset=None))]
    fn ik(
        &self,
        config: &PySolverConfig,
        target: Vec<Vec<f64>>,
        x0: Vec<f64>,
        ee_offset: Option<Vec<Vec<f64>>>,
    ) -> Option<(Vec<f64>, f64)> {
        let robot = &self.0;

        assert_eq!(x0.len(), self.num_positions(), "len(x0) != num_positions");

        let target = parse_pose(Some(target));
        let ee_offset = parse_pose(ee_offset);
        robot.ik(&config.0, &target, x0, &ee_offset)
    }

    #[pyo3(signature=(config, target_pose, x0, ee_offset_pose=None))]
    fn ik_medra(
        &self,
        config: &PySolverConfig,
        target_pose: Vec<f64>,
        x0: Vec<f64>,
        ee_offset_pose: Option<Vec<f64>>,
    ) -> Option<(Vec<f64>, f64)> {
        let robot = &self.0;

        assert_eq!(x0.len(), self.num_positions(), "len(x0) != num_positions");

        let target = pose_to_isometry(Some(target_pose));
        let ee_offset = pose_to_isometry(ee_offset_pose);
        robot.ik(&config.0, &target, x0, &ee_offset)
    }

    #[pyo3(signature=(seed))]
    fn initalize_rng(&mut self, seed: u64){
        self.0.initialize_rng(seed);
    }

    #[pyo3(signature=(source_vec_in_tip_frame, target_vec, max_angle, ee_offset_pose, seed_joint_angles, config))]
    fn apply_angle_between_two_vectors_constraint(
        &mut self,
        source_vec_in_tip_frame: Vec<f64>,
        target_vec: Vec<f64>,
        max_angle: f64,
        ee_offset_pose: Vec<f64>,
        seed_joint_angles: Vec<f64>,
        config: &PySolverConfig,
    ) -> Option<Vec<f64>> {
        let source_vector_tip_frame: UnitVector3<f64> =
            UnitVector3::new_normalize(Vector3::from_column_slice(&source_vec_in_tip_frame));
        let target_vector: UnitVector3<f64> =
            UnitVector3::new_normalize(Vector3::from_column_slice(&target_vec));
        let ee_transform: Isometry3<f64> = pose_to_isometry(Some(ee_offset_pose));
        return self.0.apply_angle_between_two_vectors_constraint(
            source_vector_tip_frame,
            target_vector,
            max_angle,
            ee_transform,
            seed_joint_angles,
            &config.0,
        );
    }
}

#[pymodule()]
#[pyo3(name = "optik")]
fn optik_extension(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRobot>()?;
    m.add_class::<PySolverConfig>()?;
    Ok(())
}
