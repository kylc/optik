use std::{
    ffi::{CStr, c_char, c_double, c_uint, c_ulong},
    mem,
};

use nalgebra::{DMatrix, Isometry3, Translation3, UnitQuaternion};

use optik::{Robot, SolutionMode, SolverConfig};

#[repr(C)]
struct CSolverConfig {
    solution_mode: SolutionMode,
    max_time: c_double,
    max_restarts: c_ulong,
    tol_f: c_double,
    tol_df: c_double,
    tol_dx: c_double,
    linear_weight: [c_double; 3],
    angular_weight: [c_double; 3],
}

fn to_str(c_str: *const c_char) -> &'static str {
    unsafe { CStr::from_ptr(c_str).to_str().unwrap() }
}

#[unsafe(no_mangle)]
extern "C" fn optik_robot_from_urdf_file(
    path: *const c_char,
    base_link: *const c_char,
    ee_link: *const c_char,
) -> *mut Robot {
    let path = to_str(path);
    let base_link = to_str(base_link);
    let ee_link = to_str(ee_link);

    Box::into_raw(Box::new(Robot::from_urdf_file(path, base_link, ee_link)))
}

#[unsafe(no_mangle)]
extern "C" fn optik_robot_from_urdf_str(
    urdf: *const c_char,
    base_link: *const c_char,
    ee_link: *const c_char,
) -> *mut Robot {
    let urdf = to_str(urdf);
    let base_link = to_str(base_link);
    let ee_link = to_str(ee_link);

    Box::into_raw(Box::new(Robot::from_urdf_str(urdf, base_link, ee_link)))
}

#[unsafe(no_mangle)]
extern "C" fn optik_robot_free(robot: *mut Robot) {
    unsafe {
        drop(Box::from_raw(robot));
    }
}

#[unsafe(no_mangle)]
extern "C" fn optik_robot_set_parallelism(robot: *mut Robot, n: c_uint) {
    unsafe {
        let robot = robot.as_mut().unwrap();
        robot.set_parallelism(n as usize)
    }
}

#[unsafe(no_mangle)]
extern "C" fn optik_robot_num_positions(robot: *const Robot) -> c_uint {
    unsafe {
        let robot = robot.as_ref().unwrap();
        robot.num_positions() as c_uint
    }
}

#[unsafe(no_mangle)]
extern "C" fn optik_robot_joint_limits(robot: *const Robot) -> *const c_double {
    unsafe {
        let robot = robot.as_ref().unwrap();

        let mut bounds = robot.joint_limits().0.as_slice().to_vec();
        bounds.append(&mut robot.joint_limits().1.as_slice().to_vec());
        bounds.shrink_to_fit();

        let ptr = bounds.as_ptr();
        mem::forget(bounds);
        ptr
    }
}

#[unsafe(no_mangle)]
extern "C" fn optik_robot_joint_jacobian(
    robot: *const Robot,
    x: *const c_double,
) -> *const c_double {
    unsafe {
        let robot = robot.as_ref().unwrap();
        let x = std::slice::from_raw_parts(x, robot.num_positions());

        let fk = robot.fk(x, &Isometry3::identity());
        let jac = robot.joint_jacobian(&fk);

        jac.data.as_slice().to_vec().leak().as_ptr()
    }
}

#[unsafe(no_mangle)]
extern "C" fn optik_robot_fk(robot: *const Robot, x: *const c_double) -> *const c_double {
    unsafe {
        let robot = robot.as_ref().unwrap();
        let x = std::slice::from_raw_parts(x, robot.num_positions());

        let ee_pose = robot.fk(x, &Isometry3::identity()).ee_tfm();

        ee_pose.to_matrix().data.as_slice().to_vec().leak().as_ptr()
    }
}

#[unsafe(no_mangle)]
extern "C" fn optik_robot_random_configuration(robot: *const Robot) -> *const c_double {
    unsafe {
        let robot = robot.as_ref().unwrap();
        let q = robot.random_configuration(&mut rand::rng());
        q.leak().as_ptr()
    }
}

#[unsafe(no_mangle)]
extern "C" fn optik_robot_ik(
    robot: *const Robot,
    config: *const CSolverConfig,
    target: *const c_double,
    x0: *const c_double,
) -> *const c_double {
    unsafe {
        let robot = robot.as_ref().unwrap();
        let x0 = std::slice::from_raw_parts(x0, robot.num_positions());

        let target = std::slice::from_raw_parts(target, 4 * 4);
        let m = DMatrix::from_column_slice(4, 4, target);

        let t = Translation3::from(m.fixed_view::<3, 1>(0, 3).into_owned());
        let r = UnitQuaternion::from_matrix(&m.fixed_view::<3, 3>(0, 0).into_owned());

        let ee_pose = Isometry3::from_parts(t, r);

        let config = SolverConfig {
            solution_mode: (*config).solution_mode,
            max_time: (*config).max_time,
            max_restarts: (*config).max_restarts,
            tol_f: (*config).tol_f,
            tol_df: (*config).tol_df,
            tol_dx: (*config).tol_dx,
            linear_weight: (*config).linear_weight,
            angular_weight: (*config).angular_weight,
        };
        if let Some((v, _)) = robot.ik(&config, &ee_pose, x0.to_vec(), &Isometry3::identity()) {
            v.leak().as_ptr()
        } else {
            std::ptr::null()
        }
    }
}
