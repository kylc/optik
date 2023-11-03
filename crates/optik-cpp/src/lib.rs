use std::{
    ffi::{c_char, c_double, c_uint, CStr},
    mem,
};

use nalgebra::{DMatrix, Isometry3, Translation3, UnitQuaternion};

use optik::{Robot, SolverConfig};

fn to_str(c_str: *const c_char) -> &'static str {
    unsafe { CStr::from_ptr(c_str).to_str().unwrap() }
}

#[no_mangle]
extern "C" fn optik_set_parallelism(n: c_uint) {
    optik::set_parallelism(n as usize)
}

#[no_mangle]
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

#[no_mangle]
extern "C" fn optik_robot_free(robot: *mut Robot) {
    unsafe {
        drop(Box::from_raw(robot));
    }
}

#[no_mangle]
extern "C" fn optik_robot_dof(robot: *const Robot) -> c_uint {
    unsafe {
        let robot = robot.as_ref().unwrap();
        robot.num_positions() as c_uint
    }
}

#[no_mangle]
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

#[no_mangle]
extern "C" fn optik_robot_fk(robot: *const Robot, x: *const c_double) -> *const c_double {
    unsafe {
        let robot = robot.as_ref().unwrap();
        let x = std::slice::from_raw_parts(x, robot.num_positions());

        let ee_pose = robot.fk(x);

        ee_pose.to_matrix().data.as_slice().to_vec().leak().as_ptr()
    }
}

#[no_mangle]
extern "C" fn optik_robot_random_configuration(robot: *const Robot) -> *const c_double {
    unsafe {
        let robot = robot.as_ref().unwrap();
        let q = robot.random_configuration(&mut rand::thread_rng());
        q.leak().as_ptr()
    }
}

#[no_mangle]
extern "C" fn optik_robot_ik(
    robot: *const Robot,
    target: *const c_double,
    x0: *const c_double,
) -> *const c_double {
    unsafe {
        let robot = robot.as_ref().unwrap();
        let x0 = std::slice::from_raw_parts(x0, robot.num_positions());

        let target = std::slice::from_raw_parts(target, 4 * 4);
        let m = DMatrix::from_column_slice(4, 4, target);

        let t = Translation3::from(m.fixed_slice::<3, 1>(0, 3).into_owned());
        let r = UnitQuaternion::from_matrix(&m.fixed_slice::<3, 3>(0, 0).into_owned());

        let ee_pose = Isometry3::from_parts(t, r);

        let config = SolverConfig {
            ..Default::default()
        };
        if let Some((v, _)) = robot.ik(&config, &ee_pose, x0.to_vec()) {
            v.leak().as_ptr()
        } else {
            std::ptr::null()
        }
    }
}
