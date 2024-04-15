use nalgebra::Isometry3;

use crate::{kinematics::ForwardKinematics, math::se3, Robot};

pub fn objective(
    _robot: &Robot,
    tfm_target: &Isometry3<f64>,
    fk: &ForwardKinematics,
    tol_linear: f64,
    tol_angular: f64,
) -> f64 {
    // Compute the pose error w.r.t. the actual pose:
    //
    //   X_AT = X_WA^1 * X_WT
    let tfm_actual = fk.ee_tfm();
    let tfm_error = tfm_target.inv_mul(&tfm_actual);

    let mut e = se3::log(&tfm_error);

    // Linear and/or angular error terms go to zero if they are within the
    // tolerances.
    if tol_linear > 0.0 && e.fixed_rows::<3>(0).norm() <= tol_linear {
        e.fixed_rows_mut::<3>(0).fill(0.0);
    }
    if tol_angular > 0.0 && e.fixed_rows::<3>(3).norm() <= tol_angular {
        e.fixed_rows_mut::<3>(3).fill(0.0);
    }

    // Minimize the sqaure Euclidean distance of the log pose error. We choose
    // to use the square distance due to its smoothness.
    e.norm_squared()
}

/// Compute the gradient `g` w.r.t. the local parameterization.
pub fn objective_grad(
    robot: &Robot,
    tfm_target: &Isometry3<f64>,
    fk: &ForwardKinematics,
    g: &mut [f64],
    tol_linear: f64,
    tol_angular: f64,
) {
    // Pose error is computed as in the objective function.
    let tfm_actual = fk.ee_tfm();
    let tfm_error = tfm_target.inv_mul(&tfm_actual);

    // We compute the Jacobian of our task w.r.t. the joint angles.
    //
    // - Jqdot: the local (body-coordinate) frame Jacobian of X w.r.t. q
    // - Jlog6: the right (body-coordinate) Jacobian of log6(X)
    // - Jtask: the Jacobian of our pose tracking task
    //
    //   Jtask(q) = Jlog6(X) * J(q)
    let j_qdot = robot.joint_jacobian(fk);
    let j_log6 = se3::right_jacobian(&tfm_error);
    let j_task = j_log6 * j_qdot;

    // We must compute the objective function gradient:
    //
    //   ∇h = [ ∂h/∂x1  ...  ∂h/∂xn ]
    //
    // Given the pose task Jacobian of the form:
    //
    //   J = ⎡ ∂f1/∂x1 ... ∂f1/∂xn ⎤
    //       ⎢    .           .    ⎥
    //       ⎢    .           .    ⎥
    //       ⎣ ∂fm/∂x1 ... ∂fm/∂xn ⎦
    //
    // Apply the chain rule to compute the derivative:
    //
    //   g = log(X)
    //   f = || g ||^2
    //   h' = (f' ∘ g) * g' = (2.0 * log6(X)) * J

    let mut e = se3::log(&tfm_error);

    // Linear and/or angular error terms go to zero if they are within the
    // tolerances.
    if tol_linear > 0.0 && e.fixed_rows::<3>(0).norm() <= tol_linear {
        e.fixed_rows_mut::<3>(0).fill(0.0);
    }
    if tol_angular > 0.0 && e.fixed_rows::<3>(3).norm() <= tol_angular {
        e.fixed_rows_mut::<3>(3).fill(0.0);
    }

    let fdot_g = 2.0 * e.transpose();
    let grad_h = fdot_g * j_task;

    g.copy_from_slice(grad_h.as_slice());
}
