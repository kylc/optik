use nalgebra::Isometry3;

use crate::{kinematics::ForwardKinematics, math::se3, Robot};

pub fn objective(_robot: &Robot, tfm_target: &Isometry3<f64>, fk: &ForwardKinematics) -> f64 {
    // Compute the pose error w.r.t. the actual pose:
    //
    //   X_AT = X_WA^1 * X_WT
    let tfm_actual = fk.ee_tfm();
    let tfm_error = tfm_target.inv_mul(&tfm_actual);

    // Minimize the sqaure Euclidean distance of the log pose error. We choose
    // to use the square distance due to its smoothness.
    se3::log(&tfm_error).norm_squared()
}

/// Compute the gradient `g` w.r.t. the local parameterization.
pub fn objective_grad(
    robot: &Robot,
    tfm_target: &Isometry3<f64>,
    fk: &ForwardKinematics,
    g: &mut [f64],
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
    let fdot_g = 2.0 * se3::log(&tfm_error).transpose();
    let grad_h = fdot_g * j_task;

    g.copy_from_slice(grad_h.as_slice());
}
