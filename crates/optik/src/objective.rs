use nalgebra::{Isometry3, Vector3, Vector6};

use crate::{kinematics::ForwardKinematics, math::se3, Robot};

const IDENTITY_EPS: f64 = 1e-20;

fn apply_weighting(
    mut e: Vector6<f64>,
    tfm_target: &Isometry3<f64>,
    linear_weight: &Vector3<f64>,
    angular_weight: &Vector3<f64>,
) -> Vector6<f64> {
    if !linear_weight.is_identity(IDENTITY_EPS) {
        // Rotate the error vector into the world frame.
        let e_lin_w = tfm_target * e.fixed_rows::<3>(0).clone_owned();

        // Apply the weights.
        let e_lin_w_scaled = e_lin_w.component_mul(&linear_weight);

        // Rotate back into the target frame.
        let e_lin_scaled = tfm_target.inverse_transform_vector(&e_lin_w_scaled);
        e.fixed_rows_mut::<3>(0).copy_from(&e_lin_scaled);
    }

    if !angular_weight.is_identity(IDENTITY_EPS) {
        // Rotate the error vector into the world frame.
        let e_ang_w = tfm_target * e.fixed_rows::<3>(3).clone_owned();

        // Apply the weights.
        let e_ang_w_scaled = e_ang_w.component_mul(&angular_weight);

        // Rotate back into the target frame.
        let e_ang_scaled = tfm_target.inverse_transform_vector(&e_ang_w_scaled);
        e.fixed_rows_mut::<3>(3).copy_from(&e_ang_scaled);
    }

    e
}

pub fn objective(
    _robot: &Robot,
    tfm_target: &Isometry3<f64>,
    fk: &ForwardKinematics,
    linear_weight: Vector3<f64>,
    angular_weight: Vector3<f64>,
) -> f64 {
    // Compute the pose error w.r.t. the target pose:
    let x_world_act = fk.ee_tfm();
    let x_target_act = tfm_target.inv_mul(&x_world_act);

    let e = se3::log(&x_target_act);
    let e = apply_weighting(e, &tfm_target, &linear_weight, &angular_weight);

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
    linear_weight: Vector3<f64>,
    angular_weight: Vector3<f64>,
) {
    // Pose error is computed as in the objective function.
    let x_world_act = fk.ee_tfm();
    let x_target_act = tfm_target.inv_mul(&x_world_act);

    // We compute the Jacobian of our task w.r.t. the joint angles.
    //
    // - Jqdot: the local (body-coordinate) frame Jacobian of X w.r.t. q
    // - Jlog6: the right (body-coordinate) Jacobian of log6(X)
    // - Jtask: the Jacobian of our pose tracking task
    //
    //   Jtask(q) = Jlog6(X) * J(q)
    let j_qdot = robot.joint_jacobian(fk);
    let j_log6 = se3::right_jacobian(&x_target_act);
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

    let e = se3::log(&x_target_act);

    let linear_weight_2 = linear_weight.component_mul(&linear_weight);
    let angular_weight_2 = angular_weight.component_mul(&angular_weight);
    let e = apply_weighting(e, &tfm_target, &linear_weight_2, &angular_weight_2);

    let fdot_g = 2.0 * e.transpose();
    let grad_h = fdot_g * j_task;

    g.copy_from_slice(grad_h.as_slice());
}
