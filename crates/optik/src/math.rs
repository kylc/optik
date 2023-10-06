#![allow(non_snake_case)]

use std::f64::consts::PI;

use nalgebra::{Isometry3, Matrix3, Matrix6, Vector3, Vector6};

const EPSILON: f64 = 1e-5;
const _1_6: f64 = 1.0 / 6.0;
const _1_8: f64 = 0.125;
const _1_12: f64 = 1.0 / 12.0;
const _1_15: f64 = 1.0 / 15.0;
const _1_24: f64 = 1.0 / 24.0;
const _1_48: f64 = 1.0 / 48.0;
const _1_120: f64 = 1.0 / 120.0;
const _1_720: f64 = 1.0 / 720.0;

pub mod so3 {
    use super::*;

    /// Hat operator.
    /// Goes from so3 parameterization to so3 element (skew-symmetric matrix).
    #[rustfmt::skip]
    pub fn hat(w: Vector3<f64>) -> Matrix3<f64> {
        Matrix3::new(
            0.0,  -w.z,   w.y,
            w.z,   0.0,  -w.x,
            -w.y,  w.x,   0.0,
        )
    }

    /// Squared hat operator (`hat_2(w) == hat(w) * hat(w)`).
    /// Result is a symmetric matrix.
    #[rustfmt::skip]
    pub fn hat_2(w: Vector3<f64>) -> Matrix3<f64> {
        let w11 = w.x * w.x;
        let w12 = w.x * w.y;
        let w13 = w.x * w.z;
        let w22 = w.y * w.y;
        let w23 = w.y * w.z;
        let w33 = w.z * w.z;
        Matrix3::new(
            -w22 - w33,   w12,         w13,
             w12,        -w11 - w33,   w23,
             w13,         w23,        -w11 - w22,
        )
    }

    pub fn right_jacobian(omega: Vector3<f64>) -> Matrix3<f64> {
        let theta_sq = omega.norm_squared();
        let omega_hat = hat(omega);
        let (alpha, diag_value) = if theta_sq < EPSILON {
            let alpha = _1_12 + theta_sq * _1_720;
            let diag_value = 0.5 * (2. - theta_sq * _1_6);

            (alpha, diag_value)
        } else {
            let theta = theta_sq.sqrt();
            let (st, ct) = theta.sin_cos();
            let st_1mct = st / (1. - ct);

            let alpha = 1. / theta_sq - st_1mct / (2. * theta);
            let diag_value = 0.5 * (theta * st_1mct);

            (alpha, diag_value)
        };

        alpha * omega * omega.transpose()
            + Matrix3::from_diagonal_element(diag_value)
            + 0.5 * omega_hat
    }
}

pub mod se3 {
    use super::*;

    // Copied and modified from https://github.com/mpizenberg/visual-odometry-rs
    pub fn log(iso: Isometry3<f64>) -> Vector6<f64> {
        let imag_vector = iso.rotation.vector();
        let imag_norm_2 = imag_vector.norm_squared();
        let real_factor = iso.rotation.scalar();
        if imag_norm_2 < EPSILON {
            let theta_by_imag_norm = 2.0 / real_factor; // TAYLOR
            let w = theta_by_imag_norm * imag_vector;
            let (omega, omega_2) = (so3::hat(w), so3::hat_2(w));
            let x_2 = imag_norm_2 / (real_factor * real_factor);
            let coef_omega_2 = _1_12 * (1.0 + _1_15 * x_2); // TAYLOR
            let v_inv = Matrix3::identity() - 0.5 * omega + coef_omega_2 * omega_2;
            let xi_v = v_inv * iso.translation.vector;

            Vector6::new(xi_v[0], xi_v[1], xi_v[2], w[0], w[1], w[2])
        } else {
            let imag_norm = imag_norm_2.sqrt();
            let theta = if real_factor.abs() < EPSILON {
                let alpha = real_factor.abs() / imag_norm;
                real_factor.signum() * (PI - 2.0 * alpha) // TAYLOR
            } else {
                // Is this correct? should I use atan2 instead?
                2.0 * (imag_norm / real_factor).atan()
            };
            let theta_2 = theta * theta;
            let w = (theta / imag_norm) * imag_vector;
            let (omega, omega_2) = (so3::hat(w), so3::hat_2(w));
            let coef_omega_2 = (1.0 - 0.5 * theta * real_factor / imag_norm) / theta_2;
            let v_inv = Matrix3::identity() - 0.5 * omega + coef_omega_2 * omega_2;
            let xi_v = v_inv * iso.translation.vector;

            Vector6::new(xi_v[0], xi_v[1], xi_v[2], w[0], w[1], w[2])
        }
    }

    // Reference: Pinocchio
    fn jacobian_upper_right_block(iso: Isometry3<f64>) -> Matrix3<f64> {
        let w = log(iso).fixed_rows::<3>(3).into_owned();
        let t = w.norm();
        let t2 = t * t;

        let (beta, beta_dot_over_theta) = if t2 < EPSILON {
            let beta = 1. / 12. + t2 / 720.;
            let beta_dot_over_theta = 1. / 360.;

            (beta, beta_dot_over_theta)
        } else {
            let tinv = 1. / t;
            let t2inv = tinv * tinv;
            let (st, ct) = t.sin_cos();
            let inv_2_2ct = 1. / (2. * (1. - ct));

            let beta = t2inv - st * tinv * inv_2_2ct;
            let beta_dot_over_theta = -2. * t2inv * t2inv + (1. + st * tinv) * t2inv * inv_2_2ct;

            (beta, beta_dot_over_theta)
        };

        let p = iso.translation.vector;
        let w_p = w.dot(&p);
        let v3_tmp = (beta_dot_over_theta * w_p) * w - (t2 * beta_dot_over_theta + 2. * beta) * p;

        let C = v3_tmp * w.transpose()
            + beta * w * p.transpose()
            + w_p * beta * Matrix3::identity()
            + so3::hat(0.5 * p);
        C * so3::right_jacobian(w)
    }

    // References: [1] and [2]
    //
    // [1]: https://github.com/artivis/manif
    // [2]: https://github.com/strasdat/Sophus
    pub fn right_jacobian(tfm: Isometry3<f64>) -> Matrix6<f64> {
        let omega = log(tfm).fixed_rows::<3>(3).into_owned();

        let j = so3::right_jacobian(omega);
        let q = jacobian_upper_right_block(tfm);

        let mut u = Matrix6::zeros();
        u.fixed_slice_mut::<3, 3>(0, 0).copy_from(&j);
        u.fixed_slice_mut::<3, 3>(0, 3).copy_from(&q);
        u.fixed_slice_mut::<3, 3>(3, 3).copy_from(&j);
        u
    }
}
