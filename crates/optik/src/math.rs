#![allow(non_snake_case)]

use nalgebra::{Isometry3, Matrix3, Matrix6, UnitQuaternion, Vector3, Vector6};

/// Bound on unstable trigonometric computations, beneath which Taylor
/// polynomials are used as an approximation technique.
const EPSILON: f64 = 1e-6;

pub mod so3 {
    use super::*;

    /// Hat operator ⎣ω⎦ₓ.
    pub fn hat(w: &Vector3<f64>) -> Matrix3<f64> {
        w.cross_matrix()
    }

    /// Efficient computation of the squared hat operator ⎣ω⎦ₓ².
    #[rustfmt::skip]
    pub fn hat_2(w: &Vector3<f64>) -> Matrix3<f64> {
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

    /// The logarithmic map of SO(3), the inverse of the group exponential.
    ///
    /// References:
    ///
    /// [1] “Tables of Useful Lie Group Identities.” Accessed: Dec. 03, 2023.
    /// [Online]. Available:
    /// http://jamessjackson.com/lie_algebra_tutorial/09-lie_group_tables/
    pub fn log(q: &UnitQuaternion<f64>) -> Vector3<f64> {
        // Handle double cover of SO(3), i.e. q and -q represent the same
        // rotation. Ensure we have a quaternion with a real part w > 0.
        let (w, v) = if q.scalar() >= 0.0 {
            (q.scalar(), q.imag())
        } else {
            (-q.scalar(), -q.imag())
        };

        let v_norm_2 = v.norm_squared();

        #[rustfmt::skip]
        let theta_over_v_norm = if v_norm_2 > EPSILON {
            let v_norm = v_norm_2.sqrt();
            v_norm.atan2(w) / v_norm
        } else {
            // Taylor series expansion of arctan(||v|| / w) / ||v||
            1. / w
                - 1. / (3. * w.powi(3)) * v_norm_2
                + 1. / (5. * w.powi(5)) * v_norm_2.powi(2)
        };

        2.0 * v * theta_over_v_norm
    }

    /// The right Jacobian of ∂/∂R log(R), R ∈ SO(3).
    ///
    /// References:
    ///
    /// [1] “Tables of Useful Lie Group Identities.” Accessed: Dec. 03, 2023.
    /// [Online]. Available:
    /// http://jamessjackson.com/lie_algebra_tutorial/09-lie_group_tables/
    pub fn right_jacobian(w: &Vector3<f64>) -> Matrix3<f64> {
        let theta_2 = w.norm_squared();
        let theta_4 = theta_2 * theta_2;
        let theta = theta_2.sqrt();
        let (s, c) = theta.sin_cos();

        let a = if theta_2 > EPSILON {
            s / theta
        } else {
            // Taylor series expansion of sin(θ) / θ
            1. - 1. / 6. * theta_2 + 1. / 120.0 * theta_4
        };
        let b = if theta_2 > EPSILON {
            (1. - c) / theta_2
        } else {
            // Taylor series expansion of (1 - cos(θ)) / θ
            1. / 2. - 1. / 24. * theta_2 + 1. / 720. * theta_4
        };
        let c = (1. - a) / theta_2;
        let e = (b - 2. * c) / (2. * a);

        Matrix3::identity() + 0.5 * hat(w) + e * hat_2(w)
    }
}

pub mod se3 {
    use super::*;

    /// The logarithmic map of SE(3), the inverse of the group exponential.
    ///
    /// References:
    ///
    /// [1] “Tables of Useful Lie Group Identities.” Accessed: Dec. 03, 2023.
    /// [Online]. Available:
    /// http://jamessjackson.com/lie_algebra_tutorial/09-lie_group_tables/
    pub fn log(tfm: &Isometry3<f64>) -> Vector6<f64> {
        let w = so3::log(&tfm.rotation);
        let theta_sq = w.norm_squared();
        let theta = theta_sq.sqrt();

        let p = if theta > EPSILON {
            let (s, c) = theta.sin_cos();
            0.5 * (theta * s) / (1. - c)
        } else {
            // Taylor series expansion of 1/2 * (θ * sin(θ)) / (1 - cos(θ))
            1. - theta_sq / 12. - theta_sq * theta_sq / 720.
        };

        let v_inv =
            Matrix3::identity() - 0.5 * so3::hat(&w) + 1. / theta_sq * (1. - p) * so3::hat_2(&w);
        let v = v_inv * tfm.translation.vector;
        Vector6::new(v[0], v[1], v[2], w[0], w[1], w[2])
    }

    /// The upper-right block of the right Jacobian ∂/∂T log(T).
    ///
    /// References:
    ///
    /// [1] J. Solà, J. Deray, and D. Atchuthan, “A micro Lie theory for state
    /// estimation in robotics.” arXiv, Dec. 08, 2021. Accessed: Jul. 24, 2023.
    /// [Online]. Available: http://arxiv.org/abs/1812.01537
    ///
    /// [2]: https://github.com/stack-of-tasks/pinocchio/blob/v2.9.1/src/spatial/log.hxx
    fn right_jacobian_q_matrix(v: &Vector3<f64>, w: &Vector3<f64>) -> Matrix3<f64> {
        // TODO: This is translated from the Pinocchio source code, but I can't
        // find a good literature reference. The result of this computation
        // seems to disagree with the formula provided in [1].
        let theta = w.norm();
        let theta_2 = f64::powi(theta, 2);
        let theta_4 = theta_2 * theta_2;

        let (a, b) = if theta_2 > EPSILON {
            let (s, c) = theta.sin_cos();

            let s_t = s / theta;
            let inv_1mc = 1. / (2. * (1. - c));

            let a = 1. / theta_2 - s_t * inv_1mc;
            let b = -2. / theta_4 + (1. + s_t) * inv_1mc / theta_2;

            (a, b)
        } else {
            let a = 1. / 12. + theta_2 / 720.;
            let b = 1. / 360.;

            (a, b)
        };

        let d = w.dot(v);
        let c = b * d * w - (theta_2 * b + 2. * a) * v;

        let C = 0.5 * so3::hat(v)
            + c * w.transpose()
            + a * w * v.transpose()
            + d * a * Matrix3::identity();
        let E = so3::right_jacobian(w);

        C * E
    }

    /// The right Jacobian ∂/∂T log(T) is is given by:
    ///
    ///   ⎡ J(θ)  Q(p,θ) ⎤
    ///   ⎣  0     J(θ)  ⎦
    ///
    /// Where:
    ///
    ///   J(θ)   = so3::right_jacobian(θ)
    ///   Q(p,θ) = se3::right_jacobian_q_matrix(p,θ)
    ///
    /// Reference:
    ///
    /// [1] J. Solà, J. Deray, and D. Atchuthan, “A micro Lie theory for state
    /// estimation in robotics.” arXiv, Dec. 08, 2021. Accessed: Jul. 24, 2023.
    /// [Online]. Available: http://arxiv.org/abs/1812.01537
    ///
    /// [2] “Tables of Useful Lie Group Identities.” Accessed: Dec. 03, 2023.
    /// [Online]. Available:
    /// http://jamessjackson.com/lie_algebra_tutorial/09-lie_group_tables/
    pub fn right_jacobian(tfm: &Isometry3<f64>) -> Matrix6<f64> {
        let v = &tfm.translation.vector;
        let w = so3::log(&tfm.rotation);

        let j = so3::right_jacobian(&w);
        let q = se3::right_jacobian_q_matrix(v, &w);

        let mut u = Matrix6::zeros();
        u.fixed_slice_mut::<3, 3>(0, 0).copy_from(&j);
        u.fixed_slice_mut::<3, 3>(0, 3).copy_from(&q);
        u.fixed_slice_mut::<3, 3>(3, 3).copy_from(&j);
        u
    }
}
