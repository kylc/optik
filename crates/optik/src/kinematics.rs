use nalgebra::{Isometry3, Matrix6xX, RealField, Translation3, Unit, UnitQuaternion, Vector3};

#[derive(Debug, Clone)]
pub struct Joint<T: RealField> {
    offset: Isometry3<T>,
    typ: JointType<T>,
}

#[derive(Debug, Clone)]
enum JointType<T: RealField> {
    Revolute { axis: Unit<Vector3<T>> },
    Prismatic { axis: Unit<Vector3<T>> },
}

impl<T: RealField + Copy> Joint<T> {
    pub fn revolute(offset: Isometry3<T>, axis: Unit<Vector3<T>>) -> Joint<T> {
        Joint {
            offset,
            typ: JointType::Revolute { axis },
        }
    }

    pub fn prismatic(offset: Isometry3<T>, axis: Unit<Vector3<T>>) -> Joint<T> {
        Joint {
            offset,
            typ: JointType::Prismatic { axis },
        }
    }

    fn local_transform(&self, q: &[T]) -> Isometry3<T> {
        match self.typ {
            JointType::Revolute { axis } => {
                // let s = q[0].sin();
                // let c = q[0].cos();

                // let v = Vector3::x();
                // let k = &axis;
                // let v_rot =
                //     v * c + k.cross(&v) * s + k.scale(k.dot(&v) * (nalgebra::one::<T>() - c));
                self.offset * UnitQuaternion::from_axis_angle(&axis, q[0])
            }
            JointType::Prismatic { axis } => self.offset * Translation3::from(axis.scale(q[0])),
        }
    }

    pub fn nq(&self) -> usize {
        match self.typ {
            JointType::Revolute { .. } => 1,
            JointType::Prismatic { .. } => 1,
        }
    }
}

pub fn joint_kinematics<'a>(
    chain: &'a [Joint<f64>],
    q: &'a [f64],
) -> impl Iterator<Item = Isometry3<f64>> + 'a {
    struct State {
        qidx: usize,
        tfm: Isometry3<f64>,
    }

    chain.iter().scan(
        State {
            tfm: Isometry3::identity(),
            qidx: 0,
        },
        |state, joint| {
            state.tfm *= joint.local_transform(&q[state.qidx..(state.qidx + joint.nq())]);
            state.qidx += joint.nq();

            Some(state.tfm)
        },
    )
}

pub fn joint_jacobian(
    arm: &[Joint<f64>],
    ee_offset: &Isometry3<f64>,
    tfms: &[Isometry3<f64>],
) -> Matrix6xX<f64> {
    let dof = arm.len();
    let tfm_w_ee = tfms.last().unwrap() * ee_offset;

    let mut m = Matrix6xX::zeros(dof);
    for (col, joint) in arm.iter().enumerate() {
        let tfm_w_i = tfms[col];

        match joint.typ {
            JointType::Revolute { axis, .. } => {
                let a_i = tfm_w_i.rotation * axis;
                let dp_i = a_i.cross(&(tfm_w_ee.translation.vector - tfm_w_i.translation.vector));

                // in local frame
                let a_i = tfm_w_ee.inverse_transform_vector(&a_i);
                let dp_i = tfm_w_ee.inverse_transform_vector(&dp_i);

                m.fixed_slice_mut::<3, 1>(0, col).copy_from(&dp_i);
                m.fixed_slice_mut::<3, 1>(3, col).copy_from(&a_i);
            }
            JointType::Prismatic { .. } => todo!(),
        };
    }

    m
}

/// A cache structure which stores the expensive forward kinematics computation
/// for its respective generalized joint configuration.
#[derive(Clone, Default)]
pub struct KinematicsCache {
    q: Vec<f64>,
    frames: Vec<Isometry3<f64>>,
}

impl KinematicsCache {
    /// If a cache entry exists for the given joint configuration vector `q`,
    /// return that. Otherwise, compute the forward kinematics, replace the the
    /// cache with the results, and return them.
    pub fn get_or_update(&mut self, joints: &[Joint<f64>], q: &[f64]) -> &[Isometry3<f64>] {
        if self.q != q {
            // Clear and extend the cached vectors to prevent any allocations.
            self.q.clear();
            self.q.extend_from_slice(q);

            self.frames.clear();
            self.frames.extend(joint_kinematics(joints, q));
        }

        &self.frames
    }
}
