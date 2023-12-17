use nalgebra::{Isometry3, Matrix6xX, Translation, Translation3, Unit, UnitQuaternion, Vector3};
use petgraph::{
    algo::is_cyclic_directed,
    prelude::{DiGraph, EdgeIndex},
};

#[derive(Clone)]
pub struct KinematicChain {
    pub joints: Vec<Joint>,
}

impl KinematicChain {
    /// Parse a kinematic chain from the given URDF representation spanning from
    /// the given base link to the EE link.
    ///
    /// Panics if one of the links does not exist or there is no path between
    /// them.
    pub fn from_urdf(robot: &urdf_rs::Robot, base_link: &str, ee_link: &str) -> Self {
        let graph = parse_urdf(robot);

        assert!(!is_cyclic_directed(&graph), "robot model contains loops");

        // Identify the links along the chain from base to EE.
        let sorted_links = {
            let base_link_ix = graph
                .node_indices()
                .find(|&ix| graph[ix].name == base_link)
                .unwrap_or_else(|| panic!("base link '{}' does not exist", base_link));

            let ee_link_ix = graph
                .node_indices()
                .find(|&ix| graph[ix].name == ee_link)
                .unwrap_or_else(|| panic!("EE link '{}' does not exist", base_link));

            petgraph::algo::astar(
                &graph,
                base_link_ix,
                |ix| ix == ee_link_ix,
                |_| 1.0,
                |_| 0.0,
            )
            .expect("no path from base to EE link")
            .1
        };

        // Identify the serial chain of joints that connects the links of
        // interest.
        let joints = sorted_links.windows(2).map(|links| {
            let joint = graph.find_edge(links[0], links[1]).unwrap();
            graph[joint].clone()
        });

        // OPTIMIZATION: Fold fixed joints to avoid recomputing constant offsets
        // on every forward kinematic computation.
        let joints = joints
            .scan(Isometry3::identity(), |tfm, joint| match joint.typ {
                JointType::Fixed => {
                    // Accumulate the transforms from successive fixed joints.
                    *tfm = joint.origin * *tfm;
                    Some(None)
                }
                _ => {
                    // When we encounter a non-fixed joint, apply the
                    // accumulated fixed joint transforms (if any) and then
                    // reset the accumulation.
                    let new_joint = Joint {
                        origin: joint.origin * *tfm,
                        ..joint
                    };
                    *tfm = Isometry3::identity();

                    Some(Some(new_joint))
                }
            })
            .flatten();

        let joints: Vec<Joint> = joints.collect();

        // We don't care to support empty chains.
        assert!(!joints.is_empty(), "kinematic chain is empty");

        Self { joints }
    }

    pub fn nq(&self) -> usize {
        self.joints.iter().map(|j| j.typ.nq()).sum()
    }

    pub fn forward_kinematics(&self, q: &[f64]) -> ForwardKinematics {
        struct State {
            qidx: usize,
            tfm: Isometry3<f64>,
        }

        let joint_tfms: Vec<Isometry3<f64>> = self
            .joints
            .iter()
            .scan(
                State {
                    tfm: Isometry3::identity(),
                    qidx: 0,
                },
                |state, joint| {
                    let qrange = state.qidx..(state.qidx + joint.typ.nq());
                    let joint_q = &q[qrange];

                    state.tfm *= joint.origin * joint.typ.local_transform(joint_q);
                    state.qidx += joint.typ.nq();

                    Some(state.tfm)
                },
            )
            .collect();

        let ee_tfm = *joint_tfms.last().unwrap();
        ForwardKinematics { joint_tfms, ee_tfm }
    }

    pub fn joint_jacobian(&self, fk: &ForwardKinematics) -> Matrix6xX<f64> {
        let dof = self.nq();
        let tfm_w_ee = fk.ee_tfm();

        let mut m = Matrix6xX::zeros(dof);
        let mut col = 0;
        for (i, joint) in self.joints.iter().enumerate() {
            let tfm_w_i = fk.joint_tfms[i];

            match joint.typ {
                JointType::Revolute(axis) => {
                    let a_i = tfm_w_i.rotation * axis;
                    let dp_i =
                        a_i.cross(&(tfm_w_ee.translation.vector - tfm_w_i.translation.vector));

                    // in local frame
                    let a_i = tfm_w_ee.inverse_transform_vector(&a_i);
                    let dp_i = tfm_w_ee.inverse_transform_vector(&dp_i);

                    m.fixed_view_mut::<3, 1>(0, col).copy_from(&dp_i);
                    m.fixed_view_mut::<3, 1>(3, col).copy_from(&a_i);
                }
                JointType::Prismatic(_) => todo!(),
                JointType::Fixed => {}
            };

            col += joint.typ.nq();
        }

        m
    }
}

#[derive(Default, Clone)]
pub struct ForwardKinematics {
    joint_tfms: Vec<Isometry3<f64>>,
    ee_tfm: Isometry3<f64>,
}

impl ForwardKinematics {
    pub fn joint_tfm(&self, joint_ix: EdgeIndex) -> &Isometry3<f64> {
        &self.joint_tfms[joint_ix.index()]
    }

    pub fn ee_tfm(&self) -> &Isometry3<f64> {
        &self.ee_tfm
    }
}

#[derive(Clone)]
pub struct Joint {
    pub name: String,
    pub typ: JointType,
    pub limits: (f64, f64),
    pub origin: Isometry3<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum JointType {
    Revolute(Unit<Vector3<f64>>),
    Prismatic(Unit<Vector3<f64>>),
    Fixed,
}

impl JointType {
    pub fn nq(&self) -> usize {
        match self {
            JointType::Revolute(_) => 1,
            JointType::Prismatic(_) => 1,
            JointType::Fixed => 0,
        }
    }

    pub fn local_transform(&self, q: &[f64]) -> Isometry3<f64> {
        match self {
            JointType::Revolute(axis) => {
                let rotation = UnitQuaternion::from_axis_angle(axis, q[0]);
                Isometry3::from_parts(Translation3::identity(), rotation)
            }
            JointType::Prismatic(axis) => {
                let translation = axis.scale(q[0]);
                Isometry3::from_parts(Translation3::from(translation), UnitQuaternion::identity())
            }
            JointType::Fixed => Isometry3::identity(),
        }
    }
}

#[derive(Clone)]
pub struct Link {
    pub name: String,
}

fn urdf_to_tfm(pose: &urdf_rs::Pose) -> Isometry3<f64> {
    let xyz = Vector3::from_row_slice(&pose.xyz.0);
    let rot = UnitQuaternion::from_euler_angles(pose.rpy.0[0], pose.rpy.0[1], pose.rpy.0[2]);
    Isometry3::from_parts(Translation::from(xyz), rot)
}

fn parse_urdf(urdf: &urdf_rs::Robot) -> DiGraph<Link, Joint> {
    let mut graph = DiGraph::<Link, Joint>::new();

    for link in &urdf.links {
        graph.add_node(Link {
            name: link.name.clone(),
        });
    }

    for joint in &urdf.joints {
        let parent_ix = graph
            .node_indices()
            .find(|&l| graph[l].name == joint.parent.link)
            .unwrap_or_else(|| panic!("joint parent link '{}' does not exist", joint.parent.link));
        let child_ix = graph
            .node_indices()
            .find(|&l| graph[l].name == joint.child.link)
            .unwrap_or_else(|| panic!("joint child link '{}' does not exist", joint.parent.link));

        let joint_type = match &joint.joint_type {
            urdf_rs::JointType::Revolute => JointType::Revolute(Unit::new_normalize(
                Vector3::from_row_slice(&joint.axis.xyz.0),
            )),
            urdf_rs::JointType::Prismatic => JointType::Prismatic(Unit::new_normalize(
                Vector3::from_row_slice(&joint.axis.xyz.0),
            )),
            urdf_rs::JointType::Fixed => JointType::Fixed,
            x => panic!("joint type not supported: {:?}", x),
        };

        let limits = if joint.limit.upper - joint.limit.lower > 0.0 {
            (joint.limit.lower, joint.limit.upper)
        } else {
            (f64::NEG_INFINITY, f64::INFINITY)
        };
        let origin = urdf_to_tfm(&joint.origin);

        graph.add_edge(
            parent_ix,
            child_ix,
            Joint {
                name: joint.name.clone(),
                typ: joint_type,
                limits,
                origin,
            },
        );
    }

    graph
}

/// A cache structure which stores the expensive forward kinematics computation
/// for its respective generalized joint configuration.
#[derive(Default, Clone)]
pub struct KinematicsCache {
    q: Vec<f64>,
    kinematics: ForwardKinematics,
}

impl KinematicsCache {
    /// If a cache entry exists for the given joint configuration vector `q`,
    /// return that. Otherwise, compute the forward kinematics, replace the the
    /// cache with the results, and return them.
    pub fn get_or_update(&mut self, chain: &KinematicChain, q: &[f64]) -> &ForwardKinematics {
        if self.q != q {
            // Clear and extend the cached vectors to prevent any allocations.
            self.q.clear();
            self.q.extend_from_slice(q);

            self.kinematics = chain.forward_kinematics(q)
        }

        &self.kinematics
    }
}
