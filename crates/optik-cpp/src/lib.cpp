#include <stdexcept>

#include "optik.hpp"

extern "C" {

extern optik::detail::robot* optik_robot_from_urdf_file(const char* path,
                                                        const char* base_link,
                                                        const char* ee_link);
extern optik::detail::robot* optik_robot_from_urdf_str(const char* urdf,
                                                       const char* base_link,
                                                       const char* ee_link);
extern void optik_robot_free(optik::detail::robot* robot);

extern void optik_robot_set_parallelism(optik::detail::robot*, unsigned int n);
extern unsigned int optik_robot_num_positions(
    const optik::detail::robot* robot);
extern double* optik_robot_joint_limits(const optik::detail::robot* robot);
extern double* optik_robot_random_configuration(
    const optik::detail::robot* robot);
extern double* optik_robot_joint_jacobian(const optik::detail::robot* robot,
                                          const double* x);
extern double* optik_robot_fk(const optik::detail::robot* robot,
                              const double* x);
extern double* optik_robot_ik(const optik::detail::robot* robot,
                              const optik::SolverConfig* config,
                              const double* target, const double* x0);
}

namespace optik {

Robot::Robot(Robot&& other) : inner_(std::move(other.inner_)) {
  // Leave a sentinel value in the pointer so that we don't accidentally to
  // double free it in the moved-from object destructor.
  other.inner_ = nullptr;
};

Robot::~Robot() {
  if (inner_ != nullptr) {
    optik_robot_free(inner_);
  }
}

Robot& Robot::operator=(Robot&& other) {
  inner_ = std::move(other.inner_);
  other.inner_ = nullptr;  // same as in move constructor
  return *this;
};

Robot Robot::FromUrdfFile(const std::string& path, const std::string& base_link,
                          const std::string& ee_link) {
  return optik_robot_from_urdf_file(path.c_str(), base_link.c_str(),
                                    ee_link.c_str());
}

Robot Robot::FromUrdfStr(const std::string& urdf, const std::string& base_link,
                         const std::string& ee_link) {
  return optik_robot_from_urdf_str(urdf.c_str(), base_link.c_str(),
                                   ee_link.c_str());
}

void Robot::SetParallelism(unsigned int n) {
  optik_robot_set_parallelism(inner_, n);
}

Eigen::VectorXd Robot::RandomConfiguration() const noexcept {
  double* q_data = optik_robot_random_configuration(inner_);
  Eigen::VectorXd q = Eigen::Map<Eigen::VectorXd>(q_data, num_positions());
  free(q_data);

  return q;
}

Eigen::Matrix<double, 6, Eigen::Dynamic> Robot::JointJacobian(
    const Eigen::VectorXd& q) const {
  using JacobianMatrix = Eigen::Matrix<double, 6, Eigen::Dynamic>;

  if (q.size() != num_positions()) {
    throw std::runtime_error("dof mismatch");
  }

  double* jac_data = optik_robot_joint_jacobian(inner_, q.data());
  JacobianMatrix jac = Eigen::Map<JacobianMatrix>(jac_data, 6, num_positions());
  free(jac_data);

  return jac;
}

Eigen::Isometry3d Robot::DoFk(const Eigen::VectorXd& q) const {
  if (q.size() != num_positions()) {
    throw std::runtime_error("dof mismatch");
  }

  double* m_data = optik_robot_fk(inner_, q.data());
  Eigen::Isometry3d t;
  t.matrix() = Eigen::Map<Eigen::Matrix4d>(m_data);
  free(m_data);

  return t;
}

bool Robot::DoIk(const SolverConfig& config, const Eigen::Isometry3d& target,
                 const Eigen::VectorXd& x0, Eigen::VectorXd* q) const {
  if (x0.size() != num_positions()) {
    throw std::runtime_error("dof mismatch");
  }

  double* q_data =
      optik_robot_ik(inner_, &config, target.matrix().data(), x0.data());
  if (q_data != nullptr) {
    *q = Eigen::Map<Eigen::VectorXd>(q_data, num_positions());
    free(q_data);
    return true;
  } else {
    return false;
  }
}

unsigned int Robot::num_positions() const noexcept {
  return optik_robot_num_positions(inner_);
}

Robot::Robot(struct detail::robot* inner) : inner_(inner) {}

}  // namespace optik
