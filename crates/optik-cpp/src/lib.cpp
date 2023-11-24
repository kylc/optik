#include <stdexcept>

#include "optik.hpp"

extern "C" {

extern void optik_set_parallelism(unsigned int n);

extern optik::detail::robot* optik_robot_from_urdf_file(const char* path,
                                                        const char* base_link,
                                                        const char* ee_link);
extern void optik_robot_free(optik::detail::robot* robot);

extern unsigned int optik_robot_dof(const optik::detail::robot* robot);
extern double* optik_robot_joint_limits(const optik::detail::robot* robot);
extern double* optik_robot_random_configuration(
    const optik::detail::robot* robot);
extern double* optik_robot_fk(const optik::detail::robot* robot,
                              const double* x);
extern double* optik_robot_ik(const optik::detail::robot* robot,
                              const optik::SolverConfig* config,
                              const double* target, const double* x0);
}

namespace optik {

void SetParallelism(unsigned int n) { optik_set_parallelism(n); }

Robot::~Robot() { optik_robot_free(inner_); }

Robot Robot::FromUrdfFile(const std::string& path, const std::string& base_link,
                          const std::string& ee_link) {
  return optik_robot_from_urdf_file(path.c_str(), base_link.c_str(),
                                    ee_link.c_str());
}

Eigen::VectorXd Robot::RandomConfiguration() const noexcept {
  double* q_data = optik_robot_random_configuration(inner_);
  Eigen::VectorXd q = Eigen::Map<Eigen::VectorXd>(q_data, num_positions());
  free(q_data);

  return q;
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
  return optik_robot_dof(inner_);
}

Robot::Robot(struct detail::robot* inner) : inner_(inner) {}

}  // namespace optik
