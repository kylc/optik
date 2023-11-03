#pragma once

#include <Eigen/Dense>
#include <string>

namespace optik {

namespace detail {
//! Opaque struct to be used exclusively to interface with the underlying C API.
struct robot;
}  // namespace detail

//! Sets the number of threads to be used for various parallel operations within
//! subsequent library calls.
//!
//! NOTE: This must be called at most one time and before any parallel library
//! operations are called (e.g. solving IK).
void SetParallelism(unsigned int n);

class Robot final {
 public:
  ~Robot();

  //! Load a URDF model file from the given path, and build a chain from the
  //! named base to the named end-effector link.
  //!
  //! Panics if the file does not exist or does not contain a valid model file,
  //! or if either of the links don't exist.
  static Robot FromUrdfFile(const std::string& path,
                            const std::string& base_link,
                            const std::string& ee_link);

  //! Draw a random generalized position vector from a uniform distribution
  //! subject to the joint limits.
  Eigen::VectorXd RandomConfiguration() const noexcept;

  //! Compute the pose of the end-effector link w.r.t. the base link, expressed
  //! in the base link.
  //!
  //! Panics if `q` does not match the number of generalized positions.
  Eigen::Isometry3d DoFk(const Eigen::VectorXd& q) const;

  //! Attempts to solve for a joint configuration `q` for which `FK(q) =
  //! target`.
  //!
  //! Parameter `x0` may be provided as a seed for the optimization.
  //!
  //! If `true` is returned, then out-parameter `q` will contain an inverse
  //! kinematics solution. Otherwise, the solver has not converged and `q` will
  //! be left untouched.
  bool DoIk(const Eigen::Isometry3d& target, const Eigen::VectorXd& x0,
            Eigen::VectorXd* q) const;

  //! Returns the size of the generalized position vector of this robot.
  unsigned int num_positions() const noexcept;

 private:
  //! Private constructor because callers should use the named factory methods
  //! instead.
  Robot(detail::robot* inner);

  detail::robot* inner_;
};

}  // namespace optik
