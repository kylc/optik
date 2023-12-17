#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include <Eigen/Dense>

#include "optik.hpp"

int main(int argc, char **argv) {
  using namespace std::chrono;

  if (argc < 4) {
    std::cerr << "usage: " << argv[0] << " <urdf_file> <base_link> <ee_link>"
              << std::endl;
    std::exit(1);
  }

  auto robot = optik::Robot::FromUrdfFile(argv[1], argv[2], argv[3]);
  robot.SetParallelism(std::thread::hardware_concurrency() / 2);

  optik::SolverConfig config;

  int n = 10000;
  int total_us = 0;

  for (int i = 0; i < n; i++) {
    Eigen::VectorXd x0 = robot.RandomConfiguration();
    Eigen::VectorXd q_target = robot.RandomConfiguration();
    Eigen::Isometry3d target_ee_pose = robot.DoFk(q_target);

    auto start = steady_clock::now();
    Eigen::VectorXd q;
    robot.DoIk(config, target_ee_pose, x0, &q);
    auto end = steady_clock::now();

    auto us = duration_cast<microseconds>(end - start).count();
    total_us += us;
    std::cout << "Total time: " << us << "us" << std::endl;
  }

  std::cout << "Average time: " << (total_us / n) << "us" << std::endl;
}
