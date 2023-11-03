// g++ -Wall -Wextra -std=c++11 -O2 $(pkg-config --cflags eigen3) -Icrates/optik-cpp/include examples/example.cpp target/release/liboptikcpp.a -lgfortran

#include <chrono>
#include <iostream>
#include <thread>

#include <Eigen/Dense>

#include "optik.hpp"

int main(int, char**)
{
  using namespace std::chrono;

  optik::SetParallelism(std::thread::hardware_concurrency() / 2);
  const auto robot = optik::Robot::FromUrdfFile("models/ur3e.urdf", "ur_base_link", "ur_ee_link");

  int n = 10000;
  int total_us = 0;

  for (int i = 0; i < n; i++)
  {
    Eigen::VectorXd x0 = robot.RandomConfiguration();
    Eigen::VectorXd q_target = robot.RandomConfiguration();
    Eigen::Isometry3d target_ee_pose = robot.DoFk(q_target);

    auto start = steady_clock::now();
    Eigen::VectorXd q;
    robot.DoIk(target_ee_pose, x0, &q);
    auto end = steady_clock::now();

    auto us = duration_cast<microseconds>(end - start).count();
    total_us += us;
    std::cout << "Total time: " << us << "us" << std::endl;
  }

  std::cout << "Average time: " << (total_us / n) << "us" << std::endl;
}
