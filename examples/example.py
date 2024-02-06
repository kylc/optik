#!/usr/bin/env python3

"""
Demonstration of the usage of the OptIK library via its Python interface.  Loads
a model, configures the solver, and performs forward and inverse kinematics.

Usage:
    Run this script by providing your own URDF and kinematic chain:

        $ python example.py <my_robot.urdf> <base_link> <ee_link>
"""

import sys
import time

import numpy as np
from optik import Robot, SolverConfig

urdf_path, base_name, ee_name = sys.argv[1:4]

robot = Robot.from_urdf_file(urdf_path, base_name, ee_name)
config = SolverConfig()

N = 10000

total_time = 0
for i in range(N):
    # Compute a randomized joint configuration to seed the solver.
    x0 = np.random.uniform(*robot.joint_limits())

    # Generate a target pose which is known to be valid, but don't tell the
    # optimizer anything about the joint angles we used to get there!
    q_target = np.random.uniform(*robot.joint_limits())
    target_ee_pose = robot.fk(q_target)

    start = time.time()
    sol = robot.ik(config, target_ee_pose, x0)
    end = time.time()

    if sol is not None:
        q_opt, c = sol

        target_ee_pose = np.reshape(target_ee_pose, (4, 4))
        actual_ee_pose = np.reshape(robot.fk(q_opt), (4, 4))

        total_time += end - start
        print("Solve time: {}us (to {:.1e})".format(int(1e6 * (end - start)), c))

print("Average time: {}us".format(int(1e6 * total_time / N)))
