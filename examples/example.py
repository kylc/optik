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
from optikpy import Robot, SolverConfig

robot = Robot.from_urdf_file(*sys.argv[1:4])
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
    q_opt, c = robot.ik(config, target_ee_pose, x0)
    end = time.time()

    if q_opt is not None:
        total_time += end - start
        print("Total time: {}us (to {:.1e})".format(int(1e6 * (end - start)), c))

print("Average time: {}us".format(int(1e6 * total_time / N)))
