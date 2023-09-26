#!/usr/bin/env python3

"""
Demonstration of the usage of the OptIK library via its Python interface.  Loads
a model, configures the solver, and performs forward and inverse kinematics.

Usage:
    Run this script by providing your own URDF and kinematic chain:

        $ python example.py <my_robot.urdf> <base_link> <ee_link>
"""

import sys
import numpy as np

import optik

robot = optik.load_model(*sys.argv[1:4])
config = optik.SolverConfig(xtol_abs=1e-20)

# Generate a target pose which is known to be valid, but don't tell the
# optimizer anything about the joint angles we used to get there!
q_target = np.random.uniform(*robot.joint_limits())
ee_pose = robot.fk(q_target)

# Now work backwards and try to solve for a joint configuration which achieves
# the desired pose, from a random seed.
q_seed = np.random.uniform(*robot.joint_limits())
q_opt, c = robot.ik(config, ee_pose, q_seed)

np.set_printoptions(precision=2)
print("EE target pose:")
print(np.array(ee_pose))
print("EE solution pose:")
print(np.array(robot.fk(q_opt)))
print(f"q_seed: {np.array(q_seed)}")
print(f"q_opt:  {np.array(q_opt)}")
print(f"c:      {np.array([c])}")
