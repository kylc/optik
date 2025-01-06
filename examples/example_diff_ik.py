#!/usr/bin/env python3

"""
Demonstration of the differential IK feature of OptIK.  Loads a model,
configures the solver, and steps of diff-IK with varying velocity limits.

Usage:
    Run this script by providing your own URDF and kinematic chain:

        $ python example_diff_ik.py <my_robot.urdf> <base_link> <ee_link>
"""

import sys

import numpy as np
from optik import Robot

np.set_printoptions(suppress=True, precision=2)

urdf_path, base_name, ee_name = sys.argv[1:4]

robot = Robot.from_urdf_file(urdf_path, base_name, ee_name)
n = robot.num_positions()

rng = np.random.default_rng(seed=42)
x0 = rng.uniform(*robot.joint_limits())

for v_max in [0.1, 0.5, 1.0, 10.0]:
    V_tgt = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 1.0])

    if (sol := robot.diff_ik(x0, V_tgt, [v_max] * n)) is not None:
        alpha, v_star = sol

        J = np.array(robot.joint_jacobian(x0))
        R_W = np.array(robot.fk(x0))[:3, :3]
        J_W = np.vstack(
            (
                R_W @ J[:3, :],
                R_W @ J[3:, :],
            )
        )
        V_star = np.matmul(J_W, v_star)

        print("------")
        print("  x0    =", np.array(x0))
        print("  v_max =", np.array(v_max))
        print("  V_tgt =", np.array(V_tgt))
        print("  alpha =", alpha)
        print("  v*    =", np.array(v_star))
        print("  V*    =", np.array(V_star))

        assert 0.0 <= alpha <= 1.0
        np.testing.assert_allclose(V_tgt, V_star / alpha, atol=1e-6)
