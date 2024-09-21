#!/usr/bin/env python3

"""
A simple implementation of Differential Inverse Kinematics, using OptIK and the
Clarabel convex optimization solver.

We formulate a simple quadratic program to solve for joint velocities which
attempt to achieve the desired end-effector velocity, subject to joint velocity
limits.

See: https://manipulation.csail.mit.edu/pick.html#diff_ik_w_constraints

Usage:
    Run this script by providing your own URDF and kinematic chain:

        $ python diff_ik.py <my_robot.urdf> <base_link> <ee_link>
"""

import sys

import clarabel
from scipy import sparse
import numpy as np
from optik import Robot, SolverConfig

np.set_printoptions(precision=2)

urdf_path, base_name, ee_name = sys.argv[1:4]

robot = Robot.from_urdf_file(urdf_path, base_name, ee_name)
config = SolverConfig()

# Parameters
n = robot.num_positions()  # robot position DoF
V = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])  # desired end-effector spatial velocity
qd_max = 0.75  # maximum joint velocity
dt = 0.1  # timestep

# Initial conditions
rng = np.random.default_rng(seed=42)
q0 = rng.uniform(*robot.joint_limits())  # initial configuration


settings = clarabel.DefaultSettings()
settings.verbose = False

q = q0
for t in np.arange(0.0, 1.0, step=dt):
    X_WE = np.array(robot.fk(q))
    R_WE = X_WE[:3, :3]

    # Use OptIK to compute the local frame Jacobian, and then rotate it to be
    # world-aligned since we specify our target velocity in the world frame.
    JEq = np.array(robot.joint_jacobian(q))
    JWq = np.vstack(
        (
            R_WE @ JEq[:3, :],
            R_WE @ JEq[3:, :],
        )
    )

    # Form the constrained least-squares problem:
    #
    #   min_v  || J * qd - V ||^2
    #     s.t. -qd_max <= qd <= qd_max
    H = JWq.T @ JWq
    f = -JWq.T @ V

    # A = [ -I I ]'
    # b = [ v_max v_max ]'
    A = sparse.vstack((-sparse.eye(n), sparse.eye(n))).tocsc()
    b = np.concatenate([qd_max * np.ones(n), qd_max * np.ones(n)])

    cones = [clarabel.NonnegativeConeT(2 * n)]

    # Solve the QP.
    solver = clarabel.DefaultSolver(sparse.csc_matrix(H), f, A, b, cones, settings)
    solution = solver.solve()

    # The solver could fail if it runs out of time, the constraints are
    # infeasible, etc.
    #
    # A non-failure is when the end-effector velocity can not be achieved. This
    # simply results in a high cost.
    assert solution.status == clarabel.SolverStatus.Solved

    # Extract the joint velocity vector from the solution.
    #
    # We can then compute the resulting Cartesian velocity as J * v.
    qd_star = np.array(solution.x)
    V_star = JWq @ qd_star

    q += qd_star * dt

    print(f"--- t={t}")
    print("qd*      = ", qd_star)
    print("V*       = ", V_star)
    print("|V* - V| = ", np.linalg.norm(V_star - V))
    print("p_WE     =", np.array(robot.fk(q))[:3, 3])

    print(f"Solve time: {int(1e6 * solution.solve_time)}µs")
