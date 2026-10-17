import numpy as np

from urkin.transforms import so3_log

from .conftest import random_configs


def numerical_jacobian(robot, q, tool=None, h=1e-6):
    T0 = robot.fk(q, tool)
    J = np.zeros((6, 6))
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = h
        T1 = robot.fk(q + dq, tool)
        J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / h
        J[3:, i] = so3_log(T1[:3, :3] @ T0[:3, :3].T) / h
    return J


def test_geometric_jacobian_matches_finite_differences(robot, rng):
    for q in random_configs(rng, 25):
        np.testing.assert_allclose(robot.jacobian(q), numerical_jacobian(robot, q), atol=1e-5)


def test_jacobian_with_tool(robot, rng):
    tool = np.eye(4)
    tool[:3, 3] = [0.01, -0.03, 0.2]
    for q in random_configs(rng, 10):
        np.testing.assert_allclose(
            robot.jacobian(q, tool), numerical_jacobian(robot, q, tool), atol=1e-5
        )


def test_manipulability_vanishes_at_wrist_singularity(robot):
    q = np.array([0.2, -1.1, 1.3, -0.7, 0.0, 0.5])  # q5 = 0
    assert robot.manipulability(q) < 1e-10


def test_manipulability_vanishes_at_elbow_singularity(robot):
    q = np.array([0.2, -1.1, 0.0, -0.7, 1.0, 0.5])  # q3 = 0: arm fully stretched
    assert robot.manipulability(q) < 1e-10


def test_manipulability_positive_in_generic_pose(robot):
    q = np.array([0.3, -1.2, 1.4, -0.9, 1.1, 0.4])
    assert robot.manipulability(q) > 1e-4
