import numpy as np
import pytest

from urkin import UR3e, dh_transform, load_robot


def test_zero_configuration_matches_closed_form(robot):
    """At q = 0 the flange sits at (a2 + a3, -(d4 + d6), d1 - d5)."""
    a, d = robot.a, robot.d
    expected = np.array([a[1] + a[2], -(d[3] + d[5]), d[0] - d[4]])
    np.testing.assert_allclose(robot.fk(np.zeros(6))[:3, 3], expected, atol=1e-12)


def test_ur3e_zero_configuration_values():
    p = UR3e().fk(np.zeros(6))[:3, 3]
    np.testing.assert_allclose(p, [-0.45675, -0.22315, 0.0665], atol=1e-12)


def test_fk_returns_valid_rigid_transform(robot, rng):
    for _ in range(50):
        T = robot.fk(rng.uniform(-np.pi, np.pi, 6))
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
        np.testing.assert_array_equal(T[3], [0, 0, 0, 1])


def test_base_rotation_rotates_whole_arm(robot):
    q = np.array([0.0, -1.0, 1.2, -0.5, 0.8, 0.3])
    p0 = robot.fk(q)[:3, 3]
    q_rot = q.copy()
    q_rot[0] += np.pi / 2
    p1 = robot.fk(q_rot)[:3, 3]
    np.testing.assert_allclose(p1, [-p0[1], p0[0], p0[2]], atol=1e-12)


def test_tool_offset_is_applied_in_flange_frame(robot, rng):
    tool = np.eye(4)
    tool[2, 3] = 0.15  # 15 cm along the flange z-axis
    q = rng.uniform(-np.pi, np.pi, 6)
    T_flange = robot.fk(q)
    T_tool = robot.fk(q, tool=tool)
    np.testing.assert_allclose(T_tool[:3, 3], T_flange[:3, 3] + 0.15 * T_flange[:3, 2])


def test_dh_transform_identity():
    np.testing.assert_allclose(dh_transform(0, 0, 0, 0), np.eye(4))


def test_bad_input_shape_raises(robot):
    with pytest.raises(ValueError):
        robot.fk(np.zeros(5))


def test_load_robot():
    assert load_robot("UR5e").name == "UR5e"
    with pytest.raises(ValueError):
        load_robot("ur99")
