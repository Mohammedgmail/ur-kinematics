import numpy as np

from urkin.transforms import wrap_to_pi

from .conftest import random_configs


def contains(solutions, q, tol=1e-6):
    return any(np.all(np.abs(wrap_to_pi(s - q)) < tol) for s in solutions)


def test_every_ik_solution_reproduces_the_pose(robot, rng):
    for q in random_configs(rng, 200):
        T = robot.fk(q)
        sols = robot.ik(T)
        assert len(sols) >= 1
        for s in sols:
            np.testing.assert_allclose(robot.fk(s), T, atol=1e-8)


def test_original_configuration_is_recovered(robot, rng):
    for q in random_configs(rng, 200):
        assert contains(robot.ik(robot.fk(q)), q)


def test_generic_pose_has_eight_distinct_solutions(robot):
    q = np.array([0.3, -1.2, 1.4, -0.9, 1.1, 0.4])
    sols = robot.ik(robot.fk(q))
    assert sols.shape == (8, 6)
    for i in range(8):
        for j in range(i + 1, 8):
            assert not np.allclose(wrap_to_pi(sols[i] - sols[j]), 0, atol=1e-6)


def test_unreachable_pose_returns_empty(robot):
    T = np.eye(4)
    T[:3, 3] = [5.0, 5.0, 5.0]  # far outside every e-Series workspace
    assert robot.ik(T).shape == (0, 6)


def test_wrist_singularity_still_reaches_pose(robot):
    q = np.array([0.4, -1.0, 1.3, -0.6, 0.0, 0.7])  # q5 = 0 -> wrist singular
    T = robot.fk(q)
    sols = robot.ik(T)
    assert len(sols) >= 1
    for s in sols:
        np.testing.assert_allclose(robot.fk(s), T, atol=1e-10)


def test_wrist_singularity_full_precision(robot, rng):
    """Exactly at q5 = 0 or pi, every solution must still be exact.

    Regression test: computing q5 with arccos lost ~8 digits here (errors
    ~1e-7, platform-dependent); atan2 keeps machine precision.
    """
    for _ in range(100):
        q = rng.uniform(-np.pi, np.pi, 6)
        q[4] = rng.choice([0.0, np.pi])
        T = robot.fk(q)
        sols = robot.ik(T)
        assert len(sols) >= 1
        for s in sols:
            np.testing.assert_allclose(robot.fk(s), T, atol=1e-10)


def test_ik_with_tool_offset(robot, rng):
    tool = np.eye(4)
    tool[:3, 3] = [0.0, 0.02, 0.12]
    for q in random_configs(rng, 50):
        T_tcp = robot.fk(q, tool=tool)
        sols = robot.ik(T_tcp, tool=tool)
        assert contains(sols, q)


def test_ik_closest_picks_current_configuration(robot, rng):
    for q in random_configs(rng, 50):
        best = robot.ik_closest(robot.fk(q), q_ref=q)
        np.testing.assert_allclose(best, q, atol=1e-6)


def test_ik_closest_avoids_full_turn(robot):
    q_ref = np.array([0.1, -1.3, 1.5, -1.8, -1.5, 6.0])  # wrist 3 near +2*pi
    best = robot.ik_closest(robot.fk(q_ref), q_ref=q_ref)
    np.testing.assert_allclose(best, q_ref, atol=1e-6)


def test_numerical_ik_converges_from_nearby_seed(robot, rng):
    for q in random_configs(rng, 30):
        seed = q + rng.normal(0, 0.15, 6)
        res = robot.ik_numerical(robot.fk(q), seed)
        assert res.success
        np.testing.assert_allclose(robot.fk(res.q), robot.fk(q), atol=1e-6)


def test_numerical_ik_reports_failure_for_unreachable_pose(robot):
    T = np.eye(4)
    T[:3, 3] = [5.0, 0.0, 0.0]
    res = robot.ik_numerical(T, np.zeros(6), max_iter=50)
    assert not res.success
    assert res.position_error > 1.0
