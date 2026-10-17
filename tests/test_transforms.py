import numpy as np

from urkin.transforms import (
    inv_transform,
    make_transform,
    pose_to_rotvec,
    rotvec_to_pose,
    rpy_to_matrix,
    so3_exp,
    so3_log,
    wrap_to_pi,
)


def random_rotation(rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return so3_exp(axis * rng.uniform(0, np.pi))


def test_exp_log_round_trip(rng):
    for _ in range(200):
        R = random_rotation(rng)
        np.testing.assert_allclose(so3_exp(so3_log(R)), R, atol=1e-9)


def test_log_near_pi():
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -2, 3]):
        n = np.asarray(axis, float) / np.linalg.norm(axis)
        R = so3_exp(n * (np.pi - 1e-8))
        np.testing.assert_allclose(so3_exp(so3_log(R)), R, atol=1e-7)


def test_log_of_identity_is_zero():
    np.testing.assert_allclose(so3_log(np.eye(3)), 0.0)


def test_inverse_transform(rng):
    T = make_transform(random_rotation(rng), rng.normal(size=3))
    np.testing.assert_allclose(T @ inv_transform(T), np.eye(4), atol=1e-12)


def test_rotvec_pose_round_trip(rng):
    for _ in range(50):
        T = make_transform(random_rotation(rng), rng.normal(size=3))
        np.testing.assert_allclose(rotvec_to_pose(pose_to_rotvec(T)), T, atol=1e-9)


def test_rpy_yaw_only():
    R = rpy_to_matrix(0, 0, np.pi / 2)
    np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_wrap_to_pi():
    np.testing.assert_allclose(wrap_to_pi([3 * np.pi / 2, -3 * np.pi / 2]), [-np.pi / 2, np.pi / 2])
