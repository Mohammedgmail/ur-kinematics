"""Small, dependency-free helpers for rigid-body transforms (NumPy only).

Conventions
-----------
* Homogeneous transforms are 4x4 ``numpy.ndarray`` objects.
* Rotation vectors (axis-angle, ``[rx, ry, rz]``) are what Universal Robots
  controllers use for poses, e.g. ``p[x, y, z, rx, ry, rz]`` in URScript.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix such that ``skew(a) @ b == a x b``."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def roty(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_transform(R: np.ndarray | None = None, p=None) -> np.ndarray:
    """Build a 4x4 transform from a rotation matrix and a translation."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inv_transform(T: np.ndarray) -> np.ndarray:
    """Invert a rigid transform without a general matrix inverse."""
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rotation vector -> rotation matrix (Rodrigues' formula)."""
    w = np.asarray(w, dtype=float)
    theta = np.linalg.norm(w)
    if theta < _EPS:
        return np.eye(3) + skew(w)
    K = skew(w / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector, robust near 0 and pi."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-9:
        return 0.5 * vee
    if np.pi - theta < 1e-6:
        # Near pi: R ~ 2 n n^T - I, so (R + I) / 2 ~ n n^T.
        B = 0.5 * (R + np.eye(3))
        k = int(np.argmax(np.diag(B)))
        n = B[:, k] / np.sqrt(max(B[k, k], _EPS))
        return theta * n / np.linalg.norm(n)
    return theta / (2.0 * np.sin(theta)) * vee


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Roll-pitch-yaw (fixed X-Y-Z, i.e. R = Rz(yaw) Ry(pitch) Rx(roll))."""
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


def pose_to_rotvec(T: np.ndarray) -> np.ndarray:
    """4x4 transform -> ``[x, y, z, rx, ry, rz]`` (UR controller style)."""
    return np.concatenate([T[:3, 3], so3_log(T[:3, :3])])


def rotvec_to_pose(p) -> np.ndarray:
    """``[x, y, z, rx, ry, rz]`` -> 4x4 transform."""
    p = np.asarray(p, dtype=float)
    return make_transform(so3_exp(p[3:]), p[:3])


def pose_error(T_target: np.ndarray, T_current: np.ndarray) -> np.ndarray:
    """6-vector error ``[dp, dw]`` expressed in the base frame.

    ``dw`` is the rotation vector of ``R_target @ R_current.T``, which matches
    the angular part of the geometric Jacobian.
    """
    dp = T_target[:3, 3] - T_current[:3, 3]
    dw = so3_log(T_target[:3, :3] @ T_current[:3, :3].T)
    return np.concatenate([dp, dw])


def wrap_to_pi(q: np.ndarray) -> np.ndarray:
    """Wrap angles to the interval [-pi, pi)."""
    return (np.asarray(q) + np.pi) % (2.0 * np.pi) - np.pi
