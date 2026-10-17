"""Forward kinematics, analytical / numerical inverse kinematics and the
geometric Jacobian for Universal Robots e-Series arms.

The closed-form inverse kinematics follows the well-known derivation for the
UR family (K. P. Hawkins, "Analytic Inverse Kinematics for the Universal
Robots UR-5/UR-10 Arms", Georgia Tech, 2013) and returns up to 8 solutions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robots import UR3E, UR5E, UR10E, DHParams
from .transforms import inv_transform, pose_error, wrap_to_pi

TWO_PI = 2.0 * np.pi


def dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Standard DH link transform: Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@dataclass
class IKResult:
    """Outcome of :meth:`URRobot.ik_numerical`."""

    q: np.ndarray
    success: bool
    iterations: int
    position_error: float
    orientation_error: float


class URRobot:
    """A 6-DOF Universal Robots arm described by standard DH parameters.

    Parameters
    ----------
    dh:
        DH table (see :mod:`urkin.robots`).
    name:
        Human-readable model name.
    joint_limits:
        ``(6, 2)`` array of ``[lower, upper]`` limits in radians.
        Defaults to +/- 2*pi on every joint, as on the e-Series.
    """

    def __init__(self, dh: DHParams, name: str = "UR", joint_limits=None) -> None:
        self.dh = dh
        self.name = name
        self.a = np.array(dh.a, dtype=float)
        self.d = np.array(dh.d, dtype=float)
        self.alpha = np.array(dh.alpha, dtype=float)
        if joint_limits is None:
            joint_limits = np.tile([-TWO_PI, TWO_PI], (6, 1))
        self.joint_limits = np.asarray(joint_limits, dtype=float)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"

    # ------------------------------------------------------------------ FK
    def link_transform(self, i: int, theta: float) -> np.ndarray:
        """Transform from frame ``i`` to frame ``i + 1`` (``i`` is 0-based)."""
        return dh_transform(self.a[i], self.d[i], self.alpha[i], theta)

    def fk_frames(self, q) -> np.ndarray:
        """All frames from base to flange: array of shape ``(7, 4, 4)``."""
        q = self._as_q(q)
        frames = np.empty((7, 4, 4))
        frames[0] = np.eye(4)
        for i in range(6):
            frames[i + 1] = frames[i] @ self.link_transform(i, q[i])
        return frames

    def fk(self, q, tool: np.ndarray | None = None) -> np.ndarray:
        """Pose of the flange (or of ``tool``, given flange->TCP) in the base frame."""
        T = self.fk_frames(q)[-1]
        return T @ tool if tool is not None else T

    # ------------------------------------------------------------ Jacobian
    def jacobian(self, q, tool: np.ndarray | None = None) -> np.ndarray:
        """6x6 geometric Jacobian ``[v; w] = J @ dq`` in the base frame."""
        frames = self.fk_frames(q)
        p_end = (frames[-1] @ tool)[:3, 3] if tool is not None else frames[-1][:3, 3]
        J = np.zeros((6, 6))
        for i in range(6):
            z = frames[i][:3, 2]
            o = frames[i][:3, 3]
            J[:3, i] = np.cross(z, p_end - o)
            J[3:, i] = z
        return J

    def manipulability(self, q) -> float:
        """Yoshikawa manipulability index ``sqrt(det(J J^T))`` (0 at singularities).

        Computed as the product of the Jacobian's singular values, which is
        numerically better behaved than taking the square root of a determinant.
        """
        sigma = np.linalg.svd(self.jacobian(q), compute_uv=False)
        return float(np.prod(sigma))

    # ------------------------------------------------------ analytical IK
    def ik(self, T, tool: np.ndarray | None = None, tol: float = 1e-9) -> np.ndarray:
        """All closed-form IK solutions for a flange (or tool) pose.

        Returns an array of shape ``(n, 6)`` with ``0 <= n <= 8``, angles
        wrapped to ``[-pi, pi)``. An empty array means the pose is unreachable.

        At the wrist singularity (``sin(q5) == 0``) only ``q4 +/- q6`` is
        determined, so there is a one-parameter family of solutions. One member
        per branch is returned: ``q6`` is chosen so that the elbow sits at
        mid-reach, which is always reachable whenever the pose is.
        """
        T = np.asarray(T, dtype=float)
        if tool is not None:
            T = T @ inv_transform(tool)
        a, d = self.a, self.d
        a2, a3 = a[1], a[2]
        d4, d6 = d[3], d[5]

        solutions: list[np.ndarray] = []

        # --- theta1: wrist-centre (frame 5 origin) position
        p05 = T @ np.array([0.0, 0.0, -d6, 1.0])
        r = np.hypot(p05[0], p05[1])
        if r < abs(d4) - tol:
            return np.empty((0, 6))
        phi = np.arccos(np.clip(d4 / max(r, 1e-12), -1.0, 1.0))
        psi = np.arctan2(p05[1], p05[0])
        z6 = T[:3, 2]
        T60 = inv_transform(T)

        for t1 in (psi + phi + np.pi / 2, psi - phi + np.pi / 2):
            s1, c1 = np.sin(t1), np.cos(t1)

            # --- theta5: angle between the flange z-axis and joint 2's axis z1.
            # Using atan2(|z1 x z6|, z1 . z6) instead of arccos(c5) keeps full
            # precision near the wrist singularity (q5 = 0 or pi), where arccos
            # turns a 1e-16 rounding error into a ~1e-8 angle error.
            z1 = np.array([s1, -c1, 0.0])
            c5 = float(z1 @ z6)
            s5_abs = float(np.linalg.norm(np.cross(z1, z6)))
            for t5 in (np.arctan2(s5_abs, c5), np.arctan2(-s5_abs, c5)):
                s5 = np.sin(t5)

                # --- theta6 (free at the wrist singularity)
                if abs(s5) < 1e-10:
                    t6 = self._singular_wrist_q6(T, p05[:3], z1)
                    if t6 is None:
                        continue
                else:
                    t6 = np.arctan2(
                        (-T60[1, 0] * s1 + T60[1, 1] * c1) / s5,
                        (T60[0, 0] * s1 - T60[0, 1] * c1) / s5,
                    )

                # --- planar 2R problem for theta2, theta3
                T01 = self.link_transform(0, t1)
                T45 = self.link_transform(4, t5)
                T56 = self.link_transform(5, t6)
                T14 = inv_transform(T01) @ T @ inv_transform(T45 @ T56)
                p13 = (T14 @ np.array([0.0, -d4, 0.0, 1.0]))[:3]
                n13 = np.linalg.norm(p13)

                c3 = (n13**2 - a2**2 - a3**2) / (2.0 * a2 * a3)
                if abs(c3) > 1.0 + tol:
                    continue
                c3 = np.clip(c3, -1.0, 1.0)
                for t3 in (np.arccos(c3), -np.arccos(c3)):
                    t2 = -np.arctan2(p13[1], -p13[0]) + np.arcsin(
                        np.clip(a3 * np.sin(t3) / n13, -1.0, 1.0)
                    )
                    # --- theta4
                    T12 = self.link_transform(1, t2)
                    T23 = self.link_transform(2, t3)
                    T34 = inv_transform(T12 @ T23) @ T14
                    t4 = np.arctan2(T34[1, 0], T34[0, 0])
                    solutions.append(wrap_to_pi(np.array([t1, t2, t3, t4, t5, t6])))

        if not solutions:
            return np.empty((0, 6))
        sols = np.array(solutions)
        # drop numerically failed branches (e.g. clipped edge cases)
        ok = [self._pose_close(self.fk(s), T, 1e-6) for s in sols]
        return self._unique(sols[ok])

    def ik_closest(self, T, q_ref, tool: np.ndarray | None = None) -> np.ndarray | None:
        """The IK solution nearest to ``q_ref`` (e.g. the current joint state).

        Each joint is shifted by multiples of 2*pi within the joint limits to
        minimise motion, which avoids unnecessary full-turn wrist rotations.
        Returns ``None`` if the pose is unreachable.
        """
        q_ref = self._as_q(q_ref)
        best, best_cost = None, np.inf
        for sol in self.ik(T, tool=tool):
            cand = self._nearest_equivalent(sol, q_ref)
            if cand is None:
                continue
            cost = np.sum((cand - q_ref) ** 2)
            if cost < best_cost:
                best, best_cost = cand, cost
        return best

    # ------------------------------------------------------- numerical IK
    def ik_numerical(
        self,
        T,
        q0,
        tool: np.ndarray | None = None,
        max_iter: int = 200,
        damping: float = 0.05,
        pos_tol: float = 1e-6,
        rot_tol: float = 1e-6,
    ) -> IKResult:
        """Damped least-squares IK from a seed ``q0`` with adaptive damping.

        The damping factor shrinks after every step that reduces the error and
        grows after a rejected step (Levenberg-Marquardt), which keeps the
        solver stable near singularities but fast close to the solution.
        Useful when a tool offset or extra constraints make a closed form
        inconvenient, and as an independent check of :meth:`ik`.
        """
        T = np.asarray(T, dtype=float)
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        q = np.clip(self._as_q(q0), lo, hi)
        lam = damping
        e = pose_error(T, self.fk(q, tool))
        err = np.linalg.norm(e)
        for it in range(1, max_iter + 1):
            J = self.jacobian(q, tool)
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e)
            q_new = np.clip(q + dq, lo, hi)
            e_new = pose_error(T, self.fk(q_new, tool))
            err_new = np.linalg.norm(e_new)
            if err_new < err:  # accept step, trust the linear model more
                q, e, err = q_new, e_new, err_new
                lam = max(lam / 3.0, 1e-9)
            else:  # reject step, damp harder
                lam = min(lam * 4.0, 10.0)
            if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < rot_tol:
                return IKResult(q, True, it, *self._err_norms(e))
        return IKResult(q, False, max_iter, *self._err_norms(e))

    # ------------------------------------------------------------ helpers
    def _singular_wrist_q6(self, T: np.ndarray, p05: np.ndarray, z1: np.ndarray) -> float | None:
        """Pick q6 at the wrist singularity so the elbow is at mid-reach.

        With z6 parallel to z1, joint 5's axis z4 may point anywhere in the plane
        normal to z1. Frame 3's origin then lies on a circle of radius d5 around
        ``v = p05 - d4*z1 - [0, 0, d1]`` (relative to frame 1), and the 2-link
        chain a2, a3 must reach it. We choose the point on that circle whose
        distance from the shoulder is closest to |a2| (the middle of the
        reachable annulus) and convert the resulting z4 direction into q6.
        """
        a2, a3 = abs(self.a[1]), abs(self.a[2])
        d1, d4, d5 = self.d[0], self.d[3], self.d[4]
        lo, hi = abs(a2 - a3), a2 + a3

        v = p05 - d4 * z1 - np.array([0.0, 0.0, d1])
        n = float(np.linalg.norm(v))
        if n > 1e-12:
            v_hat = v / n
        else:  # wrist centre on the shoulder: any in-plane direction will do
            v_hat = np.cross(z1, [0.0, 0.0, 1.0])
            v_hat /= np.linalg.norm(v_hat)

        rho = float(np.clip((lo + hi) / 2.0, abs(n - d5), n + d5))
        if rho < lo - 1e-12 or rho > hi + 1e-12:
            return None
        cos_b = 1.0 if n <= 1e-12 else (n**2 + d5**2 - rho**2) / (2.0 * n * d5)
        beta = np.arccos(np.clip(cos_b, -1.0, 1.0))
        z4 = np.cos(beta) * v_hat + np.sin(beta) * np.cross(z1, v_hat)

        # z4 = -(sin(q6) x6 + cos(q6) y6) at the singularity
        x6, y6 = T[:3, 0], T[:3, 1]
        return float(np.arctan2(-(z4 @ x6), -(z4 @ y6)))

    @staticmethod
    def _as_q(q) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape != (6,):
            raise ValueError(f"expected 6 joint angles, got shape {q.shape}")
        return q

    @staticmethod
    def _err_norms(e: np.ndarray) -> tuple[float, float]:
        return float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))

    @staticmethod
    def _pose_close(A: np.ndarray, B: np.ndarray, tol: float) -> bool:
        return bool(np.allclose(A, B, atol=tol))

    @staticmethod
    def _unique(sols: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        kept: list[np.ndarray] = []
        for s in sols:
            if not any(np.all(np.abs(wrap_to_pi(s - k)) < tol) for k in kept):
                kept.append(s)
        return np.array(kept) if kept else np.empty((0, 6))

    def _nearest_equivalent(self, q: np.ndarray, q_ref: np.ndarray) -> np.ndarray | None:
        out = np.empty(6)
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        for i in range(6):
            k = np.round((q_ref[i] - q[i]) / TWO_PI)
            cands = q[i] + TWO_PI * np.array([k - 1, k, k + 1])
            cands = cands[(cands >= lo[i] - 1e-9) & (cands <= hi[i] + 1e-9)]
            if cands.size == 0:
                return None
            out[i] = cands[np.argmin(np.abs(cands - q_ref[i]))]
        return out


class UR3e(URRobot):
    def __init__(self, joint_limits=None) -> None:
        super().__init__(UR3E, "UR3e", joint_limits)


class UR5e(URRobot):
    def __init__(self, joint_limits=None) -> None:
        super().__init__(UR5E, "UR5e", joint_limits)


class UR10e(URRobot):
    def __init__(self, joint_limits=None) -> None:
        super().__init__(UR10E, "UR10e", joint_limits)


def load_robot(model: str) -> URRobot:
    """Create a robot by name: ``"ur3e"``, ``"ur5e"`` or ``"ur10e"``."""
    classes = {"ur3e": UR3e, "ur5e": UR5e, "ur10e": UR10e}
    try:
        return classes[model.lower()]()
    except KeyError:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(classes)}") from None
