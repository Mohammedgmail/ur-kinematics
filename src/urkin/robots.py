"""Denavit-Hartenberg parameters for Universal Robots e-Series arms.

Values are the standard DH parameters published by Universal Robots
("DH parameters for calculations of kinematics and dynamics"), in metres
and radians. Note: UR's ``a`` corresponds to Wikipedia's ``r``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HALF_PI = np.pi / 2


@dataclass(frozen=True)
class DHParams:
    """Standard DH table for a 6-joint UR arm."""

    a: tuple[float, ...]
    d: tuple[float, ...]
    alpha: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("a", "d", "alpha"):
            if len(getattr(self, name)) != 6:
                raise ValueError(f"DH '{name}' must have 6 entries")


_ALPHA = (HALF_PI, 0.0, 0.0, HALF_PI, -HALF_PI, 0.0)

UR3E = DHParams(
    a=(0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0),
    d=(0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921),
    alpha=_ALPHA,
)

UR5E = DHParams(
    a=(0.0, -0.425, -0.3922, 0.0, 0.0, 0.0),
    d=(0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996),
    alpha=_ALPHA,
)

UR10E = DHParams(
    a=(0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0),
    d=(0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655),
    alpha=_ALPHA,
)

MODELS: dict[str, DHParams] = {"ur3e": UR3E, "ur5e": UR5E, "ur10e": UR10E}
