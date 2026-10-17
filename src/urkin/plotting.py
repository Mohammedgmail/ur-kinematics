"""Minimal Matplotlib helpers (install with ``pip install urkin[plot]``)."""

from __future__ import annotations

import numpy as np

from .kinematics import URRobot


def draw_frame(ax, T: np.ndarray, length: float = 0.05, lw: float = 1.5) -> None:
    """Draw RGB axes of a coordinate frame on a 3D Axes."""
    o = T[:3, 3]
    for k, c in enumerate(("tab:red", "tab:green", "tab:blue")):
        e = o + length * T[:3, k]
        ax.plot(*zip(o, e, strict=True), color=c, lw=lw)


def draw_robot(
    ax,
    robot: URRobot,
    q,
    color: str = "#1f4e79",
    lw: float = 4.0,
    show_flange_frame: bool = True,
) -> None:
    """Draw the arm as a stick figure through the DH frame origins."""
    frames = robot.fk_frames(q)
    pts = frames[:, :3, 3]
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", color=color, lw=lw, solid_capstyle="round")
    ax.scatter(pts[1:-1, 0], pts[1:-1, 1], pts[1:-1, 2], s=18, color="white", edgecolors=color)
    ax.scatter(*pts[0], s=60, color="#444444", marker="s")
    if show_flange_frame:
        draw_frame(ax, frames[-1], length=0.06)


def set_equal_3d(ax, center=(0.0, 0.0, 0.2), radius: float = 0.5) -> None:
    """Equal aspect ratio for 3D plots."""
    cx, cy, cz = center
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_zlim(cz - radius, cz + radius)
    ax.set_box_aspect((1, 1, 1))
