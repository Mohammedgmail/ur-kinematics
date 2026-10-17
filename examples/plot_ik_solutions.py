"""Draw the 8 joint configurations that reach the same flange pose.

Run:  python examples/plot_ik_solutions.py   (saves docs/ik_solutions.png)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from urkin import UR3e
from urkin.plotting import draw_frame, draw_robot, set_equal_3d

robot = UR3e()
q = np.array([0.3, -1.2, 1.4, -0.9, 1.1, 0.4])
T = robot.fk(q)
solutions = robot.ik(T)

t1_branches = np.unique(np.round(solutions[:, 0], 9))
center = 0.5 * T[:3, 3] + np.array([0.0, 0.0, 0.08])

fig = plt.figure(figsize=(14, 7.6))
for k, s in enumerate(solutions):
    ax = fig.add_subplot(2, 4, k + 1, projection="3d")
    original = np.allclose(s, q, atol=1e-9)
    draw_robot(ax, robot, s, color="#c0392b" if original else "#1f4e79", show_flange_frame=False)
    draw_frame(ax, T, length=0.08, lw=2.5)
    set_equal_3d(ax, center=center, radius=0.3)
    ax.view_init(elev=20, azim=-35)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([])
        axis.pane.set_alpha(0.25)
    # the three binary choices in the closed-form derivation
    shoulder = "A" if np.isclose(s[0], t1_branches[0]) else "B"
    elbow = "+" if s[2] > 0 else "-"
    wrist = "+" if s[4] > 0 else "-"
    label = f"#{k + 1}   shoulder {shoulder} | elbow {elbow} | wrist {wrist}"
    ax.set_title(label + ("   (seed)" if original else ""), fontsize=9.5)

fig.suptitle("UR3e - eight closed-form IK solutions for one flange pose", fontsize=14)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs" / "ik_solutions.png"
fig.savefig(out, dpi=110)
print("saved", out)
