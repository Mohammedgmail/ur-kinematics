"""Sample the reachable workspace and colour it by manipulability.

Run:  python examples/plot_workspace.py   (saves docs/workspace.png)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from urkin import UR3e

robot = UR3e()
rng = np.random.default_rng(0)
N = 6000

Q = rng.uniform(-np.pi, np.pi, size=(N, 6))
P = np.array([robot.fk(q)[:3, 3] for q in Q])
w = np.array([robot.manipulability(q) for q in Q])

fig, axes = plt.subplots(
    1, 2, figsize=(10.5, 5.2), layout="constrained", gridspec_kw={"width_ratios": [1.1, 0.62]}
)
order = np.argsort(w)  # draw high-manipulability points on top
r = np.hypot(P[:, 0], P[:, 1])

sc = axes[0].scatter(P[order, 0], P[order, 1], c=w[order], s=3, cmap="viridis")
axes[0].set(title="Top view (x-y)", xlabel="x [m]", ylabel="y [m]", aspect="equal")

axes[1].scatter(r[order], P[order, 2], c=w[order], s=3, cmap="viridis")
axes[1].set(title="Side view (radial distance vs height)", xlabel="r [m]", ylabel="z [m]")
axes[1].set_aspect("equal")

fig.colorbar(sc, ax=axes, label="manipulability  sqrt(det(J J^T))", shrink=0.85)
fig.suptitle(f"UR3e flange workspace - {N} random joint configurations", fontsize=13)
out = Path(__file__).resolve().parents[1] / "docs" / "workspace.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
