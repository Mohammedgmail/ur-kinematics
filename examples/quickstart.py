"""Quick tour of urkin: FK, all 8 IK solutions, closest IK, Jacobian.

Run:  python examples/quickstart.py
"""

import numpy as np

from urkin import UR3e, pose_to_rotvec

np.set_printoptions(precision=4, suppress=True)

robot = UR3e()
q_now = np.array([0.3, -1.2, 1.4, -0.9, 1.1, 0.4])  # radians

# Forward kinematics: joint angles -> flange pose (4x4)
T = robot.fk(q_now)
print("Flange position [m]:", T[:3, 3])
print("Pose as [x, y, z, rx, ry, rz]:", pose_to_rotvec(T))

# Analytical inverse kinematics: pose -> every joint solution (up to 8)
solutions = robot.ik(T)
print(f"\n{len(solutions)} IK solutions:")
for s in solutions:
    print("  ", s)

# Pick the solution nearest to the current joint state (no wrist flips)
print("\nClosest to current state:", robot.ik_closest(T, q_ref=q_now))

# Jacobian and manipulability (distance from singularity)
print("\nManipulability:", round(robot.manipulability(q_now), 6))
q_singular = [0.3, -1.2, 1.4, -0.9, 0.01, 0.4]  # q5 close to 0
print("Near wrist singularity:", round(robot.manipulability(q_singular), 6))

# Tool centre point: 12 cm along the flange z-axis
tool = np.eye(4)
tool[2, 3] = 0.12
T_tcp = robot.fk(q_now, tool=tool)
print("\nTCP position [m]:", T_tcp[:3, 3])
print("Numerical IK with tool:", robot.ik_numerical(T_tcp, np.zeros(6) + 0.1, tool=tool))
