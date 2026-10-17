# Changelog

## 0.1.1 (2026-10-17)

### Fixed
- **Analytical IK lost precision at the wrist singularity (q5 = 0 or π).** `q5` was computed
  with `arccos(c5)`. Near `c5 = ±1`, `arccos` turns a last-digit rounding error (~1e-16) into an
  angle error of ~1e-8, which produced pose errors up to ~1e-7. Whether it showed up depended on
  the CPU's floating-point rounding, so CI failed on some GitHub runners and passed on others.
  `q5` is now `atan2(|z1 × z6|, z1 · z6)`, which keeps machine precision everywhere.
- **Some exactly-singular poses returned no solutions.** At q5 = 0 or π only `q4 ± q6` is
  determined, and the fixed choice `q6 = 0` could put the elbow out of reach. `q6` is now chosen
  geometrically so the elbow sits at mid-reach, which is reachable whenever the pose is.

### Tests
- New regression test: 100 random poses exactly at the wrist singularity per model, checked to
  1e-10. It fails on the 0.1.0 code on every machine, not just some.

## 0.1.0 (2026-10-17)

- First release: forward kinematics, closed-form IK (up to 8 solutions), closest-solution IK,
  damped least-squares numerical IK, geometric Jacobian and manipulability for UR3e, UR5e and UR10e.
