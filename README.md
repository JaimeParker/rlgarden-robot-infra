# rlgarden-robot-infra

Robot hardware, communication, and control layer split out of
[rl-garden](https://github.com/JaimeParker/rl-garden): impedance and pose
controllers (`robot_infra/controller/`), the Franka Flask bridge and gripper
server, and teleop devices (SpaceMouse, twist-based Pico teleop).

`robot_infra.controller.core`/`adapters` are pure torch, no simulator or
robot binding required. `robot_infra.controller.simulator.maniskill` wraps
them for ManiSkill (extras: `sim`). `robot_infra.controller.real` and
`robot_infra.teleop.spacemouse` talk to real hardware (extras: `real`); ROS
itself is not pip-installable and must be set up separately — see
`robot_infra/controller/real/franka_bridge.py`'s module docstring.

This repo has no dependency on `rl_garden` — it is imported by rl-garden's
real-world RL layer ([rlgarden-real-world](https://github.com/JaimeParker/rlgarden-real-world)),
not the other way around.

See `robot_infra/controller/README.md` for the controller layering
(`simulator wrappers -> adapters -> core`).
