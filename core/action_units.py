"""The semantic action units — the single home of the model-facing vocabulary.

These tokens are the entire contract between the VLM and the robot: the model
emits one unit per decision, and an embodiment-specific interpreter grounds it
into motion (see ``interpreters/``). Every module that names a token imports it
from here so the vocabulary cannot drift between the prompt side, the runners,
and the interpreters.

Directional units are defined relative to the current reference view; their
metric realization (unit vectors, step sizes, rotation signs) lives in
``configs/primitives_<embodiment>.yaml``, never here.
"""

from __future__ import annotations

# Incremental end-effector translations.
MOVE_ATOMS = (
    "MV_FWD",
    "MV_BACK",
    "MV_LEFT",
    "MV_RIGHT",
    "MV_UP",
    "MV_DOWN",
)

# Incremental end-effector yaw, as seen in the wrist view (optional; offered by
# the rotation plugin on embodiments that support it).
ROTATE_ATOMS = ("ROTATE_CW", "ROTATE_CCW")

# Incremental end-effector rotation about the WORLD axes through the TCP — the same frame
# the MV_* units use, so a rotation unit means the same thing whatever the hand is doing.
# PITCH/ROLL are named by where the FINGERTIPS of a downward-pointing gripper tilt, reusing
# the translation words (RT_PITCH_FWD tips them toward MV_FWD); YAW by its sense seen from
# above. Distinct from ROTATE_ATOMS, which is wrist-view yaw only. Offered by the LIBERO
# vocabulary (prompts/v5); the metric step lives in configs/primitives_<embodiment>.yaml.
RT_ATOMS = (
    "RT_ROLL_LEFT",
    "RT_ROLL_RIGHT",
    "RT_PITCH_FWD",
    "RT_PITCH_BACK",
    "RT_YAW_CW",
    "RT_YAW_CCW",
)

# Sim-controller idle token (holds the current setpoint for one env step).
STOP_ATOM = "STOP"

ATOMIC_ACTIONS = MOVE_ATOMS + ROTATE_ATOMS + (STOP_ATOM,)

# Gripper and episode-termination units.
GRASP_ATOM = "GRASP"
RELEASE_ATOM = "RELEASE"
DONE_ATOM = "DONE"
GRIPPER_ATOMS = (GRASP_ATOM, RELEASE_ATOM)

# Dual-arm: hold one arm stationary while the other acts. In-vocabulary for
# dual-arm policies; single-arm loops use it only as the human-override hold.
STILL_ATOM = "STILL"
