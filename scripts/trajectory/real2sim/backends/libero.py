"""LIBERO / LIBERO-plus backend for the atomic-action discretiser.

Run with the LIBERO-plus interpreter (``<LIBERO-plus>/.venv/bin/python``); its site-packages
``.pth`` puts the checkout on ``sys.path`` and sets ``LIBERO_CONFIG_PATH`` -- without that,
LIBERO silently reads ``~/.libero/config.yaml`` and loads the ORIGINAL LIBERO's assets.

What differs from the other backends, all absorbed here:

* **OSC_POSE, 7-dim ``[dx, dy, dz, drx, dry, drz, grip]``**, normalised by the controller's
  ``output_max`` (0.05 m, 0.5 rad per control step at 20 Hz). A zero rotation delta does
  NOT hold the orientation -- the controller re-targets from wherever the hand is -- so
  every translation step carries an active correction back to :meth:`orientation_ref`
  (measured without it: a +x run pitches the hand ~2 deg per 4 cm).
* **Rotation tokens are supported** (``supports_rotation``): :meth:`apply_delta_pose` takes a
  world-frame rotation error, which is exactly the OSC's delta convention.
* **Gripper polarity is the opposite of the core's**: LIBERO +1 closes, -1 opens.
* **Scenes are restored from recorded MuJoCo states**, not from seeds: the regenerated
  LIBERO demos store the full flattened sim state per step, so :meth:`reset_to_state`
  reproduces a demo's scene exactly. :meth:`pose_from_state` reads the TCP and object
  poses of any recorded state WITHOUT rendering, which is how a demo track is built.
* **Every frame's sim state is logged** (:attr:`frame_states`), so the same token sequence
  can be re-rendered under LIBERO-plus's visual perturbations without re-executing it.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from scripts.trajectory.real2sim.atomic_tokenizer import (
    AtomicSimEnv,
    prepared_pair,
    rotation_error,
)


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class LiberoBackend(AtomicSimEnv):
    supports_rotation = True

    def __init__(
        self,
        env: Any,
        agentview_camera: str = "agentview",
        wrist_camera: str = "robot0_eye_in_hand",
        agentview_rotation_degrees: int = 180,
        wrist_rotation_degrees: int = 0,
        agentview_flip: str = "none",
        wrist_flip: str = "none",
        agentview_crop_aspect: Optional[float] = None,
        wrist_crop_aspect: Optional[float] = None,
        agentview_square_size: Optional[int] = None,
        wrist_square_size: Optional[int] = None,
        settle_steps: int = 5,
    ) -> None:
        self.env = env
        self.agentview_camera = str(agentview_camera)
        self.wrist_camera = str(wrist_camera)
        self.agentview_rotation_degrees = int(agentview_rotation_degrees)
        self.wrist_rotation_degrees = int(wrist_rotation_degrees)
        self.agentview_flip = str(agentview_flip)
        self.wrist_flip = str(wrist_flip)
        self.agentview_crop_aspect = agentview_crop_aspect
        self.wrist_crop_aspect = wrist_crop_aspect
        self.agentview_square_size = int(agentview_square_size) if agentview_square_size else None
        self.wrist_square_size = int(wrist_square_size) if wrist_square_size else None
        self.settle_steps = int(settle_steps)
        ctrl = env.env.robots[0].controller
        self.delta_bound_m = float(ctrl.output_max[0])
        self.delta_bound_rad = float(ctrl.output_max[3])
        self._robot = env.env.robots[0]
        self._obs: Optional[dict] = None
        self._rot_ref: Optional[np.ndarray] = None
        self.frame_states: list[np.ndarray] = []

    @classmethod
    def make(cls, bddl_file: str, camera_size: int = 256, **kwargs: Any) -> "LiberoBackend":
        from libero.libero.envs import OffScreenRenderEnv

        env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=camera_size,
                                 camera_widths=camera_size)
        env.seed(0)
        env.reset()
        return cls(env, **kwargs)

    # -- scene ---------------------------------------------------------------
    def reset(self, seed: int) -> None:
        raise NotImplementedError("LIBERO scenes come from recorded states: reset_to_state()")

    def reset_to_state(self, state: np.ndarray) -> None:
        # robosuite ends an episode at its horizon and then RAISES on the next step
        # ("executing action in terminated episode"). One token is several control steps,
        # so a long rollout crosses the default horizon; success is judged by the task
        # predicate (:meth:`success`), never by the step count.
        self.env.env.ignore_done = True
        self.env.reset()
        self._obs = self.env.set_init_state(np.asarray(state, dtype=np.float64))
        self._rot_ref = self.tcp_rotmat()
        # A few held steps so contacts settle under the controller before the first frame.
        for _ in range(self.settle_steps):
            self.apply_delta(np.zeros(3), 1.0, 0.0)
        self.frame_states = []

    def sim_state(self) -> np.ndarray:
        return np.asarray(self.env.get_sim_state(), dtype=np.float64).copy()

    def pose_from_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict, float, dict]:
        """(tcp_pos, tcp_rotmat, {object: pos}, gripper_width, {object: rotmat}) of a recorded
        state -- no rendering.

        Leaves the sim at ``state``; call :meth:`reset_to_state` before executing anything.
        """
        sim = self.env.sim
        sim.set_state_from_flattened(np.asarray(state, dtype=np.float64))
        sim.forward()
        pos = np.array(sim.data.site_xpos[self._robot.eef_site_id], dtype=np.float64)
        q = sim.data.get_body_xquat(self._robot.robot_model.eef_name)  # wxyz
        rot = quat_xyzw_to_mat(np.array([q[1], q[2], q[3], q[0]]))
        objs = {name: np.array(sim.data.body_xpos[bid], dtype=np.float64)
                for name, bid in self.env.env.obj_body_id.items()}
        gq = sim.data.qpos[self._robot._ref_gripper_joint_pos_indexes]
        obj_rots = {name: np.array(sim.data.body_xmat[bid], dtype=np.float64).reshape(3, 3)
                    for name, bid in self.env.env.obj_body_id.items()}
        return pos, rot, objs, float(gq[0] - gq[1]), obj_rots

    def articulations(self) -> dict:
        """{joint: {"type": "slide"|"hinge", "axis": world unit axis, "range", "body", "body_id"}} for
        every articulated joint of the scene's fixtures/objects (drawers, knobs, doors) --
        robot and gripper joints excluded. Axes are read from the CURRENT pose of each body."""
        import mujoco

        m, d = self.env.sim.model._model, self.env.sim.data._data
        out = {}
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if m.jnt_type[j] not in (2, 3) or name.startswith(("robot0", "gripper0")):
                continue
            body = int(m.jnt_bodyid[j])
            axis = d.xmat[body].reshape(3, 3) @ m.jnt_axis[j]
            rng = [float(v) for v in m.jnt_range[j]]
            anchor = np.asarray(d.xpos[body]) + d.xmat[body].reshape(3, 3) @ m.jnt_pos[j]
            out[name] = {"type": "slide" if m.jnt_type[j] == 2 else "hinge",
                         "axis": axis / np.linalg.norm(axis), "adr": int(m.jnt_qposadr[j]),
                         "range": rng, "anchor": np.array(anchor, dtype=np.float64),
                         "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body),
                         "body_id": body}
        return out

    def joint_positions(self) -> dict:
        """{joint: q} for :meth:`articulations`, at the CURRENT sim state."""
        q = self.env.sim.data.qpos
        return {n: float(q[a["adr"]]) for n, a in self._articulations().items()}

    def body_pos(self, body_id: int) -> np.ndarray:
        return np.array(self.env.sim.data.body_xpos[body_id], dtype=np.float64)

    def _articulations(self) -> dict:
        if getattr(self, "_art_cache", None) is None:
            self._art_cache = self.articulations()
        return self._art_cache

    def extent(self, which: str, axis: np.ndarray) -> tuple[float, float]:
        """(min, max) over the world AABBs of ``which``'s collision geoms, projected on
        ``axis``, at the CURRENT state. ``which``: "gripper" (hand + fingers) or an object /
        fixture name (its whole body subtree)."""
        m, d = self.env.sim.model, self.env.sim.data
        if which == "gripper":
            keep = lambda g: (m.body_id2name(int(m.geom_bodyid[g])) or "").startswith("gripper0_")
        else:
            sub, grew = {int(self.env.env.obj_body_id[which])}, True
            while grew:
                grew = False
                for b in range(m.nbody):
                    if b not in sub and int(m.body_parentid[b]) in sub:
                        sub.add(b)
                        grew = True
            keep = lambda g: int(m.geom_bodyid[g]) in sub
        a = np.asarray(axis, dtype=np.float64)
        lo, hi = np.inf, -np.inf
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]) or not keep(g):
                continue
            rot = np.asarray(d.geom_xmat[g]).reshape(3, 3)
            ctr = float((np.asarray(d.geom_xpos[g]) + rot @ np.asarray(m.geom_aabb[g][:3])) @ a)
            r = float((np.abs(rot) @ np.asarray(m.geom_aabb[g][3:])) @ np.abs(a))
            lo, hi = min(lo, ctr - r), max(hi, ctr + r)
        return lo, hi

    def floor_of(self, body_id: int) -> dict:
        """The floor of a container body (a drawer): of its OWN collision geoms, the widest
        flat one. World AABB centre and half extents, CURRENT state."""
        m, d = self.env.sim.model, self.env.sim.data
        best = None
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != int(body_id) or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            rot = np.asarray(d.geom_xmat[g]).reshape(3, 3)
            ctr = np.asarray(d.geom_xpos[g]) + rot @ np.asarray(m.geom_aabb[g][:3])
            half = np.abs(rot) @ np.asarray(m.geom_aabb[g][3:])
            if half[2] > 0.25 * min(half[0], half[1]):
                continue  # not flat
            if best is None or half[0] * half[1] > best[2][0] * best[2][1]:
                best = (g, ctr, half)
        g, ctr, half = best
        return {"geom": m.geom_id2name(g), "center": np.array(ctr, dtype=np.float64),
                "half": np.array(half, dtype=np.float64)}

    def lever_of(self, body_id: int, axis: np.ndarray, anchor: np.ndarray) -> dict:
        """Where to push a hinged part: of the collision geoms in ``body_id``'s subtree, the
        one whose centre is FURTHEST from the hinge line (a door's far edge, a knob's
        protruding grip) -- the longest lever and, for a door, the fastest travel per token.
        World centre, half extents and that distance, CURRENT state."""
        m, d = self.env.sim.model, self.env.sim.data
        sub_, grew = {int(body_id)}, True
        while grew:
            grew = False
            for b in range(m.nbody):
                if b not in sub_ and int(m.body_parentid[b]) in sub_:
                    sub_.add(b)
                    grew = True
        a = np.asarray(axis, dtype=np.float64)
        a = a / np.linalg.norm(a)
        anchor = np.asarray(anchor, dtype=np.float64)
        best = None
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) not in sub_ or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            rot = np.asarray(d.geom_xmat[g]).reshape(3, 3)
            ctr = np.asarray(d.geom_xpos[g]) + rot @ np.asarray(m.geom_aabb[g][:3])
            half = np.abs(rot) @ np.asarray(m.geom_aabb[g][3:])
            r = ctr - anchor
            dist = float(np.linalg.norm(r - (r @ a) * a))
            if best is None or dist > best[0]:
                best = (dist, g, ctr, half)
        dist, g, ctr, half = best
        return {"geom": m.geom_id2name(g), "center": np.array(ctr, dtype=np.float64),
                "half": np.array(half, dtype=np.float64), "radius": dist}

    def body_top(self, body_id: int) -> float:
        """Highest point of ``body_id``'s subtree collision geoms (a drawer's walls), NOW."""
        return float(self.tallest_geom(body_id)["center"][2] + self.tallest_geom(body_id)["half"][2])

    def tallest_geom(self, body_id: int) -> dict:
        """The collision geom of ``body_id``'s subtree reaching highest (a knob's grip stands
        on its dial). World AABB centre and half extents, CURRENT state."""
        m, d = self.env.sim.model, self.env.sim.data
        sub, grew = {int(body_id)}, True
        while grew:
            grew = False
            for b in range(m.nbody):
                if b not in sub and int(m.body_parentid[b]) in sub:
                    sub.add(b)
                    grew = True
        best = None
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) not in sub or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            rot = np.asarray(d.geom_xmat[g]).reshape(3, 3)
            ctr = np.asarray(d.geom_xpos[g]) + rot @ np.asarray(m.geom_aabb[g][:3])
            half = np.abs(rot) @ np.asarray(m.geom_aabb[g][3:])
            if best is None or ctr[2] + half[2] > best[1][2] + best[2][2]:
                best = (m.geom_id2name(g), ctr, half)
        return {"geom": best[0], "center": np.array(best[1], dtype=np.float64),
                "half": np.array(best[2], dtype=np.float64)}

    def handle_of(self, body_id: int, pull_dir: np.ndarray) -> dict:
        """The handle of an articulated part: of the collision geoms in ``body_id``'s subtree,
        the one reaching furthest along ``pull_dir`` (the drawer's bar stands 1.6 cm proud
        of its front panel). World AABB centre, half extents and long axis, CURRENT state."""
        m, d = self.env.sim.model, self.env.sim.data
        sub, grew = {int(body_id)}, True
        while grew:
            grew = False
            for b in range(m.nbody):
                if b not in sub and int(m.body_parentid[b]) in sub:
                    sub.add(b)
                    grew = True
        u = np.asarray(pull_dir, dtype=np.float64)
        u = u / np.linalg.norm(u)
        best = None
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) not in sub or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            rot = np.asarray(d.geom_xmat[g]).reshape(3, 3)
            ctr = np.asarray(d.geom_xpos[g]) + rot @ np.asarray(m.geom_aabb[g][:3])
            half = np.abs(rot) @ np.asarray(m.geom_aabb[g][3:])
            reach = float(ctr @ u + half @ np.abs(u))
            if best is None or reach > best[0]:
                best = (reach, g, ctr, half)
        _, g, ctr, half = best
        return {"geom": m.geom_id2name(g), "center": np.array(ctr, dtype=np.float64),
                "half": np.array(half, dtype=np.float64), "long_axis": int(np.argmax(half))}

    def fixture_poses(self) -> dict:
        """{fixture: [x, y, z, qw, qx, qy, qz]} as placed in the MODEL for this episode.

        Fixtures (cabinet, stove, ...) have no joints, so they are not in the state vector:
        LIBERO samples their placement into ``model.body_pos`` on every ``reset()``, a few
        mm apart each time. A state restored into another env instance therefore shows the
        fixtures shifted -- measured: 2.7% of agentview pixels off by > 20/255, all on the
        cabinet and stove edges. Store these with a rollout and :meth:`set_fixture_poses`
        them before re-rendering.
        """
        m = self.env.sim.model
        out = {}
        for name in self.env.env.fixtures_dict:
            bid = self.env.env.obj_body_id[name]
            out[name] = [float(v) for v in list(m.body_pos[bid]) + list(m.body_quat[bid])]
        return out

    def set_fixture_poses(self, poses: dict) -> None:
        m = self.env.sim.model
        for name, p in (poses or {}).items():
            bid = self.env.env.obj_body_id[name]
            m.body_pos[bid] = p[:3]
            m.body_quat[bid] = p[3:]
        self.env.sim.forward()

    def movable_objects(self) -> list[str]:
        return list(self.env.env.objects_dict.keys())

    def object_pos(self, name: str) -> np.ndarray:
        bid = self.env.env.obj_body_id[name]
        return np.array(self.env.sim.data.body_xpos[bid], dtype=np.float64)

    def object_rotmat(self, name: str) -> np.ndarray:
        bid = self.env.env.obj_body_id[name]
        return np.array(self.env.sim.data.body_xmat[bid], dtype=np.float64).reshape(3, 3)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # noqa: BLE001 -- EGL teardown noise must not fail a run
            pass

    # -- state readback ------------------------------------------------------
    def tcp_pos(self) -> np.ndarray:
        return np.asarray(self._obs["robot0_eef_pos"], dtype=np.float64).copy()

    def tcp_rotmat(self) -> np.ndarray:
        return quat_xyzw_to_mat(self._obs["robot0_eef_quat"])

    def tcp_pose7(self) -> list[float]:
        p = self._obs["robot0_eef_pos"]
        x, y, z, w = [float(v) for v in self._obs["robot0_eef_quat"]]
        return [round(float(v), 5) for v in (p[0], p[1], p[2], w, x, y, z)]

    def gripper_width(self) -> float:
        q = self._obs["robot0_gripper_qpos"]
        return float(q[0] - q[1])

    def orientation_ref(self) -> np.ndarray:
        return self._rot_ref if self._rot_ref is not None else self.tcp_rotmat()

    def set_orientation_ref(self, rotmat: np.ndarray) -> None:
        self._rot_ref = np.asarray(rotmat, dtype=np.float64).copy()

    # -- actuation -----------------------------------------------------------
    def apply_delta(self, delta_m: np.ndarray, grip_cmd: float, max_cmd_m: float) -> None:
        hold = rotation_error(self.orientation_ref(), self.tcp_rotmat())
        self.apply_delta_pose(delta_m, hold, grip_cmd, max_cmd_m, self.delta_bound_rad)

    def apply_delta_pose(self, delta_m: np.ndarray, delta_rotvec: np.ndarray,
                         grip_cmd: float, max_cmd_m: float, max_cmd_rad: float) -> None:
        d = np.clip(np.asarray(delta_m, dtype=np.float64), -max_cmd_m, max_cmd_m) \
            if max_cmd_m > 0 else np.zeros(3)
        r = np.asarray(delta_rotvec, dtype=np.float64)
        n = float(np.linalg.norm(r))
        if n > max_cmd_rad > 0:
            r = r * (max_cmd_rad / n)
        grip = -1.0 if float(grip_cmd) > 0 else 1.0  # core +1 open -> LIBERO -1 open
        action = np.concatenate([d / self.delta_bound_m, r / self.delta_bound_rad, [grip]])
        self._obs, _r, _done, _info = self.env.step(np.clip(action, -1.0, 1.0))

    # -- observation / evaluation -------------------------------------------
    def grab_frames(self) -> tuple[np.ndarray, np.ndarray]:
        if self._obs is None:
            raise RuntimeError("grab_frames() before reset_to_state()")
        # Log the state BEHIND this frame: the rollout's frames can then be re-rendered in
        # any visually-perturbed variant of the scene (same model, same state vector).
        self.frame_states.append(self.sim_state())
        return prepared_pair(self, self._obs["agentview_image"],
                             self._obs["robot0_eye_in_hand_image"])

    def success(self) -> bool:
        return bool(self.env.check_success())
