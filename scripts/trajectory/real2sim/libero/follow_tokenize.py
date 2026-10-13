#!/usr/bin/env python3
"""Scheme D on LIBERO -- re-execute recorded human demos as 15-token atomic rollouts.

Input is the regenerated LIBERO demos (``<task>_demo.hdf5``, one ``data/demo_k`` per demo,
with the FULL MuJoCo state per step). LIBERO-plus's training set is exactly these demos
re-rendered under ~10 visual perturbations (verified action-for-action), so following them
covers the LIBERO-plus trajectories one-to-one.

Per demo:

1. The demo's TCP pose track and object positions are read straight from its recorded
   states (``LiberoBackend.pose_from_state``, no rendering).
2. The track is split at gripper events; each segment is reduced to 6-D corners -- RDP on
   position plus a keyframe every ``rot_key_deg`` of orientation change.
3. The scene is restored from ``states[0]`` and the corners are chased with single-axis
   tokens (``TokenEpisode.chase_pose``): translations in 2 cm, rotations in 10 deg. Every
   stored frame is a state the discrete controller actually reached.
   A GRASP is placed where the demo's fingers FINISHED closing, not where the close
   command started: a human keeps pushing down and sideways while the fingers travel
   (measured over 10 spatial demos: 0.5-4.9 cm down and 0.7-3.7 cm sideways in the ~13
   closing steps), while the follower stops and then closes. Grasping at the command
   pose closed on air in 6 of 10 pilot episodes.
   And the grasp is defined RELATIVE TO THE OBJECT: the hand's offset from the carried
   object at that moment, re-applied to where the object actually is (see grasp_target). The absolute hand pose
   is not enough -- by the time a human's fingers stop, the object has been dragged ~2 cm
   toward the hand, so copying the demo's hand coordinates lands 1-2 cm off an object that,
   for the follower, never moved (measured: hand-object offset 20 mm off with a 1 mm TCP
   error).
   The lattice is ANCHORED on that grasp: before the first frame, the hand is servoed
   (unrecorded, < 1 cm per axis -- less than LIBERO's own start-pose spread) so the grasp
   point is a whole number of 2 cm steps away. Otherwise the lattice alone leaves up to
   1 cm per axis between the reachable node and the grasp point.
4. Before each RELEASE the carried object is steered to where the DEMO put it (the object
   whose position changed most between the demo's grasp and release) -- the follower's
   grasp offset differs from the demo's, so replaying the demo's hand pose would place the
   object off by that offset (measured on an earlier generator: 3/6 -> 12/12 once retargeted).
5. After the last RELEASE the hand lifts clear, the scene settles, and the episode is kept
   only if LIBERO's own success predicate holds.

``--mode simplified`` keeps what the demo KNOWS and drops what it merely DID. Measured on
40 spatial episodes in ``full`` mode: 52% of rotation tokens cancel an opposite one on the
same axis, 147 of 264 happen while carrying a (round) bowl, and translation runs up to 34%
longer than the key poses require. Wobbles a human never meant, which a per-image policy
cannot explain from the frame -- label noise, extra decisions, and a wrist view tilted away
from the real-robot data. So, for PICK-AND-PLACE segments only (the grasped object moves
>= ``--pick-min-move-m`` before release):

* approach: turn the wrist to the demo's grasp orientation FIRST, in place, then XY at a
  safe height, then straight down onto the object-centric grasp pose -- no demo corners;
* transport: leave the pick site the way the demo did (its displacement up to the moment
  it had moved ``--site-radius-m`` sideways -- not straight up: out of a drawer a human
  backs off first), cross at the higher of the demo's two SITE clearance heights (leaving
  the pick site / arriving over the place site; not its peak height, which on the
  cabinet-top task put the far end of the carry out of reach), turn by the demo's net wrist
  change (per world-axis component >= ``--rot-deadband-deg`` only) over the placement, go
  down to the demo's arrival height, then the usual retarget + descent. Every waypoint is
  aimed by the carried OBJECT, not the hand.

OPEN-HAND CONTACT (a drawer hooked and pulled, a plate pushed -- no GRASP anywhere: all 42
libero_goal "open the middle drawer" demos keep the fingers open) is detected from the demo's
states instead of its gripper events: an articulated joint moving, or an object moved >= 3 cm,
while the gripper is open (an object moved only while another was pushed -- knocked by it --
is not an action). Such a demo runs as a timeline of actions, and the contact actions are
CONSTRUCTED from the scene, not replayed from the human's hand: the demo says only which
part, which way, how far. Replaying the hand did not survive the discrete controller -- the
human hooks a drawer handle by its back-top edge and keeps it there by turning the wrist
while pulling (a fixed-orientation pull rides over after 2-5 cm), drags the top drawer by
friction from on top of its handle, and moves the plate by fingers caught in its centre;
with every fix it reached 7/10 on the middle drawer and 0/5 on the other two (plan doc,
section 10). Built from the scene, with only the 15 tokens:

* a drawer (slide joint): turn so the fingers point at the cabinet, opened across the handle
  bar (:func:`handle_grasp`), come in level from in front of it, GRASP the bar, whole tokens
  along the joint axis to the demo's travel, RELEASE, back off along the pull;
* a push: axis by axis in the demo's order, from OUTSIDE low behind the object where the
  hand's footprint is free, else from INSIDE with the fingers down in the rim and re-engaged
  whenever it stops following (:func:`push_engagement`); corrections for what drifted.

A pick later in such an episode is not the anchored key pose, so its grasp node is rounded
DOWN (deeper); after a drawer is opened the hand comes down clear of the drawer and slides in
over the object, drags the object out from under it before lifting, and places into it at
its floor's centre (:func:`timeline_approach`, :func:`clear_of_drawer`,
:func:`drawer_place_shift`). Every constructed orientation is the one of its two mirror
images nearer the episode's start (``r_home``): consecutive 90 deg turns otherwise added up
to 170 deg and the arm threw the TCP 24 cm.

The GRASP orientation itself is never deadbanded: it is part of what the demo knows. And
every turn happens where the arm is folded, never at full reach: turning 30 deg of pitch at
the far end of an XY run threw the TCP up 20 cm on LIBERO's OSC. All simplified chases stop
when they stop making progress (``TokenEpisode.chase_pose(progress_patience=...)``).

Everything the demo says ABOUT THE TASK is kept -- which object, the grasp point and
approach side, where it is placed, the order of events, the carry height. GRASPED contact
segments (a knob held and turned: the grasped object does not move) are followed in full,
since there the path IS the skill. ``--fallback-full`` re-runs a failed simplified episode
in full mode, so the success predicate decides which details were necessary.

Besides the teleop-layout rollout, each episode stores ``states.npz`` (the sim state behind
every frame) and ``final/`` (one frame AFTER success, for a DONE sample that does not reuse
the last action frame). ``libero/rerender.py`` turns the states into visually-perturbed
copies without re-executing anything.

    <LIBERO-plus>/.venv/bin/python scripts/trajectory/real2sim/libero/follow_tokenize.py \
        --demos <task>_demo.hdf5 --bddl <task>.bddl --out <dir> --demo-ids 0-9 [--mode simplified]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

# Control steps to wait for LIBERO's (static) success predicate after the last token.
SETTLE_STEPS = 60
# Lift after RELEASE: whole tokens, so every stored frame stays on the 2 cm lattice.
RELEASE_RETREAT_M = 0.06
# Below this finger opening a closed gripper holds nothing. Measured on LIBERO's Panda: an
# empty close reads ~1.5 mm after 25 steps, while the thinnest real grasp in the spatial
# demos (a bowl rim) holds at 3.4 mm -- 4 mm, the first guess, would reject real grasps.
EMPTY_GRASP_WIDTH_M = 0.0025


def parse_ids(spec: str, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return [i for i in out if i < n]


def bddl_language(bddl: str) -> str:
    """The instruction as the task file states it -- NOT ``task.language``, which LIBERO-plus
    builds from the file name and so carries suffixes like ``table 1`` / ``noise 3``."""
    m = re.search(r"\(:language\s+(.*?)\)", Path(bddl).read_text(), re.S)
    return " ".join(m.group(1).split()) if m else ""


class _Dropped(RuntimeError):
    """The carried object left the fingers during transport."""


class _Unsupported(RuntimeError):
    """The simplified follower has no action for part of this demo (left to --fallback-full)."""


def cancelled_rotations(tokens: list[str]) -> int:
    """Rotation tokens undone by an opposite one on the same axis (2 per matched pair)."""
    from collections import Counter

    c = Counter(tokens)
    return sum(2 * min(c[a], c[b]) for a, b in (("RT_ROLL_RIGHT", "RT_ROLL_LEFT"),
                                                ("RT_PITCH_BACK", "RT_PITCH_FWD"),
                                                ("RT_YAW_CCW", "RT_YAW_CW")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--bddl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--demo-ids", default="", help="e.g. 0-9 or 0,3,5 (default: all)")
    ap.add_argument("--mode", choices=("full", "simplified"), default="full",
                    help="full: chase every demo corner; simplified: key poses only for "
                         "pick-and-place segments (see the module docstring)")
    ap.add_argument("--fallback-full", action="store_true",
                    help="with --mode simplified: re-run a failed episode in full mode")
    ap.add_argument("--rot-deadband-deg", type=float, default=15.0,
                    help="simplified mode: ignore net per-axis wrist changes smaller than this "
                         "while carrying (the grasp orientation itself is always kept)")
    ap.add_argument("--site-radius-m", type=float, default=0.03,
                    help="simplified mode: sideways distance that ends 'leaving the pick site' "
                         "/ begins 'arriving over the place site' in the demo")
    ap.add_argument("--progress-patience", type=int, default=4,
                    help="simplified mode: tokens without progress before a chase gives up")
    ap.add_argument("--pick-min-move-m", type=float, default=0.02,
                    help="a grasp counts as pick-and-place if the object moves at least this "
                         "much before release; otherwise it is a contact segment (full follow)")
    ap.add_argument("--step-m", type=float, default=0.02)
    ap.add_argument("--rot-step-deg", type=float, default=10.0)
    ap.add_argument("--rot-key-deg", type=float, default=7.5,
                    help="orientation change that makes a track sample a corner")
    ap.add_argument("--rdp-eps", type=float, default=0.008)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--robot-config", default="configs/robot_libero.yaml")
    ap.add_argument("--empty-grasp-mm", type=float, default=EMPTY_GRASP_WIDTH_M * 1000,
                    help="finger opening below which a closed grasp counts as empty (0 disables)")
    ap.add_argument("--gripper-steps", type=int, default=25,
                    help="control steps a GRASP/RELEASE holds its command (LIBERO's fingers "
                         "are slow: an empty close still reads 2.4 mm at 15 steps, 1.2 at 30)")
    ap.add_argument("--approach-clearance", type=float, default=0.05,
                    help="demo corners lower than grasp_z + this and within 2x of it in XY "
                         "are dropped, so the last move into a grasp is a vertical descent; "
                         "simplified mode also uses it as the minimum hover/lift height")
    ap.add_argument("--keep-failures", action="store_true")
    args = ap.parse_args()

    import h5py

    from core.config import camera_contract, load_yaml
    from scripts.trajectory.real2sim.atomic_tokenizer import (
        GRASP, RELEASE, ROT_AXES, AtomicExec, RolloutWriter, TokenBudgetExceeded, TokenEpisode,
        opposite, rdp, rot_token_for_axis, rotation_error, rotvec_to_mat, token_for_axis)
    from scripts.trajectory.real2sim.backends import make_backend

    contract = camera_contract(load_yaml(ROOT / args.robot_config))
    backend = make_backend("libero", bddl_file=args.bddl, **contract)
    task_key = Path(args.bddl).stem
    language = bddl_language(args.bddl)
    rot_step = float(np.radians(args.rot_step_deg))
    deadband = float(np.radians(args.rot_deadband_deg))
    tol_rad = float(np.radians(6.0))
    out_root = Path(args.out) / f"{task_key}_follow"

    def corners(pos: np.ndarray, rot: list) -> list[int]:
        keep = set(rdp(pos, args.rdp_eps))
        last = 0
        for t in range(1, len(pos)):
            if np.linalg.norm(rotation_error(rot[t], rot[last])) >= np.radians(args.rot_key_deg):
                keep.add(t)
                last = t
        return sorted(keep)

    def monotone_runs(q: np.ndarray, min_travel: float) -> list:
        """[(t0, t1)] of the runs where ``q`` moves one way by at least ``min_travel``.

        Sign changes that last less than 10 steps (the wobble of a drawer being let go) do
        not split a run.
        """
        d = np.diff(np.asarray(q, dtype=np.float64))
        sign = np.sign(np.where(np.abs(d) > min_travel / 50, d, 0.0))
        runs, cur, start, last = [], 0.0, 0, 0
        for i, sg in enumerate(sign):
            if sg == 0:
                continue
            if sg != cur or i - last > 10:
                if cur != 0 and abs(q[last + 1] - q[start]) >= min_travel:
                    runs.append((start, last + 1))
                cur, start = sg, i
            last = i
        if cur != 0 and abs(q[last + 1] - q[start]) >= min_travel:
            runs.append((start, last + 1))
        return runs

    def kept_rot(rv: np.ndarray) -> np.ndarray:
        """Net rotation with every world-axis component under the deadband dropped."""
        rv = np.asarray(rv, dtype=np.float64)
        return np.where(np.abs(rv) >= deadband, rv, 0.0)

    def exact_rot_anchor(r_now: np.ndarray, r_goal: np.ndarray) -> np.ndarray:
        """A start orientation within half a token of ``r_now`` from which the rotation
        tokens a pursuit will pick land EXACTLY on ``r_goal``.

        Rounding each world-axis component of the error to whole tokens (the grasp
        anchoring) is only exact for a single axis: tokens about different axes do not
        commute, and a hook reached that way was 3-7 deg off -- every hook at 6.9 deg failed,
        the hand's back catching the handle above. So: replay the pursuit's own choice of
        tokens (TokenEpisode.chase_pose, rotation part: largest error first, axis lock,
        opposite guard) from a candidate start, and move the start to where that token
        product leads to the goal; repeat until the token list stops changing.
        """
        tol = max(tol_rad, 0.55 * rot_step)

        def pursue(r: np.ndarray) -> list[str]:
            toks, lock, prev = [], None, None
            for _ in range(60):
                e = rotation_error(r_goal, r)
                norm = np.where(np.abs(e) >= tol, np.abs(e) / rot_step, 0.0)
                if lock is not None and norm[lock] > 0:
                    ax = lock
                else:
                    ax = int(np.argmax(norm))
                    if norm[ax] <= 0:
                        break
                    lock = ax
                tok = rot_token_for_axis(ax, e[ax])
                if prev is not None and opposite(tok, prev):
                    break
                r = rotvec_to_mat(ROT_AXES[tok] * rot_step) @ r
                toks.append(tok)
                prev = tok
            return toks

        start, seen = r_now, None
        for _ in range(6):
            toks = pursue(start)
            if toks == seen:
                break
            seen = toks
            prod = np.eye(3)
            for tok in toks:
                prod = rotvec_to_mat(ROT_AXES[tok] * rot_step) @ prod
            start = prod.T @ r_goal
        if np.linalg.norm(rotation_error(start, r_now)) > 0.75 * rot_step:
            return None  # no nearby start makes the pursuit exact; keep the rounded anchor
        return start

    def prepare(states: np.ndarray, actions: np.ndarray) -> dict:
        """Everything about one demo that does not depend on how it is followed."""
        art = backend.articulations()
        track, jq, jbody = [], {n: [] for n in art}, {n: [] for n in art}
        for s in states:
            track.append(backend.pose_from_state(s))  # leaves the sim at s
            q = backend.joint_positions()
            for n in art:
                jq[n].append(q[n])
                jbody[n].append(backend.body_pos(art[n]["body_id"]))
        jq = {n: np.array(v) for n, v in jq.items()}
        jbody = {n: np.array(v) for n, v in jbody.items()}
        pos = np.array([t[0] for t in track])
        rot = [t[1] for t in track]
        objs = [t[2] for t in track]
        orots = [t[4] for t in track]
        width = np.array([t[3] for t in track])
        g = actions[:, 6] > 0  # LIBERO: +1 closes
        events, grasp_cmd = [], {}
        for t in range(1, len(g)):
            if g[t] == g[t - 1]:
                continue
            if g[t]:
                # Where the fingers stopped closing (width settled), not where the command
                # started -- see the module docstring. (Idle gestures are dropped below.)
                tc = next((k for k in range(t + 1, len(width) - 1)
                           if abs(width[k + 1] - width[k]) < 5e-4 and width[k] < width[t] - 0.005), t)
                events.append((tc, GRASP))
                grasp_cmd[tc] = t
            else:
                events.append((t, RELEASE))
        # A grasp/release pair that moved NOTHING is not an action: the human missed the
        # bottle (2.6 mm of grip, the bottle shifted 1.3 cm) or touched it again after it was
        # already on the rack (3.1 mm, nothing moved). Reproduced faithfully, those close on
        # air and fail an episode whose task was already done -- 2 of 10 on the wine rack.
        def moved_between(tg: int, tr: int) -> bool:
            objs_moved = max(np.linalg.norm(objs[tr][n] - objs[tg][n]) for n in backend.movable_objects())
            joints_moved = max((abs(jq[n][tr] - jq[n][tg]) for n in art), default=0.0)
            return objs_moved >= 0.01 or joints_moved >= 0.005

        idle = set()
        for (tg, eg), (tr, er) in zip(events, events[1:]):
            if eg == GRASP and er == RELEASE and not moved_between(tg, tr):
                idle |= {tg, tr}
        events = [e for e in events if e[0] not in idle]
        synthetic_release = None
        if events and events[-1][1] == GRASP:
            # The demo ends still holding (the book task): its last frame IS the placement.
            synthetic_release = len(pos) - 1
            events.append((synthetic_release, RELEASE))
        # Carried object per grasp/release pair: what the demo moved most between them.
        # ``pick`` keeps only the pairs where it really moved: pick-and-place, as opposed
        # to holding a handle or a knob (contact segments).
        carried, grasp_obj, pick_grasp, pick_release = {}, {}, {}, {}
        for (tg, eg), (tr, er) in zip(events, events[1:]):
            if eg == GRASP and er == RELEASE:
                name = max(backend.movable_objects(),
                           key=lambda n: np.linalg.norm(objs[tr][n] - objs[tg][n]))
                carried[tr] = grasp_obj[tg] = name
                if np.linalg.norm(objs[tr][name] - objs[tg][name]) >= args.pick_min_move_m:
                    pick_grasp[tg] = name
                    pick_release[tr] = tg
        # Open-hand contacts, from what moved rather than from gripper events.
        contacts = []
        for n, a in art.items():
            q = jq[n]
            thr = 0.01 if a["type"] == "slide" else 0.05
            # One action per MONOTONE run, not one per joint: a demo that opens a drawer,
            # puts a bowl in and closes it again nets out at zero travel.
            for t0, t1 in monotone_runs(q, thr):
                if g[t0:t1 + 1].mean() < 0.5:
                    contacts.append({"kind": "joint", "joint": n, "t0": int(t0), "t1": int(t1),
                                     "dq": float(q[t1] - q[t0])})
        for name in backend.movable_objects():
            tr = np.array([o[name] for o in objs])
            if np.linalg.norm(tr[-1, :2] - tr[0, :2]) < 0.03:
                continue
            t0 = int(np.flatnonzero(np.linalg.norm(tr - tr[0], axis=1) > 0.005)[0])
            late = np.flatnonzero(np.linalg.norm(tr - tr[-1], axis=1) > 0.005)
            t1 = int(late[-1] + 1) if len(late) else t0
            # Net displacement, per axis: an object shoved away and brought back has no
            # axis to push along (and asking for one raised IndexError on 1 of 40 tasks).
            if g[t0:t1 + 1].mean() < 0.2 and max(abs(tr[t1, :2] - tr[t0, :2])) >= 0.015:
                contacts.append({"kind": "push", "object": name, "t0": t0, "t1": t1})
        # An object moved only while another was being pushed was knocked by it (the cream
        # cheese in the plate's way), not pushed by the hand: not an action of its own.
        pushes = [c for c in contacts if c["kind"] == "push"]
        for c in pushes:
            for c2 in pushes:
                if c2 is not c and c2["t1"] - c2["t0"] > c["t1"] - c["t0"] and \
                        min(c["t1"], c2["t1"]) - max(c["t0"], c2["t0"]) >= 0.5 * (c["t1"] - c["t0"]):
                    c["collateral_of"] = c2["object"]
        contacts = [c for c in contacts if "collateral_of" not in c]
        timeline = sorted([dict(c) for c in contacts] +
                          [{"kind": "pick", "t0": tg, "t_g": tg, "t_r": tr}
                           for tr, tg in pick_release.items()], key=lambda a: a["t0"])
        # A grasp that is not pick-and-place (a knob held and turned) has no timeline
        # action yet: such a demo is left to the full follower. Except one on an articulated
        # part while it moves (a drawer handle, the stove knob): that action is built from
        # the part, not from the human's hand, so the grasp is not needed.
        pulls = [(c["t0"] - 15, c["t1"] + 5) for c in contacts if c["kind"] == "joint"]
        other_grasps = [t for t, e in events if e == GRASP and t not in pick_grasp
                        and not any(a_ <= t <= b_ for a_, b_ in pulls)]
        return {"states": states, "pos": pos, "rot": rot, "objs": objs, "orots": orots, "events": events,
                "carried": carried, "grasp_obj": grasp_obj, "pick_grasp": pick_grasp,
                "pick_release": pick_release, "grasp_cmd": grasp_cmd,
                "synthetic_release": synthetic_release,
                "art": art, "jq": jq, "jbody": jbody,
                "contacts": contacts, "timeline": timeline,
                "timeline_supported": not other_grasps,
                "bounds": [0] + [t for t, _ in events] + [len(pos) - 1]}

    def run(i: int, demo: dict, mode: str, out_dir: Path, extra: dict,
            grasp_mode: str = "demo") -> dict:
        t0 = time.time()
        simp = mode == "simplified"
        states, pos, rot, objs = demo["states"], demo["pos"], demo["rot"], demo["objs"]
        events, carried, grasp_obj = demo["events"], demo["carried"], demo["grasp_obj"]
        if not simp and demo["synthetic_release"] is not None:
            # Only the timeline needs a release to aim the placement; following the demo in
            # full, the hand simply holds the object where the demo ended (which satisfies
            # the predicate -- letting go there and retreating took the book out again).
            events = events[:-1]
        pick_grasp, pick_release = demo["pick_grasp"], demo["pick_release"]
        bounds = demo["bounds"] if len(events) == len(demo["events"]) \
            else [0] + [t for t, _ in events] + [len(pos) - 1]
        jbody, timeline = demo["jbody"], demo["timeline"]
        use_timeline = simp and bool(demo["contacts"])

        def constructed_grasp(name: str) -> tuple:
            """Grip the object across its narrow side, on its axis, 60% of the way up -- the
            grasp its shape allows, not the one the human found.

            Some demos pinch a can at or above its rim (a hand 10.2 cm up on a 7.6 cm can,
            2.6-4 mm of grip): a lucky human grasp that the replay, with the object upright
            where the demo had it tilted, closes on air.
            """
            ex = [backend.extent(name, ax) for ax in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                                                      np.array([0, 0, 1.0]))]
            narrow = 0 if (ex[0][1] - ex[0][0]) <= (ex[1][1] - ex[1][0]) else 1
            y = np.zeros(3)
            y[narrow] = 1.0
            z_dn = np.array([0.0, 0.0, -1.0])
            r_g = min((np.column_stack([np.cross(v, z_dn), v, z_dn]) for v in (y, -y)),
                      key=lambda rr: float(np.linalg.norm(rotation_error(rr, r_home))))
            p = np.array([0.5 * (ex[0][0] + ex[0][1]), 0.5 * (ex[1][0] + ex[1][1]),
                          ex[2][0] + 0.6 * (ex[2][1] - ex[2][0])])
            return p, r_g

        def grasp_target(t: int, obj_p: np.ndarray, obj_r: np.ndarray) -> tuple:
            """The demo's grasp at ``t``, re-anchored on where the object is NOW.

            Position: the hand's WORLD offset from the object at ``t`` (fingers closed), added
            to the object's current position. Orientation: the demo hand's world orientation.
            Deliberately not the object FRAME: the follower's object starts exactly where
            the demo's did (same recorded state), and the only difference by ``t`` is what
            the human's closing hand did to it -- a drag (kept, it is the offset we need) and
            a twist (dropped). Expressing the grasp in the twisted object frame put the
            twist into the target yaw: 9-24 deg off on the drawer task, where the outer
            finger then landed on the drawer wall.

            A TILT of a tall object is different: a wine bottle is grasped at its neck
            8-13 cm up, and the closing hand tilted it 5-31 deg in every demo -- the neck
            moved 1-5 cm while the base (the object's position) hardly did, and the world
            offset from the base pointed beside an upright neck (empty grasps). So for a
            grasp >= 6 cm over the object's origin, the tilt part of its rotation by ``t``
            (its swing away from vertical; the twist about vertical is still dropped) is
            taken out of the offset and the hand orientation, where that moves the grasp
            point by >= 1 cm. Not for a bowl's rim (~4 cm up): applied there it changed 18
            of 50 spatial episodes, and lost one net.
            """
            name = grasp_obj[t]
            if grasp_mode == "constructed":
                return constructed_grasp(name)
            d, r_h = pos[t] - objs[t][name], rot[t]
            r_delta = demo["orots"][t][name] @ demo["orots"][0][name].T
            z1 = r_delta[:, 2]
            ang = float(np.arccos(np.clip(z1[2], -1.0, 1.0)))
            if d[2] >= 0.06 and ang > 1e-3:
                axis = np.cross([0.0, 0.0, 1.0], z1)
                swing = rotvec_to_mat(axis / np.linalg.norm(axis) * ang)
                d2 = swing.T @ d
                if np.linalg.norm(d2 - d) >= 0.01:
                    d, r_h = d2, swing.T @ r_h
            return obj_p + d, r_h

        handle_plans: dict = {}

        def opens_out(c: dict) -> bool:
            """This joint action pulls a drawer OUT (it has to be grasped); closing it, or
            turning a hinge, can be pushed instead."""
            a = art[c["joint"]]
            if a["type"] != "slide":
                return False
            far = max(a["range"], key=abs)
            return np.sign(c["dq"]) == np.sign(far)

        def handle_grasp(c: dict) -> dict:
            """Grasp the drawer's handle bar, built from the scene, not from the demo.

            The demo's open-hand hook is not reproducible with a fixed-orientation token pull
            (pilot: 7/10 with retries; the top drawer's demos drag the handle by friction
            from on top of it, 0/5). Closing the fingers on the bar is: fingers pointing at
            the fixture (approach = -pull), opening across the bar (pull x bar axis --
            vertical for a horizontal bar, so one finger passes above it and one below).
            Of the two opening signs, the one nearer the hand's current orientation.
            Computed once per episode, at the scene's CURRENT pose (reset state).
            """
            key = id(c)
            if key not in handle_plans:
                a = art[c["joint"]]
                # The handle sticks out on the side the part opens towards: the joint's range
                # end furthest from zero (models are built closed). The hand always comes in
                # from THERE -- the drive itself may go either way (opening or closing).
                far = max(a["range"], key=abs)
                u_out = a["axis"] * np.sign(far)
                u = a["axis"] * np.sign(c["dq"])
                h = backend.handle_of(a["body_id"], u_out)
                long = np.zeros(3)
                long[h["long_axis"]] = 1.0
                v = np.cross(u_out, long)
                v /= np.linalg.norm(v)
                z = -u_out / np.linalg.norm(u_out)
                r_now = backend.orientation_ref()
                cands = []
                for y in (v, -v):
                    r = np.column_stack([np.cross(y, z), y, z])
                    cands.append((float(np.linalg.norm(rotation_error(r, r_now))), r))
                r_g = min(cands, key=lambda c_: c_[0])[1]
                k = int(np.argmax(np.abs(u_out)))
                out_lat = np.zeros(3)
                out_lat[k] = np.sign(u_out[k])  # lattice axis pointing out of the fixture
                handle_plans[key] = {"pos": h["center"].copy(), "rot": r_g, "u": out_lat,
                                     "geom": h["geom"]}
            return handle_plans[key]

        def push_plan(c: dict) -> dict:
            """Which axes to push along, in the demo's order: an axis comes first when the
            demo got HALF of its displacement along it done first (the plate: +Y, then -X)."""
            key = ("push", id(c))
            if key not in handle_plans:
                name, t0, t1 = c["object"], c["t0"], c["t1"]
                tr = np.array([objs[t][name] for t in range(t0, t1 + 1)])
                d = tr[-1] - tr[0]
                axes = [k for k in (0, 1) if abs(d[k]) >= 0.015] or [int(np.argmax(np.abs(d[:2])))]
                half = {k: int(np.argmax((tr[:, k] - tr[0, k]) * np.sign(d[k]) >= 0.5 * abs(d[k]))) for k in axes}
                handle_plans[key] = {"order": sorted(axes, key=lambda k: half[k]),
                                     "goal": objs[t1][name].copy()}
            return handle_plans[key]

        def footprint_free(name: str, lo: np.ndarray, hi: np.ndarray) -> bool:
            """No other object / fixture's XY box overlaps [lo, hi] (XY), 1 cm margin."""
            for n in backend.env.env.obj_body_id:
                if n == name:
                    continue
                bx = backend.extent(n, np.array([1.0, 0, 0])), backend.extent(n, np.array([0, 1.0, 0]))
                if lo[0] - 0.01 < bx[0][1] and hi[0] + 0.01 > bx[0][0] and \
                        lo[1] - 0.01 < bx[1][1] and hi[1] + 0.01 > bx[1][0]:
                    return False
            return True

        def is_knob(c: dict) -> bool:
            """A hinge small enough to be turned by the wrist: its lever is within reach of a
            grip on its own axis (a stove knob at 5 cm; a microwave door's is 24 cm)."""
            a = art[c["joint"]]
            if a["type"] != "hinge" or abs(float(a["axis"] @ np.array([0.0, 0.0, 1.0]))) < 0.9:
                return False
            return backend.lever_of(a["body_id"], a["axis"], a["anchor"])["radius"] < 0.08

        def knob_grasp(c: dict) -> tuple:
            """Grip the knob ON its own axis, so that a RT_YAW token -- a turn about the world
            vertical THROUGH THE TCP -- is exactly the knob turning. Fingers across the short
            side of its grip (the stove's is a 7 x 2.4 cm wing standing on the dial).
            Pushing the knob round instead barely moved it: a 5 cm lever is too short."""
            a = art[c["joint"]]
            grip = backend.tallest_geom(a["body_id"])
            long_axis = int(np.argmax(grip["half"][:2]))
            y = np.zeros(3)
            y[1 - long_axis] = 1.0
            z_dn = np.array([0.0, 0.0, -1.0])
            r_k = min((np.column_stack([np.cross(v, z_dn), v, z_dn]) for v in (y, -y)),
                      key=lambda rr: float(np.linalg.norm(rotation_error(rr, r_home))))
            tip = backend.extent("gripper", np.array([0.0, 0.0, 1.0]))[0] - backend.tcp_pos()[2]
            top = grip["center"][2] + grip["half"][2]
            p = np.array([a["anchor"][0], a["anchor"][1], top - 0.01 - tip])
            return p, r_k

        def do_knob_turn(c: dict) -> None:
            """Grip the knob and turn the wrist: RT tokens about the world vertical, closed
            loop on the joint, then RELEASE and up."""
            pat, st = args.progress_patience, args.step_m
            n = c["joint"]
            p, r_k = knob_grasp(c)
            o = ep.exec.lattice_origin
            node = o + st * np.round((p - o) / st)
            node[2] = o[2] + st * np.floor((p[2] - o[2]) / st + 1e-6)
            ep.chase_pose(backend.tcp_pos(), r_k, tol_m=0.012, tol_rad=tol_rad, progress_patience=pat)
            hover = np.r_[node[:2], max(backend.tcp_pos()[2], node[2] + args.approach_clearance)]
            ep.chase_pose(hover, r_k, tol_m=0.008, tol_rad=tol_rad, z_order=True, progress_patience=pat)
            ep.chase_pose(node, r_k, tol_m=0.008, tol_rad=tol_rad, stall_limit=1, progress_patience=pat)
            ep.emit(GRASP)
            grip_mm = round(backend.gripper_width() * 1000, 2)
            grip_widths_mm.append(grip_mm)
            q0 = backend.joint_positions()[n]
            q_goal = q0 + c["dq"]
            sign = float(np.sign(c["dq"]))
            tok = rot_token_for_axis(2, c["dq"] * float(np.sign(art[n]["axis"][2])))
            stalled, used = 0, 0
            for _ in range(int(np.ceil(abs(c["dq"]) / rot_step)) + 4):
                if (q_goal - backend.joint_positions()[n]) * sign < 0.5 * rot_step:
                    break
                before = backend.joint_positions()[n]
                ep.emit(tok)
                used += 1
                if (backend.joint_positions()[n] - before) * sign < 0.2 * rot_step:
                    stalled += 1
                    if stalled >= 2:
                        break
                else:
                    stalled = 0
            ep.emit(RELEASE)
            grip_widths_mm.append(round(backend.gripper_width() * 1000, 2))
            ep.chase_pose(backend.tcp_pos() + np.array([0.0, 0.0, 2 * st]), backend.orientation_ref(),
                          tol_m=0.008, tol_rad=tol_rad, progress_patience=pat)
            contact_results.append({"kind": "joint", "joint": n, "how": "knob_turn", "token": tok,
                                    "grip_mm": grip_mm, "tokens": used, "dq_demo": round(c["dq"], 3),
                                    "dq_done": round(float(backend.joint_positions()[n] - q0), 3)})

        def joint_push_pose(c: dict) -> tuple:
            """(pose, rotation, token) for PUSHING an articulated part the way the demo moved
            it, NOW -- a microwave door shut, a stove knob round, a drawer closed.

            The hand goes just behind the part's contact point, fingers straight down and
            opened across the push so it is thin along it, and pushes. For a hinge the
            contact is the part's longest lever (:meth:`lever_of`) and the direction its
            tangent (axis x radius), both recomputed every token because they turn with the
            part (a microwave door swings 102 deg). For a drawer being closed it is the
            handle, straight in. Pushing beats grasping the handle there: the bottom drawer
            sits so low that reaching its handle put the arm's link6 into the wine rack.
            """
            a = art[c["joint"]]
            if a["type"] == "slide":
                far = max(a["range"], key=abs)
                u_out = a["axis"] * np.sign(far)
                lev = backend.handle_of(a["body_id"], u_out)
                lev = {**lev, "radius": 1.0}
                tan = a["axis"] * np.sign(c["dq"])
            else:
                lev = backend.lever_of(a["body_id"], a["axis"], a["anchor"])
                r = lev["center"] - a["anchor"]
                axis = a["axis"] / np.linalg.norm(a["axis"])
                r = r - (r @ axis) * axis
                tan = np.cross(axis, r)
                tan = tan / np.linalg.norm(tan) * np.sign(c["dq"])
            k = int(np.argmax(np.abs(tan[:2])))
            e = np.zeros(3)
            e[k] = np.sign(tan[k])
            lat = np.array([-e[1], e[0], 0.0])
            z_dn = np.array([0.0, 0.0, -1.0])
            if a["type"] == "slide":
                # Sideways, exactly as the handle would be grasped: a hand held upright is
                # ~10 cm tall, and pushing a low drawer's handle with it put the palm on the
                # handle of the drawer above (which it then dragged in and out). Held level it
                # is ~6 cm tall and fits between two drawers 7 cm apart.
                r_p = handle_grasp(c)["rot"]
            else:
                r_p = min((np.column_stack([np.cross(y, z_dn), y, z_dn]) for y in (lat, -lat)),
                          key=lambda rr: float(np.linalg.norm(rotation_error(rr, r_home))))
            tcp = backend.tcp_pos()
            g_lo, g_hi = backend.extent("gripper", e)
            face = g_hi - float(tcp @ e)  # the leading face, in the push direction
            p = lev["center"] - e * (face + lev["half"] @ np.abs(e) + 0.005)
            if a["type"] == "slide":
                p[2] = lev["center"][2]          # level hand: its centre is the TCP
            else:
                tip = backend.extent("gripper", np.array([0.0, 0.0, 1.0]))[0] - tcp[2]
                p[2] = lev["center"][2] - tip    # fingers down: their tips at the contact
            return p, r_p, token_for_axis(k, float(e[k]))

        def do_joint_push(c: dict) -> None:
            """Push an articulated part to where the demo left it: engage behind its contact
            (:func:`joint_push_pose`) and push one token at a time, re-engaging (up to 6
            times) when it stops moving -- after a few tokens a swinging part is out of reach
            of the hand where it stood."""
            pat, st = args.progress_patience, args.step_m
            n = c["joint"]
            a = art[n]
            q_goal = backend.joint_positions()[n] + c["dq"]
            sign = float(np.sign(c["dq"]))
            res = {"kind": "joint", "joint": n, "how": "push", "dq_demo": round(c["dq"], 3),
                   "engagements": 0, "tokens": 0}
            if a["type"] == "slide":
                # Closing means SHUT: the demo's own travel left it 3 cm out and "close it"
                # was not satisfied. The end of the joint's range is the target.
                q_goal = min(a["range"], key=abs) if sign > 0 else max(a["range"], key=abs)
                sign = float(np.sign(q_goal - backend.joint_positions()[n]))
                tol = 0.25 * st
            else:
                lev = backend.lever_of(a["body_id"], a["axis"], a["anchor"])
                tol = max(0.5 * st / max(lev["radius"], 0.02), 0.05)
            for _ in range(6):
                if (q_goal - backend.joint_positions()[n]) * sign < tol:
                    break
                res["engagements"] += 1
                p, r_p, tok = joint_push_pose(c)
                o = ep.exec.lattice_origin
                node = o + st * np.round((p - o) / st)
                part = backend.lever_of(a["body_id"], a["axis"], a["anchor"])
                top = part["center"][2] + part["half"][2]
                tip = backend.extent("gripper", np.array([0.0, 0.0, 1.0]))[0] - backend.tcp_pos()[2]
                safe = max(backend.tcp_pos()[2], o[2] + st * np.ceil((top + 0.02 - tip - o[2]) / st))
                ep.chase_pose(np.r_[backend.tcp_pos()[:2], safe], backend.orientation_ref(),
                              tol_m=0.008, tol_rad=tol_rad, progress_patience=pat)
                ep.chase_pose(backend.tcp_pos(), r_p, tol_m=0.012, tol_rad=tol_rad, progress_patience=pat)
                p, r_p, tok = joint_push_pose(c)
                node = o + st * np.round((p - o) / st)
                if a["type"] != "slide":
                    node[2] = o[2] + st * np.floor((p[2] - o[2]) / st + 1e-6)  # not above the contact
                ep.chase_pose(np.r_[node[:2], safe], r_p, tol_m=0.008, tol_rad=tol_rad,
                              z_order=True, progress_patience=pat)
                ep.chase_pose(node, r_p, tol_m=0.008, tol_rad=tol_rad, stall_limit=1,
                              progress_patience=pat)
                stalled, turned, used = 0, 0.0, 0
                for _ in range(24):
                    if (q_goal - backend.joint_positions()[n]) * sign < tol:
                        break
                    before = backend.joint_positions()[n]
                    _, _, tok = joint_push_pose(c)  # the tangent turns with the part
                    ep.emit(tok)
                    res["tokens"] += 1
                    used += 1
                    gain = (backend.joint_positions()[n] - before) * sign
                    turned += gain
                    if gain >= 0.1 * tol:
                        stalled = 0
                    elif turned > 0.2 * tol:
                        stalled += 1            # engaged, and it has stopped moving
                        if stalled >= 2:
                            break
                    elif used >= 7:
                        break                   # the hand crosses its own reach before it
                                                # touches: not a stall (0 of 14.7 cm closed)
                if turned < 0.2 * tol:
                    break
            ep.chase_pose(backend.tcp_pos() + np.array([0.0, 0.0, 2 * st]), backend.orientation_ref(),
                          tol_m=0.008, tol_rad=tol_rad, progress_patience=pat)
            res["dq_done"] = round(float(backend.joint_positions()[n] - (q_goal - c["dq"])), 3)
            contact_results.append(res)

        def push_engagement(c: dict, k: int) -> tuple:
            """(pose, rotation, "outside" | "inside") for the next push along axis ``k``, NOW,
            decided WITHOUT turning the hand (turning to look, then back, wound the wrist
            170 deg and threw the arm off: a yaw of 90 deg only swaps the hand's two
            horizontal extents, so they are read off the current pose).

            Outside -- behind the object, fingers straight down and opened ACROSS the push
            (two contacts side by side), fingertips 5 mm over the table: low and from
            outside a finger meets the underside of a plate's lip, whose reaction presses it
            DOWN rather than up the slope. Only where the hand's footprint there is clear of
            every other object/fixture; otherwise inside (:func:`push_pose`).
            """
            name = c["object"]
            plan = push_plan(c)
            sgn = float(np.sign(plan["goal"][k] - backend.object_pos(name)[k]) or 1.0)
            e = np.zeros(3)
            e[k] = 1.0
            lat = np.array([-e[1], e[0], 0.0])
            z_dn = np.array([0.0, 0.0, -1.0])
            r_now = backend.orientation_ref()
            r_across = min((np.column_stack([np.cross(y, z_dn), y, z_dn]) for y in (lat, -lat)),
                           key=lambda r: float(np.linalg.norm(rotation_error(r, r_home))))
            tcp = backend.tcp_pos()
            along_now = abs(float(r_now[:, 1] @ e)) > 0.7

            def half(ax):
                lo, hi = backend.extent("gripper", ax)
                return max(float(tcp @ ax) - lo, hi - float(tcp @ ax))
            face = half(lat) if along_now else half(e)       # along e once opened across
            width = half(e) if along_now else half(lat)      # across e once opened across
            ex = backend.extent(name, e)
            back = ex[0] if sgn > 0 else ex[1]
            p = backend.object_pos(name).copy()
            p[k] = back - sgn * (face + 0.01)
            lo = p - e * face - lat * width
            hi = p + e * face + lat * width
            if footprint_free(name, np.minimum(lo, hi), np.maximum(lo, hi)):
                tip = backend.extent("gripper", np.array([0.0, 0.0, 1.0]))[0] - tcp[2]
                p[2] = backend.extent(name, np.array([0.0, 0.0, 1.0]))[0] + 0.005 - tip
                return p, r_across, "outside"
            p, r_p = push_pose(c, k)
            return p, r_p, "inside"

        def push_pose(c: dict, k: int) -> tuple:
            """Hand over the object's centre, fingers straight down and opened ALONG the push
            axis ``k``, fingertips half-way up its rim, NOW.

            A plate has no face a token push can hold on to: its rim is a ring of ~27 deg
            slopes (3 to 6.8 cm from the centre), so any pushing finger gets twice as much
            reaction UP as back. From inside, with the fingers (at +-4 cm) each on a rim
            slope, the leading one drives the plate ~1.8 cm per token until it rides up and
            over -- so :func:`do_push` re-engages. From outside there is no room here (the
            hand lands on the cabinet's handles), and pinching the rim fails the same way
            (the slope wedges out of the fingers: 20 -> 1 mm over 10 tokens).
            """
            name = c["object"]
            z = np.array([0.0, 0.0, -1.0])  # fingers straight down (and the frame orthonormal)
            e = np.zeros(3)
            e[k] = 1.0
            cands = []
            for y in (e, -e):
                r = np.column_stack([np.cross(y, z), y, z])
                cands.append((float(np.linalg.norm(rotation_error(r, r_home))), r))
            r_p = min(cands, key=lambda c_: c_[0])[1]
            ctr = backend.object_pos(name)
            top = backend.extent(name, np.array([0.0, 0.0, 1.0]))[1]
            tip = backend.extent("gripper", np.array([0.0, 0.0, 1.0]))[0] - backend.tcp_pos()[2]
            return np.array([ctr[0], ctr[1], top - 0.5 * (top - ctr[2]) - tip]), r_p

        def contact_pose(c: dict) -> Optional[tuple]:
            """The first key pose of a contact action, NOW: a drawer's handle grasp
            (:func:`handle_grasp`), a push's first engagement (:func:`push_engagement`);
            None for a contact with no constructed action (an open-hand turn of a hinge)."""
            if c["kind"] == "joint" and opens_out(c):
                hp = handle_grasp(c)
                return hp["pos"], hp["rot"]
            if c["kind"] == "joint" and is_knob(c):
                return knob_grasp(c)
            if c["kind"] == "joint":
                p, r, _ = joint_push_pose(c)
                return p, r
            if c["kind"] == "push":
                p, r, _ = push_engagement(c, push_plan(c)["order"][0])
                return p, r
            return None

        backend.reset_to_state(states[0])
        # Joint anchors/axes at THIS episode's fixture placement (re-sampled every reset, so
        # the one prepare() saw can be 1 cm off).
        art = backend.articulations()
        # The wrist's neutral: every constructed orientation picks, of its two mirror
        # images, the one nearer THIS -- choosing by the current pose let consecutive
        # 90 deg turns add up to 170 deg, and the arm threw the TCP 24 cm.
        r_home = backend.tcp_rotmat()
        fixture_poses = backend.fixture_poses()
        anchor_cm = anchor_deg = None
        first_grasp = next((t for t, e in events if e == GRASP and t in grasp_obj), None)
        first_key = None
        if use_timeline and demo["timeline_supported"] and timeline:
            first = timeline[0]
            if first["kind"] == "pick":
                name = pick_grasp[first["t_g"]]
                first_key = grasp_target(first["t_g"], backend.object_pos(name), backend.object_rotmat(name))
            else:
                first_key = contact_pose(first)
        elif first_grasp is not None:
            # Anchor BOTH lattices on the object-centric grasp pose (the object is at rest
            # until the hand reaches it, so its initial pose is the one that will count).
            name = grasp_obj[first_grasp]
            first_key = grasp_target(first_grasp, backend.object_pos(name), backend.object_rotmat(name))
        if first_key is not None:
            g_p, g_r = first_key
            here, r_here = backend.tcp_pos(), backend.tcp_rotmat()
            off = g_p - here
            shift = off - args.step_m * np.round(off / args.step_m)
            rv = rotation_error(g_r, r_here)
            r_shift = rv - rot_step * np.round(rv / rot_step)
            r_start = rotvec_to_mat(r_shift) @ r_here
            if use_timeline:
                exact = exact_rot_anchor(r_here, g_r)
                if exact is not None:
                    r_start = exact
                    r_shift = rotation_error(exact, r_here)
            backend.set_orientation_ref(r_start)
            AtomicExec(backend, max_cmd_m=args.step_m).move_to(here + shift, tol_m=0.0005, budget=80)
            for _ in range(20):  # let the orientation settle onto the anchored reference
                backend.apply_delta(here + shift - backend.tcp_pos(), 1.0, args.step_m)
            anchor_cm = np.round(shift * 100, 2).tolist()
            anchor_deg = np.round(np.degrees(r_shift), 2).tolist()
        lattice_origin = backend.tcp_pos()
        writer = RolloutWriter(out_dir)
        ep = TokenEpisode(backend, writer,
                          AtomicExec(backend, step_m=args.step_m, max_cmd_m=args.step_m,
                                     max_ctrl_steps=24, gripper_steps=args.gripper_steps,
                                     rot_step_rad=rot_step, max_cmd_rad=rot_step,
                                     lattice_origin=lattice_origin),
                          max_tokens=args.max_tokens)

        def simplified_approach(t_ev: int) -> None:
            """Turn to the grasp orientation in place, XY at a safe height, then straight down."""
            name = pick_grasp[t_ev]
            g_p, g_r = grasp_target(t_ev, backend.object_pos(name), backend.object_rotmat(name))
            pat = args.progress_patience
            ep.chase_pose(backend.tcp_pos(), g_r, tol_m=0.012, tol_rad=tol_rad, progress_patience=pat)
            hover = np.array([g_p[0], g_p[1], max(backend.tcp_pos()[2], g_p[2] + args.approach_clearance)])
            ep.chase_pose(hover, g_r, tol_m=0.012, tol_rad=tol_rad, z_order=True, progress_patience=pat)
            ep.chase_pose(g_p, g_r, tol_m=0.008, tol_rad=tol_rad, z_order=True, progress_patience=pat)

        def timeline_approach(t_ev: int, drawer: Optional[dict]) -> np.ndarray:
            """:func:`simplified_approach` for a pick that is not the episode's first key pose.

            Only the first key pose is anchored on the lattice (in a drawer task, the
            handle), so this grasp point can be up to 1 cm off a node per axis: rounded UP,
            the fingers pinched the top of the bowl rim and it slid out on the lift. So the
            grasp height is the node at or BELOW the demo's (<= 2 cm deeper), XY the nearest.

            ``drawer`` (a drawer opened earlier in the episode: its handle plan): the open
            drawer can overhang the object -- the top drawer reaches the bowl's centre. Straight
            down, the palm landed on the drawer's front panel 15 cm above the bowl. So the hand
            comes down far enough out along the pull that its whole AABB clears the drawer's
            front (measured at the grasp orientation, whole tokens), then slides in at grasp
            height, where it is below the drawer. Returns the grasp node.
            """
            name = pick_grasp[t_ev]
            g_p, g_r = grasp_target(t_ev, backend.object_pos(name), backend.object_rotmat(name))
            o, st = ep.exec.lattice_origin, args.step_m
            node = o + st * np.round((g_p - o) / st)
            node[2] = o[2] + st * np.floor((g_p[2] - o[2]) / st + 1e-6)
            pat = args.progress_patience
            ep.chase_pose(backend.tcp_pos(), g_r, tol_m=0.012, tol_rad=tol_rad, progress_patience=pat)
            pre, k = node, 0
            if drawer is not None:
                u = drawer["u"]
                tcp = backend.tcp_pos()
                front = drawer_front(drawer)
                back = backend.extent("gripper", u)[0] - float(tcp @ u)
                k = int(np.clip(np.ceil((front + 0.01 - back - float(node @ u)) / st), 0, 5))
                # Slide in with the fingertips just over the object (at grasp height the
                # leading finger shoved the bowl away: 6/10 empty grasps), then drop onto it.
                tip = backend.extent("gripper", np.array([0.0, 0.0, 1.0]))[0] - tcp[2]
                top = backend.extent(name, np.array([0.0, 0.0, 1.0]))[1]
                h = int(np.clip(np.ceil((top + 0.005 - tip - node[2]) / st), 0, 4))
                pre = node + k * st * u + np.array([0.0, 0.0, h * st])
            hover = np.array([pre[0], pre[1], max(backend.tcp_pos()[2], node[2] + args.approach_clearance)])
            ep.chase_pose(hover, g_r, tol_m=0.012, tol_rad=tol_rad, z_order=True, progress_patience=pat)
            ep.chase_pose(pre, g_r, tol_m=0.008, tol_rad=tol_rad, z_order=True, progress_patience=pat)
            if k:
                ep.chase_pose(np.array([node[0], node[1], pre[2]]), g_r, tol_m=0.008, tol_rad=tol_rad,
                              progress_patience=pat)
            ep.chase_pose(node, g_r, tol_m=0.008, tol_rad=tol_rad, progress_patience=pat)
            return node

        def drawer_front(drawer: dict) -> float:
            """How far the (opened) drawer reaches along its pull, NOW: its handle's front face."""
            h = backend.handle_of(drawer["body_id"], drawer["u"])
            return float(h["center"] @ drawer["u"] + h["half"] @ np.abs(drawer["u"]))

        def open_drawers() -> list:
            """Every slide joint that stands open NOW, as {"u": out axis, "body_id"} -- an
            object may be placed into a drawer this episode never opened (it starts open)."""
            out = []
            q = backend.joint_positions()
            for n, a in art.items():
                if a["type"] != "slide" or abs(q[n]) < 0.05:
                    continue
                far = max(a["range"], key=abs)
                u = a["axis"] * np.sign(far)
                k = int(np.argmax(np.abs(u)))
                lat = np.zeros(3)
                lat[k] = np.sign(u[k])
                out.append({"u": lat, "body_id": a["body_id"]})
            return out

        def drawer_place_shift(t_r: int, drawer: Optional[dict]) -> Optional[np.ndarray]:
            """Where to put an object the demo released INTO the drawer opened earlier.

            The demo's absolute release point assumes the cabinet where the demo had it --
            re-sampled per reset, +-1 cm, plus a different opening. Aimed there, the bowl
            landed on the drawer's front panel and shoved the drawer half shut (2/10). So:
            the middle of the drawer's floor, NOW, at the demo's height above that floor
            (heights do not depend on the placement). None when the release was not over it.
            """
            if drawer is None:
                return None
            name = carried[t_r]
            bid = drawer["body_id"]
            fl = backend.floor_of(bid)
            demo_rel = objs[t_r][name] - (jbody_at(bid, t_r))
            now_rel_floor = fl["center"] - backend.body_pos(bid)
            if np.any(np.abs(demo_rel[:2] - now_rel_floor[:2]) > fl["half"][:2] + 0.03):
                return None
            target = np.r_[fl["center"][:2], backend.body_pos(bid)[2] + demo_rel[2]]
            return target - objs[t_r][name]

        def jbody_at(body_id: int, t: int) -> np.ndarray:
            n = next(j for j, a in art.items() if a["body_id"] == body_id)
            return jbody[n][t]

        def place_into_drawer(t_r: int, drawer: dict) -> dict:
            """Put the carried object down in the middle of the drawer's floor: up clear of
            its walls, across, straight down, NOW.

            Following the demo's arrival instead carried the object in low, which shoved the
            drawer 11 cm shut on the way -- and then the release point, computed before that,
            was outside it and the object fell on the floor.
            """
            pat, st = args.progress_patience, args.step_m
            name = carried[t_r]
            fl = backend.floor_of(drawer["body_id"])
            wall_top = backend.body_top(drawer["body_id"])
            obj_lo = backend.extent(name, np.array([0.0, 0.0, 1.0]))[0]
            off = backend.tcp_pos() - backend.object_pos(name)     # rigid while held
            u = drawer["u"]
            radius = 0.5 * max(np.ptp(backend.extent(name, ax)) for ax in (np.array([1.0, 0, 0]),
                                                                          np.array([0, 1.0, 0])))
            # In the part of the drawer that is OUT of the cabinet: over the floor's centre
            # the object's rim caught the shelf above (the gripper was forced open 3.6 -> 10.8
            # mm and the bowl never came down). Closing the drawer carries it in.
            spot = fl["center"] + u * (fl["half"] @ np.abs(u) - radius - 0.01)
            target = np.r_[spot[:2], fl["center"][2] + fl["half"][2] +
                           (backend.object_pos(name)[2] - obj_lo) + 0.005] + off
            ref = backend.orientation_ref()
            safe = max(backend.tcp_pos()[2], wall_top + (backend.tcp_pos()[2] - obj_lo) + 0.02)
            ep.chase_pose(np.r_[backend.tcp_pos()[:2], safe], ref, tol_m=0.008, tol_rad=tol_rad,
                          progress_patience=pat)
            ep.chase_pose(np.r_[target[:2], max(safe, target[2])], ref, tol_m=0.008, tol_rad=tol_rad,
                          z_order=True, progress_patience=pat)
            ep.chase_pose(target, ref, tol_m=0.008, tol_rad=tol_rad, z_order=True,
                          progress_patience=pat)
            err = backend.object_pos(name) - (target - off)
            return {"kind": "place", "into": "drawer_floor",
                    "err_cm": np.round(err * 100, 1).tolist()}

        def clear_of_drawer(name: str, drawer: dict) -> None:
            """Drag the grasped object out along the pull, on the table, until its far side is
            1 cm past the drawer's front: lifted where it stood, the bowl met the underside of
            the open top drawer and was pried out of the fingers (2/10)."""
            u, st = drawer["u"], args.step_m
            front = drawer_front(drawer) + 0.01
            back = backend.extent(name, u)[0]
            k = int(np.clip(np.ceil((front - back) / st), 0, 5))
            for _ in range(k):
                ep.emit(token_for_axis(int(np.argmax(np.abs(u))), float(u[int(np.argmax(np.abs(u)))])))

        def simplified_transport(t_g: int, t_r: int, guard=None, shift=None) -> None:
            """Leave like the demo, cross at the site clearance height, turn, arrive like the demo.

            The final alignment and descent are the shared retarget before RELEASE.
            ``guard`` (optional) runs after every sub-move and may raise to abort; ``shift``
            (optional) moves the demo's object targets (see :func:`drawer_place_shift`).
            """
            shift = np.zeros(3) if shift is None else shift
            name = carried[t_r]
            pat = args.progress_patience
            guard = guard or (lambda: None)
            ref = backend.orientation_ref()
            r_place = rotvec_to_mat(kept_rot(rotation_error(rot[t_r], rot[t_g]))) @ ref
            side = lambda t, s: float(np.linalg.norm(pos[t][:2] - pos[s][:2]))
            t_exit = next((t for t in range(t_g, t_r + 1) if side(t, t_g) >= args.site_radius_m), t_r)
            t_entry = next((t for t in range(t_r, t_g - 1, -1) if side(t, t_r) >= args.site_radius_m), t_g)
            # Leave the pick site along the demo's own displacement (order: largest first).
            ep.chase_pose(backend.tcp_pos() + (pos[t_exit] - pos[t_g]), ref, tol_m=0.012,
                          tol_rad=tol_rad, progress_patience=pat)
            guard()
            # Arrival, aimed by the carried object: where the demo's object was on arrival.
            entry = backend.tcp_pos() + (objs[t_entry][name] + shift - backend.object_pos(name))
            carry_z = max(backend.tcp_pos()[2], entry[2])
            ep.chase_pose(np.array([entry[0], entry[1], carry_z]), ref, tol_m=0.012,
                          tol_rad=tol_rad, z_order=True, progress_patience=pat)
            guard()
            ep.chase_pose(backend.tcp_pos(), r_place, tol_m=0.012, tol_rad=tol_rad, progress_patience=pat)
            guard()
            entry = backend.tcp_pos() + (objs[t_entry][name] + shift - backend.object_pos(name))
            ep.chase_pose(entry, r_place, tol_m=0.012, tol_rad=tol_rad, z_order=True, progress_patience=pat)

        reason, retarget_cm, n_corners, grip_widths_mm = "", [], 0, []
        tail_followed = False
        segment_modes: list[str] = []
        contact_results: list[dict] = []

        def place_retarget(t_ev: int, shift=None) -> None:
            """Steer the carried object onto where the demo released it: XY, then down."""
            name = carried[t_ev]
            shift = np.zeros(3) if shift is None else shift
            delta = objs[t_ev][name] + shift - backend.object_pos(name)
            retarget_cm.append(round(float(np.linalg.norm(delta)) * 100, 2))
            tcp = backend.tcp_pos()
            ref = backend.orientation_ref()
            ep.chase_pose(np.array([tcp[0] + delta[0], tcp[1] + delta[1], tcp[2]]), ref,
                          tol_m=0.008, tol_rad=tol_rad, z_order=True)
            delta = objs[t_ev][name] + shift - backend.object_pos(name)
            tcp = backend.tcp_pos()
            ep.chase_pose(np.array([tcp[0], tcp[1], tcp[2] + delta[2]]), ref,
                          tol_m=0.008, tol_rad=tol_rad, z_order=True)

        def drive_joint(c: dict) -> dict:
            """Whole tokens along the joint's world axis until it has travelled the demo's dq."""
            name, a = c["joint"], art[c["joint"]]
            q_start = backend.joint_positions()[name]
            q_goal = q_start + c["dq"]
            if a["type"] == "slide":
                disp = a["axis"] * c["dq"]
                k = int(np.argmax(np.abs(disp)))
                tok, unit, n_need = token_for_axis(k, disp[k]), args.step_m, abs(disp[k]) / args.step_m
            else:
                rv = a["axis"] * c["dq"]
                k = int(np.argmax(np.abs(rv)))
                tok, unit, n_need = rot_token_for_axis(k, rv[k]), rot_step, abs(rv[k]) / rot_step
            stalled = 0
            for _ in range(int(np.ceil(n_need)) + 3):
                if abs(backend.joint_positions()[name] - q_goal) < 0.5 * unit:
                    break
                before = backend.joint_positions()[name]
                ep.emit(tok)
                if abs(backend.joint_positions()[name] - before) < 0.2 * unit:
                    stalled += 1
                    if stalled >= 2:
                        break
                else:
                    stalled = 0
            q_end = backend.joint_positions()[name]
            return {"kind": "joint", "joint": name, "token": tok, "dq_demo": round(c["dq"], 4),
                    "dq_done": round(q_end - q_start, 4)}

        def do_handle_pull(c: dict) -> None:
            """Turn in place, come in level from in front of the handle, GRASP the bar, pull
            along the joint axis, RELEASE, back off along the pull."""
            pat = args.progress_patience
            hp = handle_grasp(c)
            p_g, r_g, u = hp["pos"], hp["rot"], hp["u"] * args.step_m
            ep.chase_pose(backend.tcp_pos(), r_g, tol_m=0.012, tol_rad=tol_rad, progress_patience=pat)
            ep.chase_pose(p_g + 3 * u, r_g, tol_m=0.008, tol_rad=tol_rad, z_order=True,
                          progress_patience=pat)
            ep.chase_pose(p_g, r_g, tol_m=0.008, tol_rad=tol_rad, progress_patience=pat)
            at = {"handle": hp["geom"],
                  "reach_err_mm": np.round((backend.tcp_pos() - p_g) * 1000, 1).tolist(),
                  "reach_err_deg": round(float(np.degrees(np.linalg.norm(
                      rotation_error(r_g, backend.tcp_rotmat())))), 1)}
            ep.emit(GRASP)
            at["grip_mm"] = round(backend.gripper_width() * 1000, 2)
            grip_widths_mm.append(at["grip_mm"])
            res = drive_joint(c)
            ep.emit(RELEASE)
            grip_widths_mm.append(round(backend.gripper_width() * 1000, 2))
            ep.chase_pose(backend.tcp_pos() + 2 * u, backend.orientation_ref(), tol_m=0.008,
                          tol_rad=tol_rad, progress_patience=pat)
            contact_results.append({**at, **res})

        def do_push(c: dict) -> None:
            """Axis by axis in the demo's order (:func:`push_plan`): up clear of the object,
            fingers turned along the axis, over its centre, down into it (:func:`push_pose`),
            then one token at a time while it follows. When it stops following short of the
            demo's end (the finger has ridden over the rim), engage again -- up to 4 times."""
            pat, st = args.progress_patience, args.step_m
            name = c["object"]
            plan = push_plan(c)
            res = {"kind": "push", "object": name, "how": "inside/outside", "phases": []}
            # The demo's order first, then up to two correction rounds for whatever drifted
            # (pushing -X drifted the plate 3.4 cm in Y, out of the goal region).
            n = len(plan["order"])
            for i, k in enumerate(list(plan["order"]) * 3):
                if i >= n and abs(plan["goal"][k] - backend.object_pos(name)[k]) < 0.015:
                    continue
                left = plan["goal"][k] - backend.object_pos(name)[k]
                if abs(left) < 0.5 * st:
                    continue
                tok, sign = token_for_axis(k, left), float(np.sign(left))
                ph = {"token": tok, "engagements": 0, "tokens": 0, "from": []}
                for _ in range(4):
                    if (plan["goal"][k] - backend.object_pos(name)[k]) * sign < 0.5 * st:
                        break
                    ph["engagements"] += 1
                    o = ep.exec.lattice_origin
                    top = backend.extent(name, np.array([0.0, 0.0, 1.0]))[1]
                    tip = backend.extent("gripper", np.array([0.0, 0.0, 1.0]))[0] - backend.tcp_pos()[2]
                    safe_z = top + 0.01 - tip
                    if backend.tcp_pos()[2] < safe_z:
                        up_to = o[2] + st * np.ceil((safe_z - o[2]) / st)
                        ep.chase_pose(np.r_[backend.tcp_pos()[:2], up_to], backend.orientation_ref(),
                                      tol_m=0.008, tol_rad=tol_rad, progress_patience=pat)
                    p, r_p, mode = push_engagement(c, k)
                    ph["from"].append(mode)
                    ep.chase_pose(backend.tcp_pos(), r_p, tol_m=0.012, tol_rad=tol_rad, progress_patience=pat)
                    node = o + st * np.round((p - o) / st)
                    node[2] = o[2] + st * np.floor((p[2] - o[2]) / st + 1e-6)
                    safe = max(backend.tcp_pos()[2], node[2] + st)
                    ep.chase_pose(np.r_[node[:2], safe], r_p, tol_m=0.008,
                                  tol_rad=tol_rad, z_order=True, progress_patience=pat)
                    # Down. From outside, stop at the first blocked token (the table); inside,
                    # the fingers must be pressed into the rim slopes or the push rides over
                    # at once (measured: stopping early there, 0 cm of 20).
                    ep.chase_pose(node, r_p, tol_m=0.008, tol_rad=tol_rad,
                                  stall_limit=1 if ph["from"][-1] == "outside" else 3,
                                  progress_patience=pat)
                    stalled, used, gained = 0, 0, 0.0
                    for _ in range(20):
                        if (plan["goal"][k] - backend.object_pos(name)[k]) * sign < 0.5 * st:
                            break
                        before = backend.object_pos(name)[k]
                        ep.emit(tok)
                        used += 1
                        step_gain = (backend.object_pos(name)[k] - before) * sign
                        gained += step_gain
                        if used > 1 and step_gain < 0.3 * st:
                            stalled += 1
                            if stalled >= 2:
                                break
                        else:
                            stalled = 0
                    ph["tokens"] += used
                    if gained < 0.5 * st:
                        break  # this engagement did nothing: another will not either
                ph["left_cm"] = round(float(plan["goal"][k] - backend.object_pos(name)[k]) * 100, 1)
                res["phases"].append(ph)
            ep.chase_pose(backend.tcp_pos() + np.array([0.0, 0.0, 2 * st]), backend.orientation_ref(),
                          tol_m=0.008, tol_rad=tol_rad, progress_patience=pat)
            err = (plan["goal"] - backend.object_pos(name))[:2]
            res["final_err_cm"] = round(float(np.linalg.norm(err)) * 100, 2)
            contact_results.append(res)

        def do_contact(c: dict) -> None:
            if c["kind"] == "push":
                return do_push(c)
            if opens_out(c):
                return do_handle_pull(c)
            if is_knob(c):
                return do_knob_turn(c)
            return do_joint_push(c)

        try:
            tail_deferred = None
            if use_timeline:
                if not demo["timeline_supported"]:
                    raise _Unsupported("grasp that is not pick-and-place in a contact demo")
                drawer = None  # the last drawer opened in this episode
                def holding():
                    if backend.gripper_width() < args.empty_grasp_mm / 1000:
                        raise _Dropped()
                for act in timeline:
                    if act["kind"] == "pick":
                        segment_modes.append("pick:simplified")
                        node = timeline_approach(act["t_g"], drawer)
                        name = pick_grasp[act["t_g"]]
                        contact_results.append({"kind": "pick", "object": name,
                                                "reach_err_mm": np.round((backend.tcp_pos() - node) * 1000, 1).tolist()})
                        ep.emit(GRASP)
                        grip_widths_mm.append(round(backend.gripper_width() * 1000, 2))
                        if backend.gripper_width() < args.empty_grasp_mm / 1000:
                            reason = "empty_grasp"
                            break
                        if drawer is not None:
                            clear_of_drawer(name, drawer)
                            holding()
                        into = next((cand for cand in [drawer] + open_drawers()
                                     if cand is not None
                                     and drawer_place_shift(act["t_r"], cand) is not None), None)
                        simplified_transport(act["t_g"], act["t_r"], guard=holding,
                                             shift=drawer_place_shift(act["t_r"], into) if into else None)
                        holding()
                        if into is not None:
                            contact_results.append(place_into_drawer(act["t_r"], into))
                        else:
                            place_retarget(act["t_r"])
                        ep.emit(RELEASE)
                        grip_widths_mm.append(round(backend.gripper_width() * 1000, 2))
                        ep.go_z(backend.tcp_pos()[2] + RELEASE_RETREAT_M)
                    else:
                        segment_modes.append(f"contact:{act['kind']}")
                        do_contact(act)
                        if act["kind"] == "joint" and art[act["joint"]]["type"] == "slide":
                            drawer = {"u": handle_grasp(act)["u"], "body_id": art[act["joint"]]["body_id"]}
                    n_corners += 3
                bounds_loop = []
            else:
                bounds_loop = range(len(bounds) - 1)
            for si in bounds_loop:
                a, b = bounds[si], bounds[si + 1]
                if si == len(bounds) - 2 and events and events[-1][1] == RELEASE:
                    # After the last RELEASE the demo is only walking away. Chasing it
                    # appended up to 30 tokens of wandering after the task was already done
                    # (and it chases the DEMO's hand, which the placement retarget moved
                    # away from on purpose -- RoboLab's "curl inward"). Lift (done above),
                    # settle, and fall back to the tail only if the predicate is not met.
                    tail_deferred = (a, b)
                    break
                t_ev, tok = events[si] if si < len(events) else (None, None)
                if simp and tok == GRASP and t_ev in pick_grasp:
                    simplified_approach(t_ev)
                    n_corners += 2
                    segment_modes.append("approach:simplified")
                elif simp and tok == RELEASE and t_ev in pick_release:
                    simplified_transport(pick_release[t_ev], t_ev)
                    n_corners += 2
                    segment_modes.append("transport:simplified")
                else:
                    segment_modes.append("followed")
                    seg_c = [a + c for c in corners(pos[a:b + 1], rot[a:b + 1])][1:]
                    if si < len(events) and events[si][1] == GRASP and seg_c:
                        # Clear the approach zone: the grasp is reached by XY (+rotation) at a
                        # safe height, then straight down (z_order does the ordering).
                        gp = pos[seg_c[-1]]
                        seg_c = [c for c in seg_c[:-1]
                                 if not (pos[c][2] < gp[2] + args.approach_clearance
                                         and np.linalg.norm(pos[c][:2] - gp[:2]) < 2 * args.approach_clearance)] + [seg_c[-1]]
                    n_corners += len(seg_c)
                    for k, c in enumerate(seg_c):
                        tight = k == len(seg_c) - 1 and si < len(events)
                        tp, tr_ = pos[c], rot[c]
                        if tight and events[si][1] == GRASP and events[si][0] in grasp_obj:
                            name = grasp_obj[events[si][0]]
                            tp, tr_ = grasp_target(events[si][0], backend.object_pos(name),
                                                   backend.object_rotmat(name))
                        ep.chase_pose(tp, tr_, tol_m=0.008 if tight else 0.012,
                                      tol_rad=tol_rad, z_order=True)
                if si < len(events):
                    if tok == RELEASE and t_ev in carried:
                        place_retarget(t_ev)
                    ep.emit(tok)
                    grip_widths_mm.append(round(backend.gripper_width() * 1000, 2))
                    if tok == GRASP and backend.gripper_width() < args.empty_grasp_mm / 1000:
                        reason = "empty_grasp"
                        break
                    if tok == RELEASE:
                        ep.go_z(backend.tcp_pos()[2] + RELEASE_RETREAT_M)
            settled = ep.settle_steps(SETTLE_STEPS) if not reason else None
            if settled is None and not reason and tail_deferred is not None:
                a, b = tail_deferred
                for c in [a + c for c in corners(pos[a:b + 1], rot[a:b + 1])][1:]:
                    ep.chase_pose(pos[c], rot[c], tol_m=0.012, tol_rad=tol_rad, z_order=True)
                settled = ep.settle_steps(SETTLE_STEPS)
                tail_followed = True
            success = settled is not None
            if not success and not reason:
                reason = "predicate_false"
        except TokenBudgetExceeded as exc:
            success, settled, reason = False, None, str(exc)
        except _Unsupported as exc:
            success, settled, reason = False, None, f"unsupported: {exc}"
        except _Dropped:
            success, settled, reason = False, None, "dropped"

        final = None
        if success:
            # One frame AFTER success: the DONE sample's image, distinct from the last
            # action frame (see the RELEASE/DONE collision on RoboLab).
            av, wr = backend.grab_frames()
            from PIL import Image
            (out_dir / "final").mkdir(parents=True, exist_ok=True)
            Image.fromarray(av).save(out_dir / "final" / "agentview.png")
            Image.fromarray(wr).save(out_dir / "final" / "wrist.png")
            final = {"agentview": "final/agentview.png", "wrist": "final/wrist.png",
                     "state_index": len(backend.frame_states) - 1}
        out_dir.mkdir(parents=True, exist_ok=True)  # an unsupported demo ends before any token
        if backend.frame_states:
            np.savez_compressed(out_dir / "states.npz", states=np.stack(backend.frame_states))
        tokens = list(writer.tokens)
        meta = {
            "source": f"follow_{task_key}", "method": "closed_loop_follower", "sim": "libero",
            "mode": mode, "segment_modes": segment_modes,
            "rot_deadband_deg": args.rot_deadband_deg if simp else None,
            "task": language, "task_key": task_key, "bddl": str(args.bddl),
            "demo_file": str(args.demos), "demo_index": i, "demo_len": int(len(states)),
            "step_m": args.step_m, "rot_step_deg": args.rot_step_deg,
            "success": bool(success), "reason": reason, "settle_steps_used": settled,
            "corners": n_corners, "retarget_cm": retarget_cm, "anchor_shift_cm": anchor_cm, "anchor_rot_deg": anchor_deg,
            "grip_widths_mm": grip_widths_mm, "gripper_steps": args.gripper_steps,
            "tail_followed": tail_followed,
            "contacts": [{k: v for k, v in c.items()} for c in demo["contacts"]],
            "contact_results": contact_results,
            "n_move": sum(t.startswith("MV_") for t in tokens),
            "n_rotate": sum(t.startswith("RT_") for t in tokens),
            "rot_cancelled": cancelled_rotations(tokens),
            "final_frame": final, "states_file": "states.npz", "fixture_poses": fixture_poses,
            "wall_s": round(time.time() - t0, 1),
            **{k: getattr(backend, k) for k in (
                "agentview_camera", "agentview_rotation_degrees", "agentview_flip",
                "wrist_camera", "wrist_rotation_degrees", "wrist_flip")},
            **extra,
        }
        writer.close(meta)
        meta["steps"] = writer.step
        return meta

    with h5py.File(args.demos) as h:
        demo_names = sorted(h["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        ids = parse_ids(args.demo_ids, len(demo_names))
        raw = {i: (np.asarray(h["data"][demo_names[i]]["states"]),
                   np.asarray(h["data"][demo_names[i]]["actions"])) for i in ids}

    kept, summary = 0, []
    for i in ids:
        demo = prepare(*raw[i])
        out_dir = out_root / f"rollout_{kept:03d}"
        meta = run(i, demo, args.mode, out_dir, {})
        if not meta["success"] and args.mode == "simplified" and meta["reason"] == "empty_grasp":
            # The demo's own grasp was a lucky one (a pinch on a can's rim): grip the object
            # where its shape allows instead, and run the episode again.
            shutil.rmtree(out_dir, ignore_errors=True)
            meta = run(i, demo, args.mode, out_dir,
                       {"grasp_mode": "constructed", "demo_grasp_result": meta["reason"]},
                       grasp_mode="constructed")
        if not meta["success"] and args.mode == "simplified" and args.fallback_full:
            first = {k: meta[k] for k in ("reason", "steps", "n_rotate")}
            shutil.rmtree(out_dir, ignore_errors=True)
            meta = run(i, demo, "full", out_dir, {"fallback_from": "simplified", "simplified_result": first})
        summary.append({k: meta.get(k) for k in ("demo_index", "mode", "success", "reason", "n_move", "n_rotate",
                                                 "rot_cancelled", "corners", "retarget_cm", "grip_widths_mm",
                                                 "segment_modes", "wall_s", "steps")})
        print(json.dumps(summary[-1]), flush=True)
        if meta["success"] or args.keep_failures:
            kept += 1
        else:
            shutil.rmtree(out_dir, ignore_errors=True)

    backend.close()
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[follow:{task_key}] kept {kept}/{len(ids)} -> {out_root}", flush=True)
    return 0 if kept else 2


if __name__ == "__main__":
    raise SystemExit(main())
