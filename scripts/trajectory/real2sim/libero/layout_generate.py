#!/usr/bin/env python3
"""MVTOKEN trajectories for LIBERO scenes that have NO human demonstration.

``follow_tokenize.py`` needs a demo: it reads the recorded states, finds the gripper
events, and retargets them onto the objects. The Objects Layout variants (extra confounding
objects on the table) ship a bddl but no demo and no init states, so that generator cannot
run on them -- which is why the training set has five of LIBERO-plus's seven perturbation
dimensions and scores 57.7% on the one it never saw.

What a spatial demo actually tells the follower, though, is four numbers: WHICH object to
grasp, the hand-object offset at the moment the fingers close, WHERE to release, and the
orientation to hold. Those are stable across a task's demos (measured: grasp-offset IQR
0.3-1.9 cm), so they can be extracted once per base task and replayed in any scene built
from the same task. The motion itself is produced by the SAME executor the demo follower
uses -- AtomicExec / TokenEpisode -- so a token means here exactly what it means there.

Initial states are sampled from the scene's own placement initialiser (``env.reset()``),
the way LIBERO's ``.init`` files were made; the variants ship none.

Only episodes that reach the task's success predicate are kept. Knocking a confounding
object over is NOT a failure -- the predicate only looks at the target bowl, which is also
how LIBERO-plus's own data behaves -- but the rate is recorded so a policy trained to
barge through obstacles cannot go unnoticed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def mirror_to(rotmat: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """The gripper is symmetric under a half turn about its own approach axis.

    Pick whichever of the two equivalent orientations is NEARER the reference. Choosing
    relative to the CURRENT pose instead lets two 90-degree turns accumulate into 170 and
    throw the TCP 24 cm across the table (measured while building the contact skills), so
    the reference is always the pose the episode started in.
    """
    flip = rotmat @ np.diag([-1.0, -1.0, 1.0])
    ang = lambda r: np.arccos(np.clip((np.trace(r @ reference.T) - 1) / 2, -1, 1))
    return rotmat if ang(rotmat) <= ang(flip) else flip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", required=True, help="extract_template.py output")
    ap.add_argument("--bddl-dir", default=None, help="default: LIBERO's libero_spatial dir")
    ap.add_argument("--scenes", required=True,
                    help="comma-separated bddl stems, or a file with one per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=21, help="init states per scene")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step-m", type=float, default=0.02)
    ap.add_argument("--rot-step-deg", type=float, default=10.0)
    ap.add_argument("--gripper-steps", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--progress-patience", type=int, default=4)
    ap.add_argument("--approach-clearance", type=float, default=0.05)
    ap.add_argument("--empty-grasp-mm", type=float, default=2.5)
    ap.add_argument("--robot-config", default="configs/robot_libero.yaml")
    ap.add_argument("--keep-failures", action="store_true")
    args = ap.parse_args()

    from PIL import Image
    from libero.libero import get_libero_path

    from core.config import camera_contract, load_yaml
    from scripts.trajectory.real2sim.atomic_tokenizer import (
        GRASP, RELEASE, AtomicExec, RolloutWriter, TokenBudgetExceeded, TokenEpisode,
        rotation_error)
    from scripts.trajectory.real2sim.backends import make_backend

    templates = json.loads(Path(args.templates).read_text())
    bddl_dir = Path(args.bddl_dir or (Path(get_libero_path("bddl_files")) / "libero_spatial"))
    # Path.exists() RAISES OSError(36) rather than returning False when the string is
    # longer than a filename may be -- which a comma-joined list of LIBERO scene names
    # always is.
    try:
        as_file = Path(args.scenes).exists()
    except OSError:
        as_file = False
    scenes = ([s.strip() for s in Path(args.scenes).read_text().split() if s.strip()]
              if as_file else [s.strip() for s in args.scenes.split(",") if s.strip()])
    contract = camera_contract(load_yaml(ROOT / args.robot_config))
    tol_rad = float(np.radians(6.0))
    rot_step = float(np.radians(args.rot_step_deg))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # A template is either one pick-and-place (spatial/object) or a list of them
    # (libero_10's "put BOTH x and y in the basket"). Normalise to a list so the executor
    # has one shape to run.
    for k, v in templates.items():
        if "steps" not in v:
            templates[k] = {"steps": [v], "n": v.get("n")}
    # Longest matching base task, so "<base>_add_17" resolves to "<base>"'s template.
    bases = sorted(templates, key=len, reverse=True)
    summary = []

    for scene in scenes:
        base = next((b for b in bases if scene.startswith(b)), None)
        if base is None:
            print(f"[skip] {scene}: no template for any base task", flush=True)
            continue
        t = templates[base]
        bddl = bddl_dir / f"{scene}.bddl"
        if not bddl.exists():
            print(f"[skip] {scene}: no bddl", flush=True)
            continue
        backend = make_backend("libero", bddl_file=str(bddl), **contract)
        steps_t = t["steps"]
        dest = out_root / f"{scene}_layout"
        n_ok = n_knock = 0
        results = []

        for i in range(args.episodes):
            backend.env.seed(args.seed + i)
            backend.env.reset()
            state = backend.sim_state()
            backend.reset_to_state(state)
            objs0 = {n: backend.object_pos(n).copy() for n in backend.movable_objects()}
            # A destination is often a FIXTURE -- the cabinet top, the stove, the microwave,
            # the desk caddy -- and fixtures are not in movable_objects(). Ask the backend
            # for the pose instead of assuming the name is in that list; checking
            # movable_objects() skipped every "put it on the cabinet" task in goal and
            # libero_10, which is most of them.
            need = {o for s in steps_t for o in (s["grasp_object"], s["place_object"])}
            missing = []
            for o in sorted(need):
                try:
                    backend.object_pos(o)
                except Exception:                      # noqa: BLE001
                    missing.append(o)
            if missing:
                print(f"[skip] {scene}: template objects missing {missing}", flush=True)
                break
            r_home = backend.tcp_rotmat()
            g_rot = mirror_to(np.asarray(steps_t[0]["grasp_rotmat"]), r_home)
            g_pos = backend.object_pos(steps_t[0]["grasp_object"]) \
                + np.asarray(steps_t[0]["grasp_offset"])

            # Anchor the lattice on the grasp pose, BEFORE the first recorded frame, exactly
            # as the demo follower does: every token then lands on a node instead of losing
            # ~0.5 mm a step. Not an action -- it is part of the initial state.
            here = backend.tcp_pos()
            shift = (g_pos - here) - args.step_m * np.round((g_pos - here) / args.step_m)
            rv = rotation_error(g_rot, r_home)
            r_start = backend.tcp_rotmat()
            backend.set_orientation_ref(r_start)
            AtomicExec(backend, max_cmd_m=args.step_m).move_to(here + shift, tol_m=0.0005,
                                                              budget=80)
            lattice_origin = backend.tcp_pos()
            writer = RolloutWriter(dest / f"rollout_{i:03d}")
            ep = TokenEpisode(
                backend, writer,
                AtomicExec(backend, step_m=args.step_m, max_cmd_m=args.step_m,
                           max_ctrl_steps=24, gripper_steps=args.gripper_steps,
                           rot_step_rad=rot_step, max_cmd_rad=rot_step,
                           lattice_origin=lattice_origin),
                max_tokens=args.max_tokens)
            pat = args.progress_patience
            reason = "ok"
            t0 = time.time()
            try:
                for si, sp in enumerate(steps_t):
                    s_rot = mirror_to(np.asarray(sp["grasp_rotmat"]), r_home)
                    g_pos = (backend.object_pos(sp["grasp_object"])
                             + np.asarray(sp["grasp_offset"]))
                    hover = max(g_pos[2], backend.tcp_pos()[2]) + args.approach_clearance
                    # Turn to the grasp orientation where the arm is FOLDED, not at full
                    # stretch: a 30 degree turn at maximum extension throws the TCP 20 cm.
                    ep.chase_pose(backend.tcp_pos(), s_rot, tol_m=0.012, tol_rad=tol_rad,
                                  progress_patience=pat)
                    ep.chase_pose(np.r_[g_pos[:2], hover], s_rot, tol_m=0.011,
                                  tol_rad=tol_rad, progress_patience=pat)
                    # Re-read the object: the approach may have nudged it.
                    g_pos = (backend.object_pos(sp["grasp_object"])
                             + np.asarray(sp["grasp_offset"]))
                    ep.chase_pose(g_pos, s_rot, tol_m=0.011, tol_rad=tol_rad, z_order=True,
                                  progress_patience=pat)
                    ep.emit(GRASP)
                    if backend.gripper_width() * 1000 < args.empty_grasp_mm:
                        reason = f"empty_grasp@{si}"
                        raise RuntimeError(reason)
                    carry = max(hover, backend.tcp_pos()[2] + args.approach_clearance)
                    ep.chase_pose(np.r_[backend.tcp_pos()[:2], carry], s_rot, tol_m=0.011,
                                  tol_rad=tol_rad, z_order=True, progress_patience=pat)
                    r_pos = (backend.object_pos(sp["place_object"])
                             + np.asarray(sp["release_offset"]))
                    ep.chase_pose(np.r_[r_pos[:2], carry], s_rot, tol_m=0.011,
                                  tol_rad=tol_rad, progress_patience=pat)
                    ep.chase_pose(r_pos, s_rot, tol_m=0.011, tol_rad=tol_rad, z_order=True,
                                  progress_patience=pat)
                    ep.emit(RELEASE)
                    ep.run_tokens(["MV_UP", "MV_UP"])
            except TokenBudgetExceeded:
                reason = "token_budget"
            except RuntimeError as exc:
                reason = str(exc) if reason == "ok" else reason
            except Exception as exc:                       # noqa: BLE001 -- recorded, not hidden
                reason = f"{type(exc).__name__}: {exc}"[:120]

            success = bool(backend.success())
            # A confounding object that moved is not a failure (the predicate only looks at
            # the bowl), but a generator that ploughs through the scene would otherwise be
            # invisible in the numbers.
            # 3 cm, not 2: some scenes settle on their own. libero_object's
            # salad_dressing_1 slides 2.02 cm with the arm held completely still --
            # identically on every seed -- so a 2 cm threshold flagged 100% of episodes as
            # "knocked something" and hid whether the generator was actually barging
            # through the scene.
            # Destinations are handled on purpose too: a basket shifts when two groceries
            # are dropped into it, and counting that as "knocked a bystander" made the
            # monitor read 10/10 on every LIVING_ROOM task.
            moved_on_purpose = ({s["grasp_object"] for s in steps_t}
                                | {s["place_object"] for s in steps_t})
            knocked = sorted(n for n, p0 in objs0.items()
                             if n not in moved_on_purpose
                             and np.linalg.norm(backend.object_pos(n) - p0) > 0.03)
            n_knock += bool(knocked)
            final = None
            if success:
                av, wr = backend.grab_frames()
                (dest / f"rollout_{i:03d}" / "final").mkdir(exist_ok=True)
                Image.fromarray(av).save(dest / f"rollout_{i:03d}" / "final" / "agentview.png")
                Image.fromarray(wr).save(dest / f"rollout_{i:03d}" / "final" / "wrist.png")
                final = {"agentview": "final/agentview.png", "wrist": "final/wrist.png",
                         "state_index": writer.step}
            meta = {"task": base, "task_key": scene, "base_task": base,
                    "bddl": str(bddl), "success": success, "reason": reason,
                    "method": "template_layout", "source": "no_demo",
                    "steps": [{"grasp": s["grasp_object"], "place": s["place_object"]}
                              for s in steps_t],
                    "knocked_objects": knocked, "num_steps": writer.step,
                    "step_m": args.step_m, "rot_step_deg": args.rot_step_deg,
                    "gripper_steps": args.gripper_steps, "seed": args.seed + i,
                    "final_frame": final, "wall_s": round(time.time() - t0, 1)}
            np.savez_compressed(dest / f"rollout_{i:03d}" / "states.npz",
                                states=np.asarray(backend.frame_states))
            writer.close(meta)
            n_ok += success
            results.append({"i": i, "success": success, "reason": reason,
                            "steps": meta["num_steps"], "knocked": len(knocked)})
            if not success and not args.keep_failures:
                import shutil
                shutil.rmtree(dest / f"rollout_{i:03d}", ignore_errors=True)
        backend.close()
        summary.append({"scene": scene, "base": base, "success": n_ok,
                        "episodes": len(results), "knocked_any": n_knock,
                        "results": results})
        print(json.dumps({k: summary[-1][k] for k in
                          ("scene", "success", "episodes", "knocked_any")}), flush=True)
        dest.mkdir(parents=True, exist_ok=True)   # a scene skipped before its first episode
        (dest / "_summary.json").write_text(json.dumps(summary[-1], indent=1))

    tot = sum(s["success"] for s in summary)
    n = sum(s["episodes"] for s in summary)
    print(f"[layout] {tot}/{n} = {tot / max(n, 1) * 100:.1f}%  "
          f"碰到干扰物的集 {sum(s['knocked_any'] for s in summary)}")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
