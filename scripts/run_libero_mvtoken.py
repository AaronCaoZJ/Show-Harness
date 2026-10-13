#!/usr/bin/env python3
"""Closed-loop MVTOKEN rollout in LIBERO: the model emits one action unit per frame.

The executor is the SAME one that produced the training labels
(``scripts/trajectory/real2sim/atomic_tokenizer``): a token means exactly what it meant
during generation, and the frames the model is shown go through the same
``backend.grab_frames()`` camera contract that wrote the training PNGs. Only the source of
the tokens changes -- a demonstration before, the policy now.

Two modes:

  --replay <rollout_dir>   re-execute the tokens recorded in that episode's actions.jsonl.
                           No model, no server. This is the harness smoke test: if a
                           recorded episode does not succeed on replay, the runner is
                           wrong and every policy number it produces is noise.

  (default)                query the policy over vLLM's OpenAI API.

Run it with the LIBERO interpreter, which is the only one that has the simulator (build one
with ``bash scripts/setup.sh libero <checkout>``):

  cd <Show-Harness> && PYTHONPATH=$PWD \
  <LIBERO checkout>/.venv/bin/python scripts/run_libero_mvtoken.py ...

``scripts/libero/eval_batch.sh`` wraps that for a whole suite list.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DONE_TOKEN = "DONE"
# What goes into the prompt's "Recent moves" line. Rotations are in it because a 90 degree
# turn is nine identical RT_* tokens and 10 degrees is nearly invisible in a 256x256 frame:
# without the history the model cannot tell which of the nine it is on. Must match the
# converter's --rotation-tokens history (core.action_units.RT_ATOMS).
from core.action_units import MOVE_ATOMS, RT_ATOMS  # noqa: E402

HISTORY_ATOMS = frozenset(MOVE_ATOMS + RT_ATOMS)
RECENT_MOVES_MAX = 5  # rollout_to_llamafactory.RECENT_WINDOW


def bddl_language(bddl_file: str) -> str:
    """The task sentence, read from the bddl the way the data generator read it.

    NOT ``task.language`` from the benchmark, which is wrong in both perturbed suites and in
    different ways: LIBERO-plus appends the variant name to it ("... on the plate table 1"),
    and LIBERO-PRO derives it from the file stem, so exactly the two suites that perturb the
    instruction (``*_lan``, ``*_task``) report the UNperturbed sentence. Neither is a sentence
    the policy was trained on, and neither fails loudly -- the run completes and the number is
    simply measuring something else. The bddl is the one place both suites keep it right.
    """
    from scripts.trajectory.real2sim.libero.follow_tokenize import bddl_language as _lang

    return _lang(bddl_file)


def make_backend_for(bddl: str, robot_config: str):
    from core.config import camera_contract, load_yaml
    from scripts.trajectory.real2sim.backends import make_backend

    contract = camera_contract(load_yaml(ROOT / robot_config))
    return make_backend("libero", bddl_file=bddl, **contract)


def run_episode(
    *,
    backend,
    init_state: np.ndarray,
    language: str,
    out_dir: Path,
    max_tokens: int,
    step_m: float,
    rot_step_deg: float,
    gripper_steps: int,
    decide,
    stuck_guard_mm: float = 5.0,
    stuck_repeat: int = 2,
    ignore_done: bool = False,
) -> dict:
    """One episode. ``decide(agentview, wrist, recent, blocked) -> token`` is the policy.

    ``ignore_done`` puts the episode on the protocol the other LIBERO policies are scored
    under: they have no stop action, so the episode ends only on the env's own success check
    or the step limit. A DONE here would otherwise end it early -- measured on spatial v1,
    every one of the 10 DONE failures had released the bowl 3.1-3.9 cm from the plate centre,
    one lattice step past what the success check accepts. When the policy says DONE it is
    counted, then asked again with DONE withheld (the stuck guard's guided decoding); every
    other step decodes freely. Off by default: it changes the policy, so say so when used.

    ``stuck_guard_mm`` keeps the decode under the same progress constraint the data
    generator already runs under. An MV_* token that moves the TCP less than that has
    not moved the arm -- it is against a joint limit or a surface. The frame then does not
    change and the five-token history saturates, so the next decision sees a byte-identical
    input and greedy decoding re-emits the same token indefinitely. After ``stuck_repeat``
    such emissions the token is withheld from the next decision (guided decoding over the
    rest), which is what the generator does for itself via ``progress_patience``: the policy
    inherits the generator's vocabulary, and this keeps it under the generator's constraint
    as well. Set 0 to decode without it.
    """
    from scripts.trajectory.real2sim.atomic_tokenizer import (
        GRASP, RELEASE, AtomicExec, RolloutWriter, token_kind)

    backend.reset_to_state(init_state)
    # The lattice the tokens are aimed at. Generation anchored it on the demo's first grasp
    # pose (privileged information we do not have at eval time) -- but the anchor only ever
    # shifted the start by less than one token, and what matters is that a token lands on a
    # node rather than drifting: so the lattice starts where the arm starts.
    exec_ = AtomicExec(
        backend,
        step_m=step_m,
        max_cmd_m=step_m,
        max_ctrl_steps=24,
        gripper_steps=gripper_steps,
        rot_step_rad=float(np.radians(rot_step_deg)),
        max_cmd_rad=float(np.radians(rot_step_deg)),
        lattice_origin=backend.tcp_pos(),
    )
    writer = RolloutWriter(out_dir)
    recent: list[str] = []
    tokens: list[str] = []
    no_move: dict[str, int] = {}
    guard_fired = 0
    done_suppressed = 0
    t0 = time.time()
    reason = "max_tokens"

    for _ in range(max_tokens):
        if backend.success():
            reason = "success"
            break
        if backend.frozen():
            reason = "env_terminated"
            break
        agentview, wrist = backend.grab_frames()
        blocked = frozenset(t for t, c in no_move.items() if c >= stuck_repeat)
        if blocked:
            guard_fired += 1
        try:
            token = decide(agentview, wrist, ", ".join(recent) or "none", blocked)
        except RuntimeError as exc:
            # The policy emitted something outside the 15 units (seen: 'MV_STRETCH'). The
            # client refuses to guess, and rightly so -- but one hallucinated token must not
            # kill a 50-episode shard, and silently resampling would hide how often it
            # happens. Count the episode as failed, name the token, keep going.
            if "invalid token" not in str(exc):
                raise
            reason = f"invalid_token:{str(exc).split(chr(39))[1] if chr(39) in str(exc) else '?'}"
            break
        if token == DONE_TOKEN and ignore_done:
            done_suppressed += 1
            token = decide(agentview, wrist, ", ".join(recent) or "none",
                           blocked | {DONE_TOKEN})
        if token is None:
            reason = "no_token"
            break
        if token == DONE_TOKEN:
            reason = "done"
            break

        kind = token_kind(token)
        writer.add_step(token=token, kind=kind, agentview=agentview, wrist=wrist,
                        gripper_closed=exec_.gripper_closed, ee_pose=backend.tcp_pose7(),
                        width=backend.gripper_width())
        before = backend.tcp_pos().copy()
        if kind == "move":
            exec_.move(token)
            if stuck_guard_mm > 0:
                if np.linalg.norm(backend.tcp_pos() - before) * 1000 < stuck_guard_mm:
                    no_move[token] = no_move.get(token, 0) + 1
                else:
                    no_move.clear()   # the arm is free again; nothing is blocked any more
        elif kind == "rotate":
            exec_.rotate(token)
        elif token == GRASP:
            exec_.grasp()
        elif token == RELEASE:
            exec_.release()
        else:
            reason = f"unhandled_token:{token}"
            break

        tokens.append(token)
        if token in HISTORY_ATOMS:
            recent.insert(0, token)
            del recent[RECENT_MOVES_MAX:]

    success = bool(backend.success())
    meta = {"task": language, "success": success, "reason": reason, "num_steps": len(tokens),
            "tokens": tokens, "wall_s": round(time.time() - t0, 1),
            "stuck_guard_mm": stuck_guard_mm, "guard_fired": guard_fired,
            "ignore_done": ignore_done, "done_suppressed": done_suppressed}
    writer.close(meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default=None,
                    help="comma-separated task names; default = the suite's ORIGINAL tasks "
                         "(those whose bddl stem matches a generated dataset directory)")
    ap.add_argument("--dataset-root",
                    default=str(ROOT / "rollouts/libero_token_dataset/rollouts/spatial"),
                    help="used only to recover the original task list")
    ap.add_argument("--episodes", type=int, default=10, help="init states per task")
    ap.add_argument("--start-episode", type=int, default=0)
    # --out, not --log-dir: the other simulator runners hand EpisodeLogger a place to archive
    # a run, while this writes teleop-format rollouts (agentview/NNNN.png + actions.jsonl +
    # metadata.json) that train/data_preparation/rollouts_to_alpaca.py consumes directly. It
    # is training data, so it keeps a name that says so.
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", "--max-tokens", dest="max_steps", type=int, default=200,
                    help="decision limit per episode; one decision is one action token")
    ap.add_argument("--step-m", type=float, default=0.02)
    ap.add_argument("--rot-step-deg", type=float, default=10.0)
    ap.add_argument("--gripper-steps", type=int, default=25)
    ap.add_argument("--robot-config", default="configs/robot_libero.yaml")
    # policy
    ap.add_argument("--vlm-url", "--base-url", dest="vlm_url",
                    default=os.environ.get("VLM_URL") or os.environ.get("VLLM_BASE_URL")
                    or "http://127.0.0.1:8000/v1",
                    help="OpenAI-compatible endpoint of the served policy")
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL"),
                    help="the LoRA name the server was started with")
    # Same pair as the RoboLab runner: the version is the prompts/ subfolder
    # matching the served LoRA's training data. --prompt still takes a file directly.
    ap.add_argument("--prompts-dir", default=str(ROOT / "prompts"))
    ap.add_argument("--version", default="v5",
                    help="prompt version subfolder under --prompts-dir, matching the served "
                         "LoRA's training data. Reads <prompts-dir>/<version>/"
                         "mvtoken_generator_lite.txt")
    ap.add_argument("--prompt", default=None,
                    help="an explicit prompt file, overriding --prompts-dir/--version")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    # harness check
    ap.add_argument("--replay", default=None,
                    help="a generated rollout dir (or a directory of them): re-execute its "
                         "recorded tokens instead of querying a policy")
    ap.add_argument("--stuck-guard-mm", type=float, default=5.0,
                    help="an MV_* token moving the TCP less than this many millimetres is "
                         "withheld from the next decision, the same progress constraint the "
                         "data generator runs under (0 decodes without it)")
    ap.add_argument("--stuck-repeat", type=int, default=2,
                    help="how many no-progress emissions before a token is withheld")
    ap.add_argument("--ignore-done", action="store_true",
                    help="end episodes only on env success or --max-steps, like the other "
                         "LIBERO policies; a DONE is counted and re-asked without DONE")
    ap.add_argument("--libero-plus", action="store_true",
                    help="LIBERO-plus protocol: every task in the perturbed suite, ONE trial "
                         "each (their README: num_trials_per_task 50 -> 1)")
    ap.add_argument("--shard", action="store_true", help="split the task list across workers")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true",
                    help="just re-read the per-task summary.json files under --out and print "
                         "the total. Run this ONCE after a sharded run: a worker that "
                         "summarises on its own way out sees whatever its siblings have "
                         "finished so far and prints a total that is neither its own nor final.")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        summarize(out_root)
        return
    if args.replay:
        run_replay(args, out_root)
        return

    from core.vlm.mvtoken_roles import MvTokenController
    from core.vlm.vlm_client import VLMClient

    # NOT .strip()ed: the converter's _load_prompt keeps the file's trailing newline, so the
    # training text ended "explanation:\n<|im_end|>". (The serving chat template's |trim
    # removes it again -- measured at 0.1pt on a held-out set -- but the two sides should
    # still say the same thing.)
    prompt_file = (Path(args.prompt) if args.prompt else
                   Path(args.prompts_dir) / args.version / "mvtoken_generator_lite.txt")
    if not prompt_file.is_file():
        raise SystemExit(f"prompt not found: {prompt_file}\n"
                         f"  pass --prompt <file>, or --version <subfolder of {args.prompts_dir}>")
    prompt_template = prompt_file.read_text(encoding="utf-8")
    client = VLMClient(base_url=args.vlm_url, model=args.model, api_key="EMPTY",
                       timeout_s=args.timeout_s, max_tokens=8, temperature=args.temperature)
    client.health_check(wait_s=600, poll_s=5)
    controller = MvTokenController(client, prompt_template)
    print(f"allowed tokens ({len(controller.allowed_tokens)}): {controller.allowed_tokens}")

    if args.libero_plus:
        run_libero_plus(args, out_root, controller)
        return

    for task_name, bddl, init_states in iter_tasks(args):
        language = bddl_language(bddl)
        backend = make_backend_for(bddl, args.robot_config)
        results = []
        for i in range(args.start_episode, min(args.start_episode + args.episodes,
                                               len(init_states))):
            def decide(agentview, wrist, recent, blocked, _c=controller, _lang=language):
                allow = [t for t in _c.allowed_tokens if t not in blocked] if blocked else None
                resp = _c.decide(task=_lang, gripper_state="", recent_moves=recent,
                                 agentview_image=agentview, wrist_image=wrist,
                                 constrain_to=allow)
                return resp.token

            m = run_episode(backend=backend, init_state=init_states[i], language=language,
                            out_dir=out_root / task_name / f"rollout_{i:03d}",
                            max_tokens=args.max_steps, step_m=args.step_m,
                            rot_step_deg=args.rot_step_deg, gripper_steps=args.gripper_steps,
                            decide=decide, stuck_guard_mm=args.stuck_guard_mm,
                            stuck_repeat=args.stuck_repeat, ignore_done=args.ignore_done)
            results.append(m)
            print(f"  {task_name} ep{i:03d}: {'OK ' if m['success'] else 'FAIL'} "
                  f"{m['reason']:14s} {m['num_steps']:3d} tokens {m['wall_s']:5.1f}s", flush=True)
        backend.close()
        n_ok = sum(r["success"] for r in results)
        print(f"{task_name}: {n_ok}/{len(results)}", flush=True)
        (out_root / task_name / "summary.json").write_text(
            json.dumps({"task": language, "success": n_ok, "episodes": len(results),
                        "results": results}, ensure_ascii=False, indent=2))
    # Only the unsharded run owns the whole output directory; a worker that was handed a
    # --tasks subset would otherwise report a total built from its siblings' half-finished
    # work. Sharded runs call --summary-only once, after every worker has exited.
    if not args.tasks:
        summarize(out_root)


def iter_libero_plus(args):
    """Every LIBERO-plus task in the suite, one trial each -- their published protocol.

    The suite is the ORIGINAL tasks perturbed along seven dimensions. Three of them
    (Background Textures, Light Conditions, Objects Layout) ship their own bddl; the other
    four (Camera Viewpoints, Sensor Noise, Robot Initial States, Language Instructions)
    encode their parameters in the task NAME and are applied by LIBERO-plus's own env, so
    the bddl path it reports does not exist on disk and must still be passed through
    verbatim.

    The instruction is the one the policy was trained on -- the base task's bddl sentence --
    EXCEPT for Language Instructions, where the rewritten sentence is the whole point of the
    perturbation and lives in ``task.language``. (``task.language`` is useless everywhere
    else: it is the task name with the variant appended, "... on the plate table 1".)
    """
    import json as _json

    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[args.suite]()
    names = suite.get_task_names()
    cls = _json.load(open(os.path.join(get_libero_path("benchmark_root"),
                                       "benchmark/task_classification.json")))[args.suite]
    bddl_root = Path(get_libero_path("bddl_files")) / args.suite
    base_names = sorted((d.name[: -len("_follow")]
                         for d in Path(args.dataset_root).glob("*_follow") if d.is_dir()),
                        key=len, reverse=True)
    base_language = {b: bddl_language(str(bddl_root / f"{b}.bddl")) for b in base_names}

    items = cls if not args.shard else [it for k, it in enumerate(cls)
                                        if k % args.shards == args.shard_index]
    for it in items:
        name = it["name"]
        i = names.index(name)
        t = suite.get_task(i)
        base = next((b for b in base_names if name.startswith(b)), None)
        if base is None:
            raise SystemExit(f"cannot map {name} to one of the {len(base_names)} base tasks")
        language = t.language if it["category"] == "Language Instructions" else base_language[base]
        yield {"name": name, "base": base, "category": it["category"],
               "difficulty": it.get("difficulty_level"), "language": language,
               "bddl": os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
               "init_states": np.asarray(suite.get_task_init_states(i))}


def iter_tasks(args):
    """(task_name, bddl_path, init_states) for the tasks to evaluate."""
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[args.suite]()
    by_name = {}
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        by_name[Path(t.bddl_file).stem] = (i, t)

    if args.tasks:
        names = [n.strip() for n in args.tasks.split(",") if n.strip()]
    else:
        # The suite in LIBERO-plus is the ORIGINAL tasks plus thousands of perturbed
        # variants; the originals are the ones our generated dataset has a directory for.
        names = sorted(d.name[: -len("_follow")]
                       for d in Path(args.dataset_root).glob("*_follow") if d.is_dir())
    # LIBERO-plus registers only the PERTURBED variants in its suites (2402 for spatial);
    # the 10 original tasks still ship their bddl and their standard 50-state
    # `.pruned_init`, they are just not in the registry. Fall back to those files so the
    # in-distribution baseline uses exactly LIBERO's own eval states.
    import torch

    bddl_root = Path(get_libero_path("bddl_files")) / args.suite
    init_root = Path(get_libero_path("init_states")) / args.suite
    for name in names:
        if name in by_name:
            i, t = by_name[name]
            yield (name,
                   os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
                   suite.get_task_init_states(i))
            continue
        bddl = bddl_root / f"{name}.bddl"
        init = init_root / f"{name}.pruned_init"
        if not bddl.exists() or not init.exists():
            raise SystemExit(f"task {name}: no registry entry and no {bddl.name}/{init.name}")
        yield name, str(bddl), np.asarray(torch.load(init, weights_only=False))


def run_libero_plus(args, out_root: Path, controller) -> None:
    """One trial per task over the whole perturbed suite, appending JSONL as it goes.

    Results stream to a per-shard JSONL rather than a directory tree: 2402 tasks x 1 episode
    is 2402 directories of one rollout each, and the point here is the success rate per
    perturbation dimension, not the frames.
    """
    shard = out_root / f"results.shard{args.shard_index:02d}.jsonl"
    done = set()
    if shard.exists():          # resume: a shard that died mid-run keeps what it finished
        done = {json.loads(l)["task"] for l in shard.open()}
        print(f"resuming shard {args.shard_index}: {len(done)} already done")
    fh = shard.open("a")
    n = ok = 0
    for spec in iter_libero_plus(args):
        if spec["name"] in done:
            continue
        backend = make_backend_for(spec["bddl"], args.robot_config)
        try:
            def decide(agentview, wrist, recent, blocked, _lang=spec["language"]):
                allow = ([t for t in controller.allowed_tokens if t not in blocked]
                         if blocked else None)
                resp = controller.decide(task=_lang, gripper_state="", recent_moves=recent,
                                         agentview_image=agentview, wrist_image=wrist,
                                         constrain_to=allow)
                return resp.token

            m = run_episode(backend=backend, init_state=spec["init_states"][0],
                            language=spec["language"],
                            out_dir=out_root / "_last" / f"shard{args.shard_index:02d}",
                            max_tokens=args.max_steps, step_m=args.step_m,
                            rot_step_deg=args.rot_step_deg,
                            gripper_steps=args.gripper_steps, decide=decide,
                            stuck_guard_mm=args.stuck_guard_mm,
                            stuck_repeat=args.stuck_repeat, ignore_done=args.ignore_done)
        finally:
            backend.close()
        n += 1
        ok += m["success"]
        fh.write(json.dumps({"task": spec["name"], "base": spec["base"],
                             "category": spec["category"], "difficulty": spec["difficulty"],
                             "success": m["success"], "reason": m["reason"],
                             "num_steps": m["num_steps"], "instruction": spec["language"],
                             "guard_fired": m["guard_fired"],
                             "done_suppressed": m["done_suppressed"]},
                            ensure_ascii=False) + "\n")
        fh.flush()
        if n % 20 == 0:
            print(f"shard{args.shard_index:02d}: {ok}/{n} ({ok / n * 100:.1f}%)", flush=True)
    fh.close()
    print(f"shard{args.shard_index:02d} done: {ok}/{n}", flush=True)


def run_replay(args, out_root: Path) -> None:
    """Execute recorded tokens: proves the runner reproduces generation-time behaviour."""
    src = Path(args.replay)
    eps = sorted(src.glob("rollout_*")) if (src / "metadata.json").exists() is False else [src]
    eps = [e for e in eps if (e / "actions.jsonl").exists()] or [src]
    print(f"replaying {len(eps)} episode(s) from {src}")
    n_ok = 0
    for ep in eps:
        meta = json.loads((ep / "metadata.json").read_text())
        states = np.load(ep / "states.npz")
        init = states[states.files[0]][0] if states.files else None
        tokens = [json.loads(l)["token"] for l in open(ep / "actions.jsonl")]
        backend = make_backend_for(meta["bddl"], args.robot_config)
        if "fixture_poses" in meta and meta["fixture_poses"]:
            backend.set_fixture_poses(meta["fixture_poses"])
        it = iter(tokens)
        m = run_episode(backend=backend, init_state=init, language=meta["task"],
                        out_dir=out_root / ep.name, max_tokens=len(tokens) + 1,
                        step_m=meta.get("step_m", args.step_m),
                        rot_step_deg=meta.get("rot_step_deg", args.rot_step_deg),
                        gripper_steps=meta.get("gripper_steps", args.gripper_steps),
                        decide=lambda a, w, r, b: next(it, DONE_TOKEN))
        backend.close()
        n_ok += m["success"]
        print(f"  {ep.name}: recorded success={meta.get('success')} -> replayed {m['success']} "
              f"({m['reason']}, {m['num_steps']}/{len(tokens)} tokens)", flush=True)
    print(f"Success rate: {n_ok}/{len(eps)} ({n_ok / max(len(eps), 1) * 100:.1f}%)", flush=True)


def summarize(out_root: Path) -> None:
    rows = []
    for f in sorted(out_root.glob("*/summary.json")):
        d = json.loads(f.read_text())
        rows.append((f.parent.name, d["success"], d["episodes"]))
    ok = sum(r[1] for r in rows)
    n = sum(r[2] for r in rows)
    print("\n=== Summary ===")
    for name, s, e in rows:
        print(f"  {s:3d}/{e:<3d}  {name}")
    # Same last line as the other simulator runners, so scripts/<sim>/eval_batch.sh can read
    # every one of them the same way.
    print(f"Success rate: {ok}/{n} ({ok / max(n, 1) * 100:.1f}%)", flush=True)
    (out_root / "summary.json").write_text(json.dumps(
        {"total_success": ok, "total_episodes": n,
         "per_task": [{"task": a, "success": b, "episodes": c} for a, b, c in rows]},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
