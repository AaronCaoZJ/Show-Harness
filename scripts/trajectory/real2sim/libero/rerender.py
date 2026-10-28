#!/usr/bin/env python3
"""Re-render follower rollouts under LIBERO-plus's visual perturbations -- no re-execution.

A visual perturbation (camera viewpoint, lighting, background texture, sensor noise) leaves
the model and its state vector untouched (measured: 92-dim in the base task and in every such
variant, restored with zero error), so the SAME token sequence is valid in the variant. Each
frame is re-rendered from the sim state the follower logged behind it (``states.npz``); the
tokens, ee_poses and gripper states are copied unchanged. Language variants reuse the images
and swap only the instruction (the variant task's ``language``).

Not applicable -- they need re-execution, not re-rendering:
  * Robot Initial States: the point of the variant IS a different start, which a replayed
    state vector would overwrite.
  * Objects Layout: extra objects change the state vector (118-dim vs 92).

Sensor noise is applied here by hand: LIBERO-plus adds it inside ``step()`` / ``reset()``
only, so a frame regenerated through ``set_init_state`` would silently come out clean.

    <LIBERO-plus>/.venv/bin/python scripts/trajectory/real2sim/libero/rerender.py \
        --rollouts <out>/<task>_follow --suite libero_spatial --out <dir> --per-category 1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

RENDER_CATEGORIES = ("Camera Viewpoints", "Light Conditions", "Background Textures", "Sensor Noise")
TEXT_CATEGORIES = ("Language Instructions",)


def libero_plus_noise(raw_agentview: np.ndarray, noise: int) -> np.ndarray:
    """LIBERO-plus's own sensor-noise dispatch (envs/env_wrapper.py ``step``), on the RAW image."""
    from PIL import Image

    from libero.libero.envs.env_wrapper import (fog, gaussian_blur, glass_blur,
                                                motion_blur, zoom_blur)
    if not noise:
        return raw_agentview
    img = Image.fromarray(raw_agentview.astype(np.uint8))
    fn, sev = [(motion_blur, noise), (gaussian_blur, noise - 10), (zoom_blur, noise - 20),
               (fog, noise - 30), (glass_blur, noise - 40)][min((noise - 1) // 10, 4)]
    return np.asarray(fn(img, severity=sev)).astype(np.uint8)


def bddl_language(bddl: str) -> str:
    m = re.search(r"\(:language\s+(.*?)\)", Path(bddl).read_text(), re.S)
    return " ".join(m.group(1).split()) if m else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", required=True, help="<task>_follow dir with rollout_*/states.npz")
    ap.add_argument("--suite", required=True, help="libero_spatial | libero_object | libero_goal | libero_10")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-category", type=int, default=1)
    ap.add_argument("--variants", default="",
                    help="comma-separated EXACT variant names (as task_classification.json "
                         "spells them), instead of sampling --per-category per category")
    ap.add_argument("--categories", default=",".join(RENDER_CATEGORIES + TEXT_CATEGORIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robot-config", default="configs/robot_libero.yaml")
    ap.add_argument("--verify-base", action="store_true",
                    help="re-render in the UNPERTURBED task and compare with the stored frames")
    args = ap.parse_args()

    from PIL import Image

    from core.config import camera_contract, load_yaml
    from libero.libero import benchmark, get_libero_path
    from scripts.trajectory.real2sim.atomic_tokenizer import prepared_pair
    from scripts.trajectory.real2sim.backends import make_backend

    contract = camera_contract(load_yaml(ROOT / args.robot_config))
    # Only successful episodes are training data; a --keep-failures run leaves the rest in place.
    rollouts = [r for r in sorted(Path(args.rollouts).glob("rollout_*"))
                if json.loads((r / "metadata.json").read_text()).get("success")]
    if not rollouts:
        raise SystemExit(f"no successful rollouts under {args.rollouts}")
    meta0 = json.loads((rollouts[0] / "metadata.json").read_text())
    task_key = meta0["task_key"]

    def render_all(backend, states: np.ndarray, noise: int,
                   fixtures: dict) -> list[tuple[np.ndarray, np.ndarray]]:
        # Fixture placement is not in the state vector -- restore it first (see
        # LiberoBackend.fixture_poses), or the cabinet/stove render a few mm off.
        backend.set_fixture_poses(fixtures)
        out = []
        for st in states:
            obs = backend.env.set_init_state(st)
            av = libero_plus_noise(obs["agentview_image"], noise)
            out.append(prepared_pair(backend, av, obs["robot0_eye_in_hand_image"]))
        return out

    if args.verify_base:
        backend = make_backend("libero", bddl_file=meta0["bddl"], **contract)
        worst = 0
        for r in rollouts:
            recs = [json.loads(l) for l in (r / "actions.jsonl").open()]
            states = np.load(r / "states.npz")["states"]
            frames = render_all(backend, states[:len(recs)], 0,
                                json.loads((r / "metadata.json").read_text()).get("fixture_poses"))
            for rec, (av, wr) in zip(recs, frames):
                a = np.asarray(Image.open(r / rec["agentview"])).astype(int)
                w = np.asarray(Image.open(r / rec["wrist"])).astype(int)
                worst = max(worst, int(np.abs(a - av).max()), int(np.abs(w - wr).max()))
        backend.close()
        print(f"[verify-base] max pixel difference vs stored frames: {worst}", flush=True)
        return 0 if worst == 0 else 1

    cls = json.load(open(os.path.join(get_libero_path("benchmark_root"), "benchmark/task_classification.json")))[args.suite]
    suite = benchmark.get_benchmark_dict()[args.suite]()
    names = suite.get_task_names()
    wanted = [c.strip() for c in args.categories.split(",") if c.strip()]
    variants = []
    if args.variants:
        want = [n.strip() for n in args.variants.split(",") if n.strip()]
        by_name = {it["name"]: it for it in cls}
        missing = [n for n in want if n not in by_name]
        if missing:
            raise SystemExit(f"[error] not in task_classification.json: {missing}")
        variants = [(by_name[n]["category"], by_name[n]) for n in want]
    else:
        rng = random.Random(args.seed)
        for cat in wanted:
            pool = [it for it in cls if it["category"] == cat and it["name"].startswith(task_key + "_")]
            variants += [(cat, it) for it in rng.sample(pool, min(args.per_category, len(pool)))]

    summary = []
    for cat, it in variants:
        task = suite.get_task(names.index(it["name"]))
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        dest = Path(args.out) / f"{task_key}__{it['name'][len(task_key) + 1:]}"
        backend = None if cat in TEXT_CATEGORIES else make_backend("libero", bddl_file=bddl, **contract)
        noise = int(getattr(backend.env, "noise", 0)) if backend else 0
        # Language variants have no bddl of their own; the rewritten instruction is the
        # task's ``language`` (for this category it IS the sentence, not a name + suffix).
        language = task.language if cat in TEXT_CATEGORIES else meta0["task"]
        for r in rollouts:
            d = dest / r.name
            shutil.rmtree(d, ignore_errors=True)
            (d / "agentview").mkdir(parents=True)
            (d / "wrist").mkdir()
            recs = [json.loads(l) for l in (r / "actions.jsonl").open()]
            meta = json.loads((r / "metadata.json").read_text())
            if backend is None:
                for rec in recs:
                    shutil.copy(r / rec["agentview"], d / rec["agentview"])
                    shutil.copy(r / rec["wrist"], d / rec["wrist"])
                if (r / "final").exists():
                    shutil.copytree(r / "final", d / "final")
            else:
                states = np.load(r / "states.npz")["states"]
                frames = render_all(backend, states, noise, meta.get("fixture_poses"))
                for rec, (av, wr) in zip(recs, frames):
                    Image.fromarray(av).save(d / rec["agentview"])
                    Image.fromarray(wr).save(d / rec["wrist"])
                if meta.get("final_frame"):
                    (d / "final").mkdir()
                    av, wr = frames[meta["final_frame"]["state_index"]]
                    Image.fromarray(av).save(d / "final" / "agentview.png")
                    Image.fromarray(wr).save(d / "final" / "wrist.png")
            shutil.copy(r / "actions.jsonl", d / "actions.jsonl")
            meta.update({"task": language, "perturbation_category": cat, "perturbation_task": it["name"],
                         "perturbation_difficulty": it["difficulty_level"], "perturbation_noise": noise,
                         "rerendered_from": str(r), "method": "closed_loop_follower+rerender"})
            (d / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        if backend is not None:
            backend.close()
        summary.append({"category": cat, "variant": it["name"], "dir": str(dest), "rollouts": len(rollouts)})
        print(json.dumps(summary[-1]), flush=True)
    (Path(args.out) / f"{task_key}__rerender_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
