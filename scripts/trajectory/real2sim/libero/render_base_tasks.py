#!/usr/bin/env python3
"""Render the ORIGINAL (unperturbed) LIBERO tasks behind LIBERO-plus, one video per task.

LIBERO-plus adds no tasks: its 10,030 entries are perturbations of the 40 original tasks of
libero_{spatial,object,goal,10}. This replays one real human demonstration per original task
-- the regenerated LIBERO demos, which store the full MuJoCo state at every step -- inside
LIBERO-plus's own base scene (``bddl_files/<suite>/<task>.bddl``), frame by frame from the
recorded states, and checks the task's success predicate on the last one. No policy, no
tokens: this is what each task IS.

Demo files: ``--local-demos <dir>/<suite>_regen/<task>_demo.hdf5`` when present, otherwise
nvidia/LIBERO-Cosmos-Policy on the Hub, read by HTTP range so only demo_0 is transferred
(~25 MB of a 150-480 MB file) and nothing is stored.

    CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 <LIBERO-plus>/.venv/bin/python \
        scripts/trajectory/real2sim/libero/render_base_tasks.py \
        --out rollouts/libero_plus/video/base_tasks
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HUB = "https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy/resolve/main/success_only"


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file over HTTP Range requests, 1 MB block cache."""

    BLOCK = 1 << 20

    def __init__(self, url: str) -> None:
        import requests

        self.s = requests.Session()
        if os.environ.get("HF_TOKEN"):
            self.s.headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0
        self.cache: collections.OrderedDict = collections.OrderedDict()
        self.bytes = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        a = i * self.BLOCK
        b = min(a + self.BLOCK, self.size) - 1
        for k in range(5):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                r.raise_for_status()
                break
            except Exception:
                if k == 4:
                    raise
                time.sleep(2 * (k + 1))
        self.bytes += len(r.content)
        self.cache[i] = r.content
        if len(self.cache) > 64:
            self.cache.popitem(last=False)
        return r.content

    def readinto(self, buf):
        if self.pos >= self.size:
            return 0
        n = min(len(buf), self.size - self.pos)
        got = 0
        while got < n:
            i, off = divmod(self.pos + got, self.BLOCK)
            blk = self._block(i)
            take = min(n - got, len(blk) - off)
            buf[got:got + take] = blk[off:off + take]
            got += take
        self.pos += n
        return n


def load_demo0(suite: str, task: str, local_root: Path) -> tuple[np.ndarray, np.ndarray, str, int]:
    import h5py

    local = local_root / f"{suite}_regen" / f"{task}_demo.hdf5"
    if local.exists():
        src, fh = str(local), open(local, "rb")
    else:
        src = f"{HUB}/{suite}_regen/{task}_demo.hdf5"
        fh = io.BufferedReader(HttpFile(src), buffer_size=1 << 20)
    with h5py.File(fh, "r") as h:
        names = sorted(h["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        d = h["data"][names[0]]
        return np.asarray(d["states"]), np.asarray(d["actions"]), src, len(names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--local-demos", default="/workspace1/zhijun/Embodied/data/LIBERO-Cosmos-Policy/success_only")
    ap.add_argument("--robot-config", default="configs/robot_libero.yaml")
    ap.add_argument("--fps", type=float, default=20.0, help="the demos are recorded at 20 Hz")
    ap.add_argument("--suites", default=",".join(SUITES))
    args = ap.parse_args()

    import imageio.v2 as iio
    from PIL import Image, ImageDraw

    from core.config import camera_contract, load_yaml
    from core.record.images import prepare_view
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from scripts.trajectory.real2sim.preview import _font

    cam = camera_contract(load_yaml(ROOT / args.robot_config))
    bddl_root = Path(get_libero_path("bddl_files"))
    init_root = Path(get_libero_path("init_states"))
    out_root = Path(args.out)
    V, HDR = 360, 76
    f_small, f_big = _font(17), _font(21)

    def fit(text, font, px):
        if font.getlength(text) <= px:
            return text
        while text and font.getlength(text + "...") > px:
            text = text[:-1]
        return text + "..."

    index = []
    for suite in args.suites.split(","):
        tasks = sorted({f.split(".")[0] for f in os.listdir(init_root / suite)})
        for i, task in enumerate(tasks, 1):
            t0 = time.time()
            bddl = bddl_root / suite / f"{task}.bddl"
            plus_lang = " ".join(re.search(r"\(:language\s+(.*?)\)", bddl.read_text(), re.S).group(1).split())
            orig = re.sub(r"^[A-Z_]+SCENE\d+_", "", task).replace("_", " ")
            states, actions, src, n_demos = load_demo0(suite, task, Path(args.local_demos))
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            env.seed(0)
            env.reset()
            dim = int(env.sim.get_state().flatten().shape[0])
            rec = {"suite": suite, "index": i, "task": task, "instruction": orig,
                   "plus_scene_instruction": plus_lang, "demo_source": src, "demos_in_file": n_demos,
                   "demo_steps": int(len(states)), "state_dim_demo": int(states.shape[1]),
                   "state_dim_scene": dim}
            if dim != states.shape[1]:
                rec["error"] = "state dimension mismatch -- scene differs from the demo's"
                env.close()
                index.append(rec)
                print(json.dumps(rec), flush=True)
                continue
            frames, first_success = [], None
            grip_closed = actions[:, 6] > 0  # LIBERO: +1 closes
            for k, st in enumerate(states):
                obs = env.set_init_state(st)
                if first_success is None and env.check_success():
                    first_success = k
                av = prepare_view(obs["agentview_image"], rotation_degrees=cam.get("agentview_rotation_degrees", 0),
                                  flip=cam.get("agentview_flip", "none"))
                wr = prepare_view(obs["robot0_eye_in_hand_image"], rotation_degrees=cam.get("wrist_rotation_degrees", 0),
                                  flip=cam.get("wrist_flip", "none"))
                c = Image.new("RGB", (2 * V, V + HDR), (16, 18, 22))
                c.paste(Image.fromarray(av).resize((V, V), Image.BILINEAR), (0, HDR))
                c.paste(Image.fromarray(wr).resize((V, V), Image.BILINEAR), (V, HDR))
                d = ImageDraw.Draw(c)
                d.text((10, 8), fit(f"{suite} #{i}  {orig}", f_small, 2 * V - 20), fill=(236, 236, 236), font=f_small)
                d.text((10, 38), f"step {k:03d}/{len(states) - 1}", fill=(150, 155, 165), font=f_big)
                d.text((230, 38), "gripper " + ("CLOSED" if grip_closed[k] else "OPEN"),
                       fill=(110, 210, 120) if grip_closed[k] else (236, 236, 236), font=f_big)
                if first_success is not None:
                    d.text((470, 38), "SUCCESS", fill=(110, 210, 120), font=f_big)
                frames.append(np.asarray(c))
            success_end = bool(env.check_success())
            env.close()
            frames.extend([frames[-1]] * int(args.fps))  # hold the end state for a second
            dest = out_root / suite / f"{i:02d}_{task}.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            iio.mimwrite(dest, frames, fps=args.fps, codec="libx264", quality=7, macro_block_size=1)
            rec.update({"video": str(dest.relative_to(out_root)), "frames": len(frames),
                        "seconds": round(len(frames) / args.fps, 1), "success_at_end": success_end,
                        "first_success_step": first_success, "wall_s": round(time.time() - t0, 1)})
            index.append(rec)
            print(json.dumps({k: rec[k] for k in ("suite", "index", "demo_steps", "success_at_end",
                                                  "first_success_step", "seconds", "wall_s")}), flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))
    ok = all(r.get("success_at_end") for r in index)
    print(f"[base_tasks] {sum(bool(r.get('success_at_end')) for r in index)}/{len(index)} replayed to success",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
