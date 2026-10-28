#!/usr/bin/env python3
"""One review video per task: every episode of a ``<task>_follow`` (or re-rendered variant)
directory, back to back.

Unlike ``real2sim/preview.py`` (first episode only, written into the dataset), this renders
ALL episodes -- failures included, since they are the ones worth looking at -- and writes
to a separate directory. Each episode opens with a title card (result, failure reason,
demo index, token mix); every frame carries the instruction, the token about to execute
and the gripper state; a successful episode ends on its ``final/`` frame, the image a DONE
sample uses.

    .venv/bin/python scripts/trajectory/real2sim/libero/make_videos.py \
        --root rollouts/libero_plus --out rollouts/libero_plus/video

Every directory under ``--root`` that holds ``rollout_*/actions.jsonl`` becomes
``<out>/<its parent relative to root>/<name without _follow>.mp4``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.trajectory.real2sim.preview import _font, episode_dirs, load_recs  # noqa: E402

VIEW_H = 360                      # each 256x256 view scaled to 360x360
HEADER_H = 72                     # two text lines above the views
WIDTH = 2 * VIEW_H
BG = (16, 18, 22)
FG = (236, 236, 236)
DIM = (150, 155, 165)
OK = (110, 210, 120)
BAD = (235, 95, 85)


def fit(text: str, font, max_px: int) -> str:
    """Truncate ``text`` with an ellipsis so it renders within ``max_px`` pixels."""
    if font.getlength(text) <= max_px:
        return text
    while text and font.getlength(text + "...") > max_px:
        text = text[:-1]
    return text.rstrip() + "..."


def wrap_px(text: str, font, max_px: int) -> list[str]:
    """Greedy word wrap by rendered width; words longer than a line (LIBERO-plus variant
    names are one underscore-joined word) are broken by character."""
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if font.getlength(cand) <= max_px:
            cur = cand
            continue
        if cur:
            lines.append(cur)
        cur = ""
        while font.getlength(word) > max_px:
            k = len(word)
            while k > 1 and font.getlength(word[:k]) > max_px:
                k -= 1
            lines.append(word[:k])
            word = word[k:]
        cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def token_color(tok: str) -> tuple:
    if tok.startswith("MV_"):
        return (120, 190, 255)
    if tok.startswith("RT_"):
        return (255, 175, 80)
    if tok == "GRASP":
        return OK
    if tok == "RELEASE":
        return (240, 215, 90)
    return FG


def canvas_with_views(ep: Path, av_rel: str, wr_rel: str) -> Image.Image:
    c = Image.new("RGB", (WIDTH, HEADER_H + VIEW_H), BG)
    for k, rel in enumerate((av_rel, wr_rel)):
        im = Image.open(ep / rel).convert("RGB").resize((VIEW_H, VIEW_H), Image.BILINEAR)
        c.paste(im, (k * VIEW_H, HEADER_H))
    return c


def title_card(lines: list[tuple[str, tuple, int]]) -> Image.Image:
    c = Image.new("RGB", (WIDTH, HEADER_H + VIEW_H), BG)
    d = ImageDraw.Draw(c)
    y = 40
    for text, color, size in lines:
        font = _font(size)
        for part in wrap_px(text, font, WIDTH - 56):
            d.text((28, y), part, fill=color, font=font)
            y += int(size * 1.35)
        y += 6
    return c


def episode_frames(ep: Path, idx: int, total: int, fps: float) -> list[np.ndarray]:
    meta = json.loads((ep / "metadata.json").read_text())
    recs = load_recs(ep)
    ok = bool(meta.get("success"))
    result = "SUCCESS" if ok else f"FAILED ({meta.get('reason') or 'unknown'})"
    counts = meta.get("token_counts", {})
    n_mv = sum(v for k, v in counts.items() if k.startswith("MV_"))
    n_rt = sum(v for k, v in counts.items() if k.startswith("RT_"))
    card = [
        (f"Episode {idx + 1}/{total}   ({ep.name}, demo {meta.get('demo_index')})", DIM, 22),
        (result, OK if ok else BAD, 34),
        (meta.get("task", ""), FG, 22),
        (f"{len(recs)} tokens:  {n_mv} move · {n_rt} rotate · "
         f"{counts.get('GRASP', 0)} grasp · {counts.get('RELEASE', 0)} release", DIM, 20),
    ]
    if meta.get("perturbation_category"):
        card.append((f"Perturbation: {meta['perturbation_category']} "
                     f"(difficulty {meta.get('perturbation_difficulty')}) -- "
                     f"{meta.get('perturbation_task', '')}", (255, 175, 80), 18))
    out = [np.asarray(title_card(card))] * max(1, int(round(fps * 1.5)))

    f_small, f_big = _font(17), _font(22)
    tag = "OK" if ok else "FAIL"
    top = fit(f"ep {idx + 1}/{total} [{tag}]  {meta.get('task', '')}", f_small, WIDTH - 20)
    for r in recs:
        c = canvas_with_views(ep, r["agentview"], r["wrist"])
        d = ImageDraw.Draw(c)
        d.text((10, 8), top, fill=DIM, font=f_small)
        d.text((10, 36), f"#{r['step']:03d}", fill=FG, font=f_big)
        d.text((80, 36), r["token"], fill=token_color(r["token"]), font=f_big)
        grip = "CLOSED" if r["gripper_closed"] else "OPEN"
        d.text((300, 36), f"gripper {grip}  w={r['gripper_width'] * 1000:.1f}mm",
               fill=FG, font=f_big)
        out.append(np.asarray(c))

    fin = meta.get("final_frame")
    if fin and (ep / fin["agentview"]).exists():
        c = canvas_with_views(ep, fin["agentview"], fin["wrist"])
        d = ImageDraw.Draw(c)
        d.text((10, 8), top, fill=DIM, font=f_small)
        d.text((10, 36), "DONE", fill=OK, font=f_big)
        d.text((100, 36), "(final frame, after success)", fill=DIM, font=f_big)
        out.extend([np.asarray(c)] * max(1, int(round(fps * 1.5))))
    else:
        out.extend([out[-1]] * max(1, int(round(fps * 1.0))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()

    import imageio.v2 as iio

    root, out_root = Path(args.root).resolve(), Path(args.out).resolve()
    task_dirs = sorted({p.parent.parent for p in root.rglob("rollout_*/actions.jsonl")
                        if out_root not in p.parents})
    report = []
    for td in task_dirs:
        eps = episode_dirs(td)
        frames: list[np.ndarray] = []
        for i, ep in enumerate(eps):
            frames.extend(episode_frames(ep, i, len(eps), args.fps))
        name = td.name[:-len("_follow")] if td.name.endswith("_follow") else td.name
        dest = out_root / td.parent.relative_to(root) / f"{name}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        iio.mimwrite(dest, frames, fps=args.fps, codec="libx264", quality=7,
                     macro_block_size=1)
        n_ok = sum(bool(json.loads((e / "metadata.json").read_text()).get("success")) for e in eps)
        report.append({"video": str(dest.relative_to(out_root)), "episodes": len(eps),
                       "success": n_ok, "frames": len(frames),
                       "seconds": round(len(frames) / args.fps, 1),
                       "mb": round(dest.stat().st_size / 1e6, 2)})
        print(json.dumps(report[-1]), flush=True)
    (out_root / "index.json").write_text(json.dumps(report, indent=1))
    return 0 if report else 1


if __name__ == "__main__":
    raise SystemExit(main())
