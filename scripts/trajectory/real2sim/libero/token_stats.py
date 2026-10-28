#!/usr/bin/env python3
"""Per-token execution statistics of generated rollouts, read from their own ee_pose records.

Frame t is stored BEFORE token t executes, so ee_pose[t+1] - ee_pose[t] is what token t did.
Reports, per token kind: travel along the named axis / off-axis travel / orientation drift
(MV_*), turn about the named axis / other-axis turn / TCP drift (RT_*), and the share of
tokens that were clean (single-axis within 5 mm and 2 deg). ManiSkill Scheme D reached 97%.

    python scripts/trajectory/real2sim/libero/token_stats.py <dir with rollout_*/>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.trajectory.real2sim.atomic_tokenizer import (  # noqa: E402
    MOVE_DIRS, ROT_AXES, rotation_error)


def quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main(root: str) -> None:
    mv, rt, grip, eps = [], [], [], []
    for d in sorted(Path(root).glob("rollout_*")):
        recs = [json.loads(l) for l in (d / "actions.jsonl").open()]
        meta = json.loads((d / "metadata.json").read_text())
        eps.append(meta)
        for a, b in zip(recs, recs[1:]):
            p0, p1 = np.array(a["ee_pose"][:3]), np.array(b["ee_pose"][:3])
            r0, r1 = quat_wxyz_to_mat(a["ee_pose"][3:]), quat_wxyz_to_mat(b["ee_pose"][3:])
            dp, rv = p1 - p0, rotation_error(r1, r0)
            t = a["token"]
            if t in MOVE_DIRS:
                u = MOVE_DIRS[t]
                mv.append((float(dp @ u) * 1000, float(np.linalg.norm(dp - (dp @ u) * u)) * 1000,
                           float(np.degrees(np.linalg.norm(rv)))))
            elif t in ROT_AXES:
                u = ROT_AXES[t]
                rt.append((float(np.degrees(rv @ u)), float(np.degrees(np.linalg.norm(rv - (rv @ u) * u))),
                           float(np.linalg.norm(dp)) * 1000))
            else:
                grip.append(float(np.linalg.norm(dp)) * 1000)
    mv, rt = np.array(mv), np.array(rt)
    q = lambda x: np.round(np.percentile(x, [5, 50, 95]), 2).tolist() if len(x) else []
    out = {
        "episodes": len(eps), "success": sum(bool(m.get("success")) for m in eps),
        "tokens_per_episode": q([m["num_steps"] for m in eps]),
        "move_tokens": len(mv), "move_along_mm_p5_50_95": q(mv[:, 0]) if len(mv) else [],
        "move_offaxis_mm": q(mv[:, 1]) if len(mv) else [], "move_rot_drift_deg": q(mv[:, 2]) if len(mv) else [],
        "move_clean_frac": round(float(np.mean((mv[:, 1] < 5) & (mv[:, 2] < 2) & (np.abs(mv[:, 0] - 20) < 5))), 4) if len(mv) else None,
        "rot_tokens": len(rt), "rot_along_deg": q(rt[:, 0]) if len(rt) else [],
        "rot_other_deg": q(rt[:, 1]) if len(rt) else [], "rot_pos_drift_mm": q(rt[:, 2]) if len(rt) else [],
        "rot_clean_frac": round(float(np.mean((rt[:, 1] < 2) & (rt[:, 2] < 5) & (np.abs(rt[:, 0] - 10) < 2.5))), 4) if len(rt) else None,
        "grip_tokens": len(grip), "grip_pos_drift_mm": q(grip),
        "token_counts": {k: int(v) for k, v in sorted(
            __import__("collections").Counter(t for m in eps for t, n in m["token_counts"].items() for _ in range(n)).items())},
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
