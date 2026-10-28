#!/usr/bin/env python3
"""Measure what one token physically does on LIBERO, against the lattice contract.

Starts from a recorded demo scene, lifts clear of the table, then executes each of the 12
motion tokens twice (and its opposite twice, to come back), measuring every execution:

* MV_*: travel along the intended axis, off-axis travel, orientation drift
* RT_*: rotation about the intended world axis, rotation about the others, TCP drift,
  and which way the fingertips actually went (the names promise a direction)

Gates (the doc's step 2): translation 20 +/- 1 mm, rotation 10 +/- 1 deg, off-axis < 3 mm and
< 2 deg. Exit code 0 only if every execution passes, so this can guard a backend change.

    <LIBERO-plus>/.venv/bin/python scripts/trajectory/real2sim/libero/probe.py \
        --demos <task>_demo.hdf5 --bddl <task>.bddl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", required=True, help="regenerated LIBERO hdf5 with data/demo_*/states")
    ap.add_argument("--bddl", required=True)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--robot-config", default="configs/robot_libero.yaml")
    ap.add_argument("--out", default="", help="optional: write the per-token table as JSON")
    args = ap.parse_args()

    import h5py

    from core.config import camera_contract, load_yaml
    from scripts.trajectory.real2sim.atomic_tokenizer import (
        MOVE_DIRS, OPPOSITE, ROT_AXES, AtomicExec, rotation_error)
    from scripts.trajectory.real2sim.backends import make_backend

    with h5py.File(args.demos) as h:
        state0 = np.asarray(h["data"][args.demo]["states"][0])
    backend = make_backend("libero", bddl_file=args.bddl,
                           **camera_contract(load_yaml(ROOT / args.robot_config)))
    backend.reset_to_state(state0)
    ex = AtomicExec(backend, max_ctrl_steps=24)
    for _ in range(4):
        ex.move("MV_UP")
    # The approach axis: the hand-frame column pointing (most) down at the start pose.
    r_start = backend.tcp_rotmat()
    k = int(np.argmax(np.abs(r_start[2])))
    sgn = -np.sign(r_start[2, k])  # so that tip = sgn * R[:, k] points down initially

    rows, ok = [], True
    for tok in list(MOVE_DIRS) + list(ROT_AXES):
        for rep in range(2):
            p0, r0 = backend.tcp_pos(), backend.tcp_rotmat()
            tip0 = sgn * r0[:, k]
            if tok in MOVE_DIRS:
                ex.move(tok)
            else:
                ex.rotate(tok)
            p1, r1 = backend.tcp_pos(), backend.tcp_rotmat()
            dp, rv = p1 - p0, rotation_error(r1, r0)
            if tok in MOVE_DIRS:
                d = MOVE_DIRS[tok]; along = float(dp @ d) * 1000
                off = float(np.linalg.norm(dp - (dp @ d) * d)) * 1000
                rot = float(np.degrees(np.linalg.norm(rv)))
                good = abs(along - 20) <= 1 and off < 3 and rot < 2
                rows.append(dict(token=tok, rep=rep, along_mm=round(along, 2), off_mm=round(off, 2), rot_drift_deg=round(rot, 2), ok=good))
            else:
                a = ROT_AXES[tok]; along = float(np.degrees(rv @ a))
                other = float(np.degrees(np.linalg.norm(rv - (rv @ a) * a)))
                drift = float(np.linalg.norm(dp)) * 1000
                tip_dxy = np.round((sgn * r1[:, k] - tip0)[:2], 3).tolist()
                good = abs(along - 10) <= 1 and other < 2 and drift < 3
                rows.append(dict(token=tok, rep=rep, along_deg=round(along, 2), other_deg=round(other, 2), pos_drift_mm=round(drift, 2), tip_dxy=tip_dxy, ok=good))
            ok &= good
            print(json.dumps(rows[-1]), flush=True)
        back = OPPOSITE[tok]
        for _ in range(2):
            (ex.move if back in MOVE_DIRS else ex.rotate)(back)
    print(f"PROBE {'PASS' if ok else 'FAIL'}: {sum(r['ok'] for r in rows)}/{len(rows)} executions within gates", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    backend.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
