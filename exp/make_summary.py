"""Write results/cells_summary.csv — per-cell aggregate numbers, NO raw content.

This is the committable results artifact: every number behind REPORT.md §6 and
the figure, with zero eval questions or model CoT. Regenerable from rollouts.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, EVALS, RESULTS
from analyze_pooled import load, pooled_gmean2, boot_ci, NO_COT

def arm(rows, x, e):
    return [r["y"] for r in rows if r["arm"] == x and r["effort"] == e and r.get("y") is not None]

def eff_ci(rows, e):
    a0, a1 = arm(rows, 0, e), arm(rows, 1, e)
    if not a0 or not a1:
        return None
    p0, p1, n0, n1 = np.mean(a0), np.mean(a1), len(a0), len(a1)
    eff = p1 - p0
    se = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    return eff, eff - 1.96 * se, eff + 1.96 * se, n0 + n1

def main():
    R, Z = load()
    by = defaultdict(list)
    for r in R:
        by[(r["eval"], r["model"])].append(r)
    rows_out = []
    for ev in EVALS:
        for mk in MODELS:
            rows = by.get((ev, mk), [])
            if not rows:
                continue
            lo = eff_ci(rows, NO_COT[mk]); hi = eff_ci(rows, "high")
            drop = drop_lo = drop_hi = None
            if lo and hi:
                d = lo[0] - hi[0]
                se = np.sqrt(((lo[2] - lo[1]) / (2 * 1.96)) ** 2 + ((hi[2] - hi[1]) / (2 * 1.96)) ** 2)
                drop, drop_lo, drop_hi = d, d - 1.96 * se, d + 1.96 * se
            gm = [{"x": r["arm"], "y": r["y"], "z": Z[r["key"]], "instance_id": r["instance_id"]}
                  for r in rows if r["effort"] == "high" and r.get("y") is not None and r["key"] in Z]
            m = pooled_gmean2(gm) if gm else None
            g2 = m["gmean2"] if (m and m.get("eligible")) else None
            ci = boot_ci(gm, "instance_id") if (m and m.get("eligible")) else None
            f = lambda v: "" if v is None else round(float(v), 3)
            rows_out.append({
                "eval": ev, "model": mk, "lowest_effort": NO_COT[mk],
                "effect_lowest": f(lo[0] if lo else None),
                "effect_lowest_ci_lo": f(lo[1] if lo else None),
                "effect_lowest_ci_hi": f(lo[2] if lo else None),
                "n_lowest": (lo[3] if lo else ""),
                "effect_high": f(hi[0] if hi else None),
                "drop": f(drop), "drop_ci_lo": f(drop_lo), "drop_ci_hi": f(drop_hi),
                "gmean2_high": f(g2),
                "gmean2_ci_lo": f(ci[0] if ci else None), "gmean2_ci_hi": f(ci[1] if ci else None),
                "n_high_gmean2": len(gm),
            })
    out = RESULTS / "cells_summary.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print(f"wrote {out} ({len(rows_out)} cells)")

if __name__ == "__main__":
    main()
