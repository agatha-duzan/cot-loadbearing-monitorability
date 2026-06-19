"""Hero figure for the load-bearing monitorability sprint.

  Fig 1 — drop dot-plot: drop = effect@lowest - effect@high per cell, 95% CIs.
      Finding: no cell has a significantly negative drop — the "robust" region,
      where the CoT would be load-bearing for the perturbation, is empty.
      A callout on wmdp·GLM ties the result to monitorability (g-mean²=0.85).

Design: one color per MODEL; filled = both legs solid, hollow = one under-powered
leg, faint = daft (truncation-confounded). n on each row = high-effort g-mean²
sample (matches REPORT §6 table).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, EVALS, RESULTS
from analyze_pooled import load, pooled_gmean2, boot_ci, NO_COT

FIG = RESULTS / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif", "font.weight": "normal",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42, "font.size": 10,
})
MODEL_COLOR = {"glm-4.7": "#4E79A7", "gpt-oss-120b": "#E15759"}
BOTH_SOLID = {("wmdp_sandbagging", "glm-4.7"), ("gpqa", "gpt-oss-120b")}

def mk_short(mk):
    return "GLM" if mk == "glm-4.7" else "oss"

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    print("saved", FIG / f"{name}.png")

def _arm(rows, x, e):
    return [r["y"] for r in rows if r["arm"] == x and r["effort"] == e and r.get("y") is not None]

def eff_se(rows, e):
    a0, a1 = _arm(rows, 0, e), _arm(rows, 1, e)
    if not a0 or not a1:
        return None
    p0, p1, n0, n1 = np.mean(a0), np.mean(a1), len(a0), len(a1)
    return p1 - p0, np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)

def compute():
    R, Z = load()
    by = defaultdict(list)
    for r in R:
        by[(r["eval"], r["model"])].append(r)
    cells = {}
    for ev in EVALS:
        for mk in MODELS:
            rows = by.get((ev, mk), [])
            if not rows:
                continue
            lo, hi = eff_se(rows, NO_COT[mk]), eff_se(rows, "high")
            drop = drop_ci = None
            if lo and hi:
                d = lo[0] - hi[0]; se = np.sqrt(lo[1] ** 2 + hi[1] ** 2)
                drop, drop_ci = d, (d - 1.96 * se, d + 1.96 * se)
            gm = [{"x": r["arm"], "y": r["y"], "z": Z[r["key"]], "instance_id": r["instance_id"]}
                  for r in rows if r["effort"] == "high" and r.get("y") is not None and r["key"] in Z]
            m = pooled_gmean2(gm) if gm else None
            cells[(ev, mk)] = {"drop": drop, "drop_ci": drop_ci,
                               "gmean2": (m["gmean2"] if (m and m.get("eligible")) else None),
                               "n_hi": len(gm)}
    return cells

def fig_drop(cells):
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    clean = [("wmdp_sandbagging", "glm-4.7"), ("gpqa", "gpt-oss-120b"),
             ("wmdp_sandbagging", "gpt-oss-120b"), ("gpqa", "glm-4.7")]
    daft = [("daft", "gpt-oss-120b"), ("daft", "glm-4.7")]
    rows = [(c, i + 2.2) for i, c in enumerate(reversed(clean))] + \
           [(c, i * 0.8) for i, c in enumerate(reversed(daft))]
    ax.axvspan(-0.45, 0, color="#e3ebf2", alpha=0.7, zorder=0)
    # short in-region label, placed in the empty shaded band between two rows
    ax.text(-0.30, 3.7, "robust zone\n(empty)", fontsize=8.5, color="#2a8", style="italic",
            ha="center", va="center", fontweight="bold")
    ax.axvline(0, color="#555", lw=0.9, zorder=1)
    yticks, ylabels = [], []
    for (ev, mk), y in rows:
        d = cells[(ev, mk)]
        c = MODEL_COLOR[mk]; conf = ev == "daft"; solid = (ev, mk) in BOTH_SOLID
        dr, ci = d["drop"], d["drop_ci"]
        ax.errorbar(dr, y, xerr=[[dr - ci[0]], [ci[1] - dr]], fmt="o", ms=7 if conf else 8,
                    color=c, ecolor=c, elinewidth=1.1, capsize=3, alpha=0.45 if conf else 1.0,
                    zorder=3, markerfacecolor=("none" if (conf or not solid) else c),
                    markeredgecolor=c, markeredgewidth=1.5)
        yticks.append(y)
        ylabels.append(f"{ev.split('_')[0]}·{mk_short(mk)}  (n={d['n_hi']})" + ("  ⚠confounded" if conf else ""))
    ax.axhline(1.7, color="#ccc", lw=0.6, ls=":")
    # callout: tie the result to monitorability via the sharpest cell
    ax.annotate("g-mean²=0.85 here →\nmonitorable, yet the\nperturbation needs no CoT",
                xy=(0.08, 5.2), xytext=(0.27, 5.05), fontsize=7.5, color="#333", va="center",
                arrowprops=dict(arrowstyle="->", color="#777", lw=0.8))
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_ylim(-0.6, 6.0); ax.set_xlim(-0.45, 0.66)
    ax.set_xlabel("drop  =  effect@lowest − effect@high     (perturbation effect lost when reasoning added)")
    ax.set_title("No cell shows the perturbation effect growing with reasoning\n"
                 "— the 'robust' region (drop < 0) is empty", fontsize=10.5)
    leg = [Line2D([], [], marker="o", color=MODEL_COLOR["glm-4.7"], ls="", ms=8, label="GLM-4.7"),
           Line2D([], [], marker="o", color=MODEL_COLOR["gpt-oss-120b"], ls="", ms=8, label="gpt-oss-120b"),
           Line2D([], [], marker="o", color="#444", ls="", ms=8, markerfacecolor="#444", label="both legs solid"),
           Line2D([], [], marker="o", color="#444", ls="", ms=8, markerfacecolor="none", label="one leg / confounded")]
    ax.legend(handles=leg, frameon=False, fontsize=7.5, loc="center right")
    fig.text(0.5, -0.07,
             "All point estimates ≥ 0; no clean cell's drop is significantly different from 0 — we claim the robust zone is empty, "
             "NOT that\ndrops are individually positive (§8.5). daft's CI appears to exclude 0, but truncation makes its high-effort arm "
             "a different,\neasier population — that manufactures the gap (§8.3); daft gestures at the decoupling but cannot establish it.",
             fontsize=6.7, color="#555", ha="center")
    save(fig, "fig1_drop")

def main():
    cells = compute()
    fig_drop(cells)
    print("\ncells:")
    for (ev, mk), d in cells.items():
        dr = f"{d['drop']:+.2f}[{d['drop_ci'][0]:+.2f},{d['drop_ci'][1]:+.2f}]" if d["drop"] is not None else "n/a"
        g = "undef" if d["gmean2"] is None else f"{d['gmean2']:.3f}"
        print(f"  {ev:16s} {mk:13s} drop={dr}  gmean2={g}  n_hi={d['n_hi']}")

if __name__ == "__main__":
    main()
