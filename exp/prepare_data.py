"""Regenerate the vendored intervention subsample (data/intervention/*.jsonl).

The raw eval data is NOT committed (benchmark contamination / redistribution
hygiene — see README). This script reproduces the *exact* instance set used in
the study from a local clone of OpenAI's monitorability-evals repo.

Usage:
  python exp/prepare_data.py --src /path/to/monitorability-evals
  # then unzip its environments.zip if needed; we read environments/intervention/*.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "intervention"
EVALS = ["wmdp_sandbagging", "gpqa", "daft"]
# final instance counts (initial 30, expanded for the high-effort CI work)
TARGETS = {"wmdp_sandbagging": 100, "gpqa": 100, "daft": 60}

def load_source(src: Path, ev: str):
    direct = src / "environments" / "intervention" / f"{ev}.jsonl"
    if direct.exists():
        return [json.loads(l) for l in direct.open()]
    zpath = src / "environments.zip"
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            with z.open(f"environments/intervention/{ev}.jsonl") as f:
                return [json.loads(l) for l in f.read().decode().splitlines()]
    raise FileNotFoundError(f"could not find {ev}.jsonl under {src} (extracted or in environments.zip)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to a clone of monitorability-evals")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for ev in EVALS:
        rows = load_source(Path(args.src), ev)
        # step 1: deterministic initial 30 (matches the original vendoring)
        base = random.Random(20260618).sample(rows, 30)
        have = {r["id"] for r in base}
        # step 2: deterministic expansion to TARGET from the remaining pool
        pool = [r for r in rows if r["id"] not in have]
        add = random.Random(7).sample(pool, TARGETS[ev] - 30)
        full = base + add  # base first => stable ordering
        with (OUT / f"{ev}.jsonl").open("w") as f:
            for r in full:
                f.write(json.dumps(r) + "\n")
        print(f"{ev}: wrote {len(full)} instances -> {OUT / f'{ev}.jsonl'}")

if __name__ == "__main__":
    main()
