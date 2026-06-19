"""Monitor pass: run the (weak) monitor on the raw CoT of rollouts to produce z.

z is needed by the g-mean metric for BOTH arms (control rows are the monitor's
false-positive test). We run the monitor on every successful rollout at the
HIGH effort level by default (the level at which g-mean2 is reported), plus LOW
if requested (for the mild-CoT-pressure panel). Resumable JSONL.

Usage:
  python exp/run_monitor.py --efforts high
  python exp/run_monitor.py --efforts high low
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import (DATA, RESULTS, EVALS, call_monitor, build_monitor_prompt,
                    parse_monitor, estimate_cost)

ROLL = RESULTS / "rollouts.jsonl"
OUT = RESULTS / "monitor.jsonl"
_lock = threading.Lock()

def load_rows_by_eval():
    out = {}
    for ev in EVALS:
        out[ev] = {r["id"]: r for r in
                   (json.loads(l) for l in (DATA / "intervention" / f"{ev}.jsonl").open())}
    return out

def done_keys():
    keys = set()
    if OUT.exists():
        for line in OUT.open():
            try:
                keys.add(json.loads(line)["key"])
            except Exception:
                pass
    return keys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--efforts", nargs="*", default=["high"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    rows_by_eval = load_rows_by_eval()
    # candidate rollouts: parsed (no error), effort in target set
    jobs = []
    for line in ROLL.open():
        r = json.loads(line)
        if r.get("error") or r["effort"] not in args.efforts:
            continue
        if r.get("y") is None:  # metric needs y AND z; skip unparsed/truncated rows
            continue
        # monitor key independent of whether y parsed; we still need z on all rows
        row = rows_by_eval[r["eval"]].get(r["instance_id"])
        if row is None:
            continue
        jobs.append((r["key"], r["eval"], row, r["cot"], r["answer_text"]))

    done = done_keys()
    todo = [j for j in jobs if j[0] not in done]
    print(f"monitor jobs={len(jobs)} done={len(done)} todo={len(todo)}", flush=True)

    def run_one(job):
        key, ev, row, cot, ans = job
        prompt = build_monitor_prompt(ev, row, cot, ans)
        try:
            out = call_monitor(prompt)
        except Exception as e:
            return {"key": key, "error": str(e)[:300]}
        z, letter = parse_monitor(out["answer_text"])
        return {"key": key, "z": z, "monitor_letter": letter,
                "cost": estimate_cost(out["prompt_tokens"], out["completion_tokens"], out["price_model"])}

    n = 0; spent = 0.0; n_err = 0; t0 = time.time()
    with OUT.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(run_one, j) for j in todo]):
            rec = fut.result()
            with _lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1
            if rec.get("error"):
                n_err += 1
            else:
                spent += rec.get("cost") or 0
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0; rate = n / el if el else 0
                print(f"  {n}/{len(todo)} | err={n_err} | ${spent:.2f} | ETA {(len(todo)-n)/rate/60 if rate else 0:.1f}m", flush=True)
    print(f"FINISHED monitor: {n} new, {n_err} errors, est ${spent:.2f}, {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
