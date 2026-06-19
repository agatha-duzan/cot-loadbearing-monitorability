"""Rollout harness: GLM-4.7 (none/low/high) + gpt-oss-120b (low/high),
both intervention arms x in {0,1}, k samples/instance. Resumable JSONL.

Usage:
  python exp/run_rollouts.py --instances 30 --k 4 --workers 12
  python exp/run_rollouts.py --smoke           # 2 instances, 1 eval, quick
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import (MODELS, EVALS, DATA, RESULTS, call_model, build_model_prompt,
                    grade_y, estimate_cost)

OUT = RESULTS / "rollouts.jsonl"
_lock = threading.Lock()

def load_rows(eval_name, limit=None):
    rows = [json.loads(l) for l in (DATA / "intervention" / f"{eval_name}.jsonl").open()]
    return rows[:limit] if limit else rows

def done_keys():
    """Keys already completed WITHOUT error (errored jobs are retried on relaunch)."""
    keys = set()
    if OUT.exists():
        for line in OUT.open():
            try:
                r = json.loads(line)
                if not r.get("error"):
                    keys.add(r["key"])
            except Exception:
                pass
    return keys

def build_jobs(evals, model_keys, instances, k, efforts=None):
    jobs = []
    for ev in evals:
        rows = load_rows(ev, instances)
        for row in rows:
            iid = row["id"]
            for mk in model_keys:
                for effort in MODELS[mk]["efforts"]:
                    if efforts and effort not in efforts:
                        continue
                    for arm in (0, 1):
                        for s in range(k):
                            key = f"{ev}|{mk}|{arm}|{effort}|{iid}|{s}"
                            jobs.append((key, ev, mk, effort, arm, row, s))
    return jobs

def run_one(job):
    key, ev, mk, effort, arm, row, s = job
    prompt = build_model_prompt(ev, row, arm)
    t = time.time()
    try:
        out = call_model(mk, prompt, effort, ev)
    except Exception as e:
        return {"key": key, "eval": ev, "model": mk, "arm": arm, "effort": effort,
                "instance_id": row["id"], "sample": s, "error": str(e)[:300]}
    y, parsed = grade_y(ev, row, out["answer_text"])
    return {
        "key": key, "eval": ev, "model": mk, "arm": arm, "effort": effort,
        "instance_id": row["id"], "sample": s,
        "y": y, "parsed_answer": parsed, "ref_answer": row["answer"],
        "cot": out["cot"], "cot_chars": len(out["cot"]),
        "answer_text": out["answer_text"], "finish_reason": out["finish_reason"],
        "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
        "cost": estimate_cost(out["prompt_tokens"], out["completion_tokens"], out["price_model"]),
        "latency": round(time.time() - t, 1),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--evals", nargs="*", default=EVALS)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--efforts", nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.instances, args.k, args.evals = 2, 1, ["wmdp_sandbagging"]

    RESULTS.mkdir(exist_ok=True)
    jobs = build_jobs(args.evals, args.models, args.instances, args.k, args.efforts)
    done = done_keys()
    todo = [j for j in jobs if j[0] not in done]
    print(f"total jobs={len(jobs)} done={len(done)} todo={len(todo)} workers={args.workers}", flush=True)

    n_done = 0; n_err = 0; spent = 0.0; t0 = time.time()
    with OUT.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            with _lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
            n_done += 1
            if rec.get("error"):
                n_err += 1
            else:
                spent += rec.get("cost") or 0
            if n_done % 25 == 0 or n_done == len(todo):
                el = time.time() - t0
                rate = n_done / el if el else 0
                eta = (len(todo) - n_done) / rate if rate else 0
                print(f"  {n_done}/{len(todo)} done | err={n_err} | ${spent:.2f} | "
                      f"{rate:.1f}/s | ETA {eta/60:.1f}m", flush=True)
    print(f"FINISHED rollouts: {n_done} new, {n_err} errors, est ${spent:.2f}, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
