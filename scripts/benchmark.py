"""Benchmark a model against the rule AI, random play and its own older versions.

    python scripts/benchmark.py runs/main/final.npz --deals 500 --out docs/results.json
    python scripts/benchmark.py runs/main/final.npz --vs runs/other/final.npz ...

Runs the duplicate-deal matches in parallel processes and prints a Markdown table.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _agent(spec, seed):
    from sandaha.ai import AIAgent, Model
    from sandaha.bots import RandomBot, RuleBot
    if spec == "rule":
        return RuleBot(seed=seed)
    if spec == "random":
        return RandomBot(seed=seed)
    return AIAgent(Model.load(spec))


def _job(args):
    a, b, deals, seed = args
    from sandaha.arena import duel
    return duel(_agent(a, 1), _agent(b, 2), deals, seed=seed)


def match(a, b, deals, workers):
    """Split deals over workers (different seeds), then merge the statistics."""
    per = max(1, deals // workers)
    jobs = [(a, b, per, 1000 + i) for i in range(workers)]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_job, jobs))
    n = sum(p["games"] for p in parts)
    out = {"games": n}
    for k in ("win_rate", "avg_score", "banker_rate"):
        out[k] = sum(p[k] * p["games"] for p in parts) / n
    bg = sum(p["banker_rate"] * p["games"] for p in parts)
    out["banker_win_rate"] = sum(p["banker_win_rate"] * p["banker_rate"] * p["games"] for p in parts) / max(1, bg)
    # standard error of avg_score is not available per game here; report win-rate CI instead
    p = out["win_rate"]
    out["win_ci95"] = 1.96 * (p * (1 - p) / n) ** 0.5
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--vs", nargs="*", default=[], help="extra opponents (model paths)")
    ap.add_argument("--history", type=int, default=4, help="older checkpoints of the same run to include")
    ap.add_argument("--out")
    args = ap.parse_args()

    opponents = [("规则AI rule", "rule"), ("随机 random", "random")]
    run_dir = os.path.dirname(os.path.abspath(args.model))
    ck = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "*.npz")))
    if ck and args.history:
        step = max(1, len(ck) // args.history)
        picks = ck[:-1][::step][:args.history]
        for p in picks:
            games = int(os.path.basename(p).split("_")[1].split(".")[0])
            opponents.append((f"旧版本 @{games / 1000:.0f}k 局", p))
    for v in args.vs:
        opponents.append((os.path.basename(os.path.dirname(v)) + "/" + os.path.basename(v), v))

    rows = []
    for label, spec in opponents:
        t0 = time.time()
        r = match(args.model, spec, args.deals, args.workers)
        r["opponent"] = label
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"{label:24s} win {r['win_rate'] * 100:5.1f}% ±{r['win_ci95'] * 100:.1f}  "
              f"score {r['avg_score']:+.3f}  banker {r['banker_rate'] * 100:4.1f}% "
              f"(win {r['banker_win_rate'] * 100:4.1f}%)  [{r['games']} games, {r['seconds']}s]", flush=True)
    print()
    print("| 对手 Opponent | 局数 Games | 胜率 Win rate | 每局净得分 Net score/game | 做庄率 Banker rate | 做庄胜率 Banker win |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['opponent']} | {r['games']} | {r['win_rate'] * 100:.1f}% ± {r['win_ci95'] * 100:.1f} | "
              f"{r['avg_score']:+.3f} | {r['banker_rate'] * 100:.1f}% | {r['banker_win_rate'] * 100:.1f}% |")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "deals": args.deals, "rows": rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
