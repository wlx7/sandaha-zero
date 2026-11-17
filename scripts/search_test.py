"""Measure what search adds: same model with vs. without rollout search.

Prints a running tally after every finished deal (duplicate format).

    python scripts/search_test.py models/sandaha-best.npz --deals 60 --rollouts 24
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OMP_NUM_THREADS", "1")

ARGS = None


def init(args):
    global ARGS
    ARGS = args


def one_deal(i):
    from sandaha.ai import AIAgent, Model
    from sandaha.ai.search import SearchAgent
    from sandaha.arena import play_game, random_deal
    from sandaha.bots import RuleBot
    m = Model.load(ARGS.model)
    a = SearchAgent(m, samples=ARGS.samples, topk=ARGS.topk, max_rollouts=ARGS.rollouts, seed=i)
    b = RuleBot(seed=i) if ARGS.opponent == "rule" else AIAgent(m)
    rng = random.Random(777 + i)
    hands, bottom = random_deal(rng)
    fb = rng.randrange(4)
    total = 0
    t0 = time.time()
    for swap in (0, 1):
        seats = [a, b, a, b] if swap == 0 else [b, a, b, a]
        mine = (0, 2) if swap == 0 else (1, 3)
        g = play_game(seats, None, hands, bottom, fb, seed=rng.randrange(1 << 30))
        total += sum(g.result["scores"][s] for s in mine)
    return i, total, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--opponent", default="plain", choices=["plain", "rule"])
    ap.add_argument("--deals", type=int, default=60)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--rollouts", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()
    scores = []
    with Pool(args.workers, initializer=init, initargs=(args,)) as pool:
        for i, total, sec in pool.imap_unordered(one_deal, range(args.deals)):
            scores.append(total)
            n = len(scores)
            mean = sum(scores) / n
            var = sum((x - mean) ** 2 for x in scores) / max(1, n - 1)
            se = (var / n) ** 0.5
            wins = sum(1 for x in scores if x > 0) + 0.5 * sum(1 for x in scores if x == 0)
            print(f"[{n}/{args.deals}] deal {i}: {total:+d}  | search side: mean {mean:+.3f} ± {1.96 * se:.3f} "
                  f"per deal (2 games), deal win {wins / n * 100:.1f}%  | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
