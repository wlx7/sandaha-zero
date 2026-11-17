"""Command line entry points.

    python -m sandaha.train.cli train --name myrun --preset medium --hours 2
    python -m sandaha.train.cli train --config runs/myrun/config.json
    python -m sandaha.train.cli eval models/a.npz rule --deals 500
    python -m sandaha.train.cli eval models/a.npz models/b.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _agent(spec, seed=1):
    from ..ai import AIAgent, Model
    from ..bots import RandomBot, RuleBot
    if spec == "rule":
        return RuleBot(seed=seed)
    if spec == "random":
        return RandomBot(seed=seed)
    return AIAgent(Model.load(spec))


def main(argv=None):
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="sandaha-train")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config")
    t.add_argument("--name")
    t.add_argument("--preset")
    t.add_argument("--actors", type=int)
    t.add_argument("--hours", type=float)
    t.add_argument("--games", type=int)
    t.add_argument("--lr", type=float)
    t.add_argument("--lr-end", type=float)
    t.add_argument("--lr-schedule", choices=["constant", "linear", "cosine"])
    t.add_argument("--batch", type=int)
    t.add_argument("--init")
    t.add_argument("--baseline")
    t.add_argument("--eval-minutes", type=float)
    t.add_argument("--eval-deals", type=int)
    t.add_argument("--device")
    t.add_argument("--learner-threads", type=int)
    t.add_argument("--games-per-actor", type=int)
    t.add_argument("--replay", type=float)
    t.add_argument("--runs-dir", default="runs")
    e = sub.add_parser("eval")
    e.add_argument("a")
    e.add_argument("b")
    e.add_argument("--deals", type=int, default=300)
    e.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args(argv)

    if args.cmd == "train":
        from .config import TrainConfig
        from .dmc import train
        cfg = TrainConfig.load(args.config) if args.config else TrainConfig()
        for k, attr in (("name", "name"), ("preset", "preset"), ("actors", "num_actors"),
                        ("hours", "max_hours"), ("games", "max_games"), ("lr", "lr"), ("lr_end", "lr_end"), ("lr_schedule", "lr_schedule"),
                        ("batch", "batch_size"), ("init", "init_from"), ("baseline", "baseline"),
                        ("eval_minutes", "eval_every_minutes"), ("eval_deals", "eval_deals"),
                        ("device", "device"), ("learner_threads", "learner_threads"), ("games_per_actor", "games_per_actor"),
                        ("replay", "replay_ratio")):
            v = getattr(args, k)
            if v is not None:
                setattr(cfg, attr, v)
        if not args.config and not args.name:
            cfg.name = time.strftime("run-%Y%m%d-%H%M%S")
        run_dir = os.path.dirname(os.path.abspath(args.config)) if args.config else \
            os.path.join(args.runs_dir, cfg.name)
        os.makedirs(run_dir, exist_ok=True)
        train(cfg, run_dir)
    elif args.cmd == "eval":
        from ..arena import duel
        t0 = time.time()
        r = duel(_agent(args.a), _agent(args.b, 2), args.deals, seed=args.seed)
        r["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
