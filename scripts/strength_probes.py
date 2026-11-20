"""Headroom probes: auction shading, no-surrender, and a cheating "sees every hand" player.

    python scripts/strength_probes.py <workers> bid_0.3:4000 no_surrender:4000 oracle_play:800 ...
"""
import json, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMP_NUM_THREADS", "1")
BEST = "models/sandaha-best.npz"

def make(kind, seed):
    import numpy as np
    from sandaha.ai import AIAgent, Model
    from sandaha.ai.search import SearchAgent, make_world, rollout_batch
    from sandaha.engine import PLAY, TRUMP, BURY
    m = Model.load(BEST)
    if kind == "net":
        return AIAgent(m)
    if kind == "deep":      # honest search, much larger budget than the shipped one, every play decision
        return SearchAgent(m, samples=40, topk=5, gap=9.0, seed=seed, max_rollouts=200)

    if kind.startswith("bid_"):   # shade the auction: positive delta = bid more readily, negative = pass more readily
        delta = float(kind.split("_", 1)[1]) / 9.0
        from sandaha.engine import BID, PASS
        class BidTweak(AIAgent):
            def act(self, g):
                if g.phase != BID:
                    return super().act(g)
                acts, phase, st, A, out = self._scores(g)
                if PASS not in acts or len(acts) == 1:
                    return acts[int(np.argmax(out[:, 0]))]
                ip = acts.index(PASS)
                q = out[:, 0].copy(); qp = q[ip]; q[ip] = -1e9
                ib = int(np.argmax(q))
                return acts[ib] if q[ib] + delta >= qp else PASS
        return BidTweak(m)
    if kind == "no_surrender":
        from sandaha.engine import SURRENDER
        class NoSur(AIAgent):
            def act(self, g):
                if g.phase != TRUMP:
                    return super().act(g)
                acts, phase, st, A, out = self._scores(g)
                q = out[:, 0].copy()
                if SURRENDER in acts and len(acts) > 1:
                    q[acts.index(SURRENDER)] = -1e9
                return acts[int(np.argmax(q))]
        return NoSur(m)
    class Oracle(AIAgent):  # cheats: evaluates candidates in the TRUE deal (others keep playing the net)
        phases = (PLAY,) if kind == "oracle_play" else (PLAY, TRUMP)
        def act(self, g):
            if g.phase not in self.phases:
                return super().act(g)
            acts, phase, st, A, out = self._scores(g)
            if len(acts) == 1:
                return acts[0]
            order = np.argsort(-out[:, 0])[:8]
            s = g.to_act; worlds = []
            for i in order:
                w = make_world(g, s, {}, seed=0)
                w.step(acts[int(i)]); worlds.append(w)
            rollout_batch(worlds, self.model)
            vals = [w.result["scores"][s] + 1e-3 * float(out[int(i), 0]) for w, i in zip(worlds, order)]
            return acts[int(order[int(np.argmax(vals))])]
    return Oracle(m)

def seat_test(args):
    kind, seed, n = args
    from sandaha.arena import random_deal, play_game
    rng = random.Random(seed); ax = make(kind, seed); ay = make("net", seed)
    tot = sq = 0.0; games = 0; byrole = {"banker": [0, 0.0], "xian": [0, 0.0]}
    for _ in range(n):
        hands, bottom = random_deal(rng); fb = rng.randrange(4); gs = rng.randrange(1 << 30)
        for k in range(4):
            seats = [ay] * 4; seats[k] = ax
            g = play_game(seats, None, hands, bottom, fb, seed=gs)
            sc = g.result["scores"][k]; tot += sc; sq += sc * sc; games += 1
            r = "banker" if g.result["banker"] == k else "xian"
            byrole[r][0] += 1; byrole[r][1] += sc
            # paired baseline: same deal, all four seats play the plain net
        for k in range(1):
            g0 = play_game([ay] * 4, None, hands, bottom, fb, seed=gs)
            for kk in range(4):
                r = "banker0" if g0.result["banker"] == kk else "xian0"
                byrole.setdefault(r, [0, 0.0]); byrole[r][0] += 1; byrole[r][1] += g0.result["scores"][kk]
    return {"tot": tot, "sq": sq, "games": games, "byrole": byrole}

if __name__ == "__main__":
    W = int(sys.argv[1]); out = {}
    with ProcessPoolExecutor(W) as ex:
        for kind, deals in [(k, int(d)) for k, d in (a.split(":") for a in sys.argv[2:])]:
            t0 = time.time()
            parts = list(ex.map(seat_test, [(kind, 9000 + i, max(1, deals // W)) for i in range(W)]))
            n = sum(p["games"] for p in parts); tot = sum(p["tot"] for p in parts); sq = sum(p["sq"] for p in parts)
            mean = tot / n; se = ((sq / n - mean * mean) / n) ** 0.5
            roles = {}
            for p in parts:
                for r, (c, v) in p["byrole"].items():
                    a = roles.setdefault(r, [0, 0.0]); a[0] += c; a[1] += v
            out[kind] = {"games": n, "avg_score": round(mean, 4), "se": round(se, 4), "sec": round(time.time() - t0, 1),
                         "roles": {r: {"n": c, "avg": round(v / max(1, c), 3)} for r, (c, v) in roles.items()}}
            print(kind, out[kind], flush=True)
            json.dump(out, open("strength_probes.json", "w"), indent=1)
