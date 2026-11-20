"""Self-play profile and single-seat matches for the default model (numbers in docs/AI-STRENGTH.zh-CN.md).

    python scripts/strength_profile.py <workers> <selfplay_hands> <deals_per_match>
"""
import json, os, random, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("OMP_NUM_THREADS", "1")
BEST = "models/sandaha-best.npz"; LITE = "models/sandaha-lite.npz"

def mk(spec, seed):
    from sandaha.server import registry
    return registry.make_agent(spec, seed=seed)

def selfplay(args):
    seed, n = args
    import numpy as np
    from sandaha.arena import random_deal
    from sandaha.engine import Game, BID, TRUMP
    ag = mk(BEST, seed); rng = random.Random(seed); rows = []
    for _ in range(n):
        hands, bottom = random_deal(rng); fb = rng.randrange(4)
        g = Game(None, seed=rng.randrange(1 << 30), first_bidder=fb, hands=hands, bottom=bottom)
        pred = {}
        while not g.over:
            s = g.to_act; ph = g.phase
            acts, phase, st, A, out = ag._scores(g)
            i = int(np.argmax(out[:, 0]))
            if ph in (BID, TRUMP):
                pred[(ph, s)] = float(out[i, 0] * 9)   # last decision of that seat in that phase
            g.step(acts[i])
        r = g.result
        nb = sum(1 for _, v in g.bid_history if v)
        rows.append({"contract": r["contract"], "banker": r["banker"], "xian_win": bool(r["xian_win"]), "tier": r["tier"],
                     "xp": r["xian_points"], "bscore": r["scores"][r["banker"]], "nbids": nb,
                     "surrender": any(e["type"] == "surrender" for e in g.events),
                     "trump": g.trump, "koudi": bool(r.get("koudi") and r["koudi"].get("mult")),
                     "pred_trump": pred.get((TRUMP, r["banker"])), "pred_bid": pred.get((BID, r["banker"]))})
    return rows

def seat_test(args):
    """X sits alone against three Y; every deal is played 4 times with X in each seat."""
    x, y, seed, n = args
    from sandaha.arena import random_deal, play_game
    rng = random.Random(seed); ax = mk(x, seed); ay = mk(y, seed + 1)
    tot = 0.0; games = 0; bank = 0; bank_win = 0; sq = 0.0; xwin = 0
    for _ in range(n):
        hands, bottom = random_deal(rng); fb = rng.randrange(4); gs = rng.randrange(1 << 30)
        for k in range(4):
            seats = [ay] * 4; seats[k] = ax
            g = play_game(seats, None, hands, bottom, fb, seed=gs)
            sc = g.result["scores"][k]; tot += sc; sq += sc * sc; games += 1; xwin += sc > 0
            if g.result["banker"] == k:
                bank += 1; bank_win += not g.result["xian_win"]
    return {"tot": tot, "sq": sq, "games": games, "bank": bank, "bank_win": bank_win, "xwin": xwin}

def run_seat(ex, x, y, deals, W):
    parts = list(ex.map(seat_test, [(x, y, 7000 + i, deals // W) for i in range(W)]))
    n = sum(p["games"] for p in parts); tot = sum(p["tot"] for p in parts); sq = sum(p["sq"] for p in parts)
    mean = tot / n; se = ((sq / n - mean * mean) / n) ** 0.5
    bank = sum(p["bank"] for p in parts)
    return {"x": x, "y": y, "games": n, "avg_score": round(mean, 4), "se": round(se, 4), "pos_rate": round(sum(p["xwin"] for p in parts) / n, 4),
            "banker_rate": round(bank / n, 4), "banker_win": round(sum(p["bank_win"] for p in parts) / max(1, bank), 4)}

if __name__ == "__main__":
    W = int(sys.argv[1]); N = int(sys.argv[2]); D = int(sys.argv[3]); out = {}; t0 = time.time()
    with ProcessPoolExecutor(W) as ex:
        rows = [r for part in ex.map(selfplay, [(100 + i, N // W) for i in range(W)]) for r in part]
        n = len(rows); by = defaultdict(list)
        for r in rows: by[r["contract"]].append(r)
        sp = {"hands": n, "banker_win": round(sum(not r["xian_win"] for r in rows) / n, 4),
              "banker_avg_score": round(sum(r["bscore"] for r in rows) / n, 4),
              "avg_contract": round(sum(r["contract"] for r in rows) / n, 2),
              "all_pass": round(sum(r["nbids"] == 0 for r in rows) / n, 4),
              "surrender": round(sum(r["surrender"] for r in rows) / n, 4),
              "no_trump": round(sum(r["trump"] == 4 for r in rows) / n, 4),
              "koudi": round(sum(r["koudi"] for r in rows) / n, 4),
              "tiers": dict(Counter(r["tier"] for r in rows).most_common()),
              "by_contract": {c: {"n": len(v), "share": round(len(v) / n, 4), "banker_win": round(sum(not r["xian_win"] for r in v) / len(v), 3),
                                  "banker_avg": round(sum(r["bscore"] for r in v) / len(v), 3),
                                  "pred_bid": round(sum(r["pred_bid"] or 0 for r in v) / len(v), 3),
                                  "pred_trump": round(sum(r["pred_trump"] or 0 for r in v) / len(v), 3),
                                  "xp": round(sum(r["xp"] for r in v) / len(v), 1)} for c, v in sorted(by.items(), reverse=True)}}
        pb = [(r["pred_trump"], r["bscore"]) for r in rows if r["pred_trump"] is not None]
        sp["calib_trump_pred_mean"] = round(sum(a for a, _ in pb) / len(pb), 4)
        sp["calib_trump_real_mean"] = round(sum(b for _, b in pb) / len(pb), 4)
        out["selfplay"] = sp; out["t_selfplay"] = round(time.time() - t0, 1)
        json.dump(out, open("strength_profile.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        out["seat"] = []
        for x, y in ((BEST, "rule"), ("rule", BEST), (BEST, LITE), (LITE, BEST), (BEST + "?search", BEST)):
            d = D if "search" not in x else max(W, D // 16)
            out["seat"].append(run_seat(ex, x, y, d, W))
            json.dump(out, open("strength_profile.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open("strength_profile.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("done", out["seconds"])
