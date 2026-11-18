"""Search-enhanced play: belief-weighted determinized Monte-Carlo rollouts
accelerated with batched neural inference (``model.q_batch``).

Upgrades over v1 search:
  1. Belief-weighted determinization: respects suit/trump voids, **pair voids**
     (players who failed to follow a pair cannot hold a pair in that class),
     unforced-pass changzhu caps (< 7 changzhu), and realistic **Banker bottom
     bury priors** (Banker almost never buries Jokers/changzhu/Side Aces).
  2. Lockstep batched rollouts via ``model.q_batch``: ~10x faster than sequential
     single-game rollouts, allowing 16-24 worlds per close decision in ~20 ms.
  3. Searches both ``TRUMP`` (trump selection / surrender) and ``PLAY``, with
     deeper sampling in the endgame where card counting is tightest.
"""
from __future__ import annotations

import copy
import random

import numpy as np

from ..cards import BJ, NUM_TYPES, POINTS, RA, SJ, TRUMP_CLASS, is_changzhu, rank_of
from ..engine import PLAY, TRUMP
from . import features as F
from .agent import AIAgent

CHANGZHU_SET = frozenset(t for t in range(NUM_TYPES) if is_changzhu(t))


def sample_world(g, s, rng, tries=40):
    """Return ``hands`` mapping each hidden seat (and ``'bottom'`` if unknown) to
    a list of card types, consistent with known voids, pair-voids, bidding caps,
    and realistic Banker burying priors."""
    ctx = g.ctx
    unseen = g.unseen_for(s)
    pool = []
    for t in range(NUM_TYPES):
        c = unseen[t]
        if c > 0:
            pool.extend([t] * c)

    others = [o for o in range(4) if o != s]
    need = {o: g.hand_size(o) for o in others}
    bottom_need = 0
    if g.phase == PLAY and s != g.banker:
        bottom_need = len(g.buried)
    slots = others + (["bottom"] if bottom_need > 0 else [])

    pv = getattr(g, "pair_void", None)
    pu = getattr(g, "passed_unforced", [False, False, False, False])
    sbids = getattr(g, "seat_bids", [0, 0, 0, 0])
    cz_cap = {}
    for o in others:
        if pu[o] and g.rules.force_enabled:
            played_cz = sum(g.played[o][t] for t in CHANGZHU_SET)
            cz_cap[o] = max(0, g.rules.force_changzhu - 1 - played_cz)
        else:
            cz_cap[o] = 99

    def base_allowed(slot, t):
        if slot == "bottom":
            return True
        if ctx is not None and g.void[slot][ctx.cls[t]]:
            return False
        return True

    def prior_mult(slot, t):
        if ctx is None:
            if slot != "bottom" and sbids[slot] > 0 and t in CHANGZHU_SET:
                return 1.4
            return 1.0
        ist = ctx.is_trump[t]
        if slot == "bottom":
            if t == BJ or t == SJ or (ist and ctx.pos[t] <= 5):
                return 0.015
            if ist:
                return 0.10
            if rank_of(t) == RA:
                return 0.06
            if POINTS[t] > 0:
                return 1.45
            return 2.2
        if slot == g.banker:
            if ist:
                return 1.75 if ctx.pos[t] <= 5 else 1.4
            if rank_of(t) == RA:
                return 1.25
            return 0.9
        if sbids[slot] > 0 and t in CHANGZHU_SET:
            return 1.35
        return 1.0

    mult_table = {(sl, t): prior_mult(sl, t) for sl in slots for t in set(pool)}
    # Sort most constrained cards first (voided suits, pairs when pair_void exists, changzhu)
    constraint_score = {}
    for t in set(pool):
        n_ok = sum(1 for o in others if base_allowed(o, t))
        p_pen = 0
        if ctx is not None and pv is not None and unseen[t] >= 2:
            p_pen = sum(1 for o in others if pv[o][ctx.cls[t]])
        cz_pen = 1 if t in CHANGZHU_SET and any(cz_cap[o] < 20 for o in others) else 0
        constraint_score[t] = (n_ok, -p_pen, -cz_pen)

    for attempt in range(tries):
        strict = attempt < tries - 6
        rng.shuffle(pool)
        order = sorted(pool, key=lambda t: constraint_score[t])
        cap = dict(need)
        if bottom_need > 0:
            cap["bottom"] = bottom_need
        hands = {sl: [] for sl in slots}
        type_cnt = {sl: {} for sl in others}
        cz_cnt = {o: 0 for o in others}
        ok = True

        for t in order:
            is_cz = t in CHANGZHU_SET
            cands = []
            for sl in slots:
                if cap[sl] <= 0 or not base_allowed(sl, t):
                    continue
                if strict and sl != "bottom":
                    if is_cz and cz_cnt[sl] >= cz_cap[sl]:
                        continue
                    if ctx is not None and pv is not None and pv[sl][ctx.cls[t]] and type_cnt[sl].get(t, 0) >= 1:
                        continue
                cands.append(sl)
            if not cands:
                # Relax soft constraints (pair_void / cz_cap) before relaxing hard suit voids
                cands = [sl for sl in slots if cap[sl] > 0 and base_allowed(sl, t)]
                if not cands:
                    if attempt < tries - 1:
                        ok = False
                        break
                    cands = [sl for sl in slots if cap[sl] > 0]
            weights = [cap[sl] * mult_table[(sl, t)] for sl in cands]
            sl = rng.choices(cands, weights=weights, k=1)[0]
            hands[sl].append(t)
            cap[sl] -= 1
            if sl != "bottom":
                type_cnt[sl][t] = type_cnt[sl].get(t, 0) + 1
                if is_cz:
                    cz_cnt[sl] += 1
        if ok:
            return hands
    return None


def make_world(g, s, hands, seed=0):
    w = copy.copy(g)
    w.hands = [list(h) for h in g.hands]
    for o, cards in hands.items():
        if o == "bottom":
            continue
        h = [0] * NUM_TYPES
        for t in cards:
            h[t] += 1
        w.hands[o] = h
    if "bottom" in hands:
        w.buried = list(hands["bottom"])
    else:
        w.buried = list(g.buried)
    w.played = [list(p) for p in g.played]
    w.played_total = list(g.played_total)
    w.void = [list(v) for v in g.void]
    if hasattr(g, "pair_void"):
        w.pair_void = [list(pv) for pv in g.pair_void]
    if hasattr(g, "seat_min_bid"):
        w.seat_min_bid = list(g.seat_min_bid)
        w.seat_bids = list(g.seat_bids)
        w.passed_unforced = list(g.passed_unforced)
    w.cur = list(g.cur)
    w.tricks = list(g.tricks)
    w.events = []
    w.rng = random.Random(seed)
    w.result = None
    return w


def rollout_batch(worlds: list, model):
    """Advance all ``worlds`` in lockstep using ``model.q_batch`` until game over."""
    fv = model.feat_ver
    live = [w for w in worlds if not w.over]
    while live:
        groups = {}
        nxt_live = []
        # Fast-forward forced single-action states without touching the network!
        for w in live:
            while not w.over:
                acts = w.legal_actions()
                if len(acts) == 1:
                    w.step(acts[0])
                else:
                    s = w.to_act
                    ph, st, A = F.encode(w, s, acts, fv)
                    groups.setdefault(ph, []).append((w, acts, st, A))
                    break
            if not w.over:
                nxt_live.append(w)
        if not groups:
            break
        for ph, items in groups.items():
            outs = model.q_batch(ph, [it[2] for it in items], [it[3] for it in items])
            for (w, acts, _, _), out in zip(items, outs):
                w.step(acts[int(np.argmax(out[:, 0]))])
        live = [w for w in nxt_live if not w.over]


class SearchAgent(AIAgent):
    def __init__(self, model, samples=16, topk=4, gap=0.22, seed=None, name=None, max_rollouts=80):
        super().__init__(model, 0.0, seed, name or f"{model.name}+search")
        self.samples = samples
        self.topk = topk
        self.gap = gap
        self.max_rollouts = max_rollouts
        self.last_search = None

    def act(self, g):
        if g.phase not in (PLAY, TRUMP):
            return super().act(g)
        acts, phase, st, A, out = self._scores(g)
        if len(acts) == 1:
            return acts[0]
        q = out[:, 0]
        order = np.argsort(-q)

        # Adaptive search budget: deeper in endgame & trump selection
        s = g.to_act
        if g.phase == TRUMP:
            gap_th = max(self.gap, 0.35) / 9.0
            topk = min(len(acts), 3)
            target_samples = max(12, self.samples)
            rollout_cap = max(48, self.max_rollouts)
        else:
            hsz = g.hand_size(s)
            endgame = hsz <= 8
            gap_th = (max(self.gap, 0.36) if endgame else self.gap) / 9.0
            topk = min(len(acts), self.topk + (1 if endgame else 0))
            target_samples = self.samples + (8 if endgame else 0)
            rollout_cap = self.max_rollouts + (32 if endgame else 0)

        if q[order[0]] - q[order[1]] > gap_th:
            return acts[int(order[0])]

        cand = [int(i) for i in order[:topk]]
        # Include any extra near-tied candidates within 0.08 points of best
        for idx in order[topk:topk + 2]:
            if q[order[0]] - q[int(idx)] <= 0.08 / 9.0:
                cand.append(int(idx))

        n = max(4, min(target_samples, rollout_cap // len(cand)))
        sampled_deals = []
        for k in range(n):
            hands = sample_world(g, s, self.rng)
            if hands is not None:
                sampled_deals.append((hands, self.rng.randrange(1 << 30)))
        if not sampled_deals:
            return acts[int(order[0])]

        worlds = []
        world_meta = []
        for hands, wseed in sampled_deals:
            for j, ci in enumerate(cand):
                w = make_world(g, s, hands, seed=wseed)
                w.step(acts[ci])
                worlds.append(w)
                world_meta.append(j)

        rollout_batch(worlds, self.model)

        totals = np.zeros(len(cand), dtype=np.float64)
        counts = np.zeros(len(cand), dtype=np.int32)
        for w, j in zip(worlds, world_meta):
            totals[j] += w.result["scores"][s]
            counts[j] += 1

        if counts.min() == 0:
            return acts[int(order[0])]
        mean = totals / counts
        # Network Q prior blends smoothly with Monte-Carlo rollout estimate
        score = mean + 0.08 * q[cand] * 9.0
        best = cand[int(np.argmax(score))]
        self.last_search = {
            "candidates": [acts[i] for i in cand],
            "mean": mean.tolist(),
            "n": int(counts[0]),
        }
        return acts[best]
