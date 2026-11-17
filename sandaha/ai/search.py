"""Search-enhanced play: determinized Monte-Carlo rollouts on top of the Q-network.

At a card-play decision the agent
  1. ranks the legal moves with the network and keeps the top-K;
  2. samples N full deals consistent with everything this seat knows
     (its own hand, every card played, its own buried cards, known voids);
  3. for every candidate and every sample, plays the hand out with the
     greedy network for all four seats and records this seat's final score;
  4. picks the candidate with the best average score (ties: network prior).

Bidding / trump / bury still use the network directly.  When the network is
already confident (gap between the two best Q-values > ``gap``) the search is
skipped, which keeps most decisions instant.
"""
from __future__ import annotations

import copy
import random

import numpy as np

from ..cards import NUM_TYPES, TRUMP_CLASS
from ..engine import PLAY
from . import features as F
from .agent import AIAgent


def sample_world(g, s, rng, tries=30):
    """Return (hands, buried) for the hidden cards, consistent with voids."""
    ctx = g.ctx
    pool = []
    unseen = g.unseen_for(s)
    for t in range(NUM_TYPES):
        pool += [t] * max(0, unseen[t])
    others = [o for o in range(4) if o != s]
    need = {o: g.hand_size(o) for o in others}
    bottom_need = 0 if s == g.banker else len(g.buried)
    slots = others + (["bottom"] if bottom_need else [])

    def allowed(slot, t):
        if slot == "bottom":
            return True
        return not g.void[slot][ctx.cls[t]]

    for attempt in range(tries):
        rng.shuffle(pool)
        # most constrained cards first
        order = sorted(pool, key=lambda t: sum(1 for o in others if allowed(o, t)))
        cap = dict(need)
        if bottom_need:
            cap["bottom"] = bottom_need
        hands = {o: [] for o in slots}
        ok = True
        for t in order:
            cands = [sl for sl in slots if cap[sl] > 0 and allowed(sl, t)]
            if not cands:
                ok = attempt == tries - 1  # last attempt: ignore voids
                if not ok:
                    break
                cands = [sl for sl in slots if cap[sl] > 0]
            # weight by remaining capacity so big hands get more cards
            w = [cap[sl] for sl in cands]
            sl = rng.choices(cands, weights=w)[0]
            hands[sl].append(t)
            cap[sl] -= 1
        if ok:
            return hands
    return None


def make_world(g, s, hands):
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
    w.cur = list(g.cur)
    w.tricks = list(g.tricks)
    w.events = []
    w.rng = random.Random(0)
    w.result = None
    return w


class SearchAgent(AIAgent):
    def __init__(self, model, samples=10, topk=4, gap=0.12, seed=None, name=None, max_rollouts=64):
        super().__init__(model, 0.0, seed, name or f"{model.name}+search")
        self.samples = samples
        self.topk = topk
        self.gap = gap
        self.max_rollouts = max_rollouts
        self.greedy = AIAgent(model)
        self.last_search = None

    def act(self, g):
        if g.phase != PLAY:
            return super().act(g)
        acts, phase, st, A, out = self._scores(g)
        if len(acts) == 1:
            return acts[0]
        q = out[:, 0]
        order = np.argsort(-q)
        if q[order[0]] - q[order[1]] > self.gap / 9.0:
            return acts[int(order[0])]
        cand = [int(i) for i in order[: self.topk]]
        n = max(2, min(self.samples, self.max_rollouts // len(cand)))
        s = g.to_act
        totals = np.zeros(len(cand))
        counts = np.zeros(len(cand))
        for k in range(n):
            hands = sample_world(g, s, self.rng)
            if hands is None:
                break
            for j, i in enumerate(cand):
                w = make_world(g, s, hands)
                w.step(acts[i])
                while not w.over:
                    w.step(self.greedy.act(w))
                totals[j] += w.result["scores"][s]
                counts[j] += 1
        if counts.min() == 0:
            return acts[int(order[0])]
        mean = totals / counts
        # tiny prior from the network to break ties
        score = mean + 0.05 * q[cand] * 9
        best = cand[int(np.argmax(score))]
        self.last_search = {"candidates": [acts[i] for i in cand], "mean": mean.tolist(), "n": int(counts[0])}
        return acts[best]
