"""Hand-written heuristic player used as the evaluation baseline ("规则AI")."""
from __future__ import annotations

import random

from .. import moves as M
from ..cards import (BJ, NO_TRUMP, NUM_TYPES, POINTS, RA, RK, SJ, TRUMP_CLASS, rank_of, suit_of,
                     trump_context)
from ..engine import BID, BURY, PASS, PLAY, SURRENDER, TRUMP


def trump_strength(hand, trump):
    ctx = trump_context(trump)
    sc = 0.0
    ntrump = 0
    for t in range(NUM_TYPES):
        c = hand[t]
        if not c:
            continue
        if ctx.cls[t] == TRUMP_CLASS:
            p = ctx.pos[t]
            w = {0: 3.0, 1: 2.6, 2: 2.2, 3: 2.0, 4: 1.8, 5: 1.6}.get(p, 1.4 if p <= 7 else 1.0)
            if trump == NO_TRUMP:
                w = {0: 3.0, 1: 2.6, 2: 2.1, 3: 1.7}[p]
            sc += w * c
            ntrump += c
            if c == 2:
                sc += 0.8
        else:
            r = rank_of(t)
            if r == RA:
                sc += 1.2 * c + (0.8 if c == 2 else 0)
            elif r == RK and hand[t + 1]:
                sc += 0.4 * c
    # shortness in side suits is valuable when long in trump
    for s in range(4):
        if trump != NO_TRUMP and s == trump:
            continue
        ln = sum(hand[t] for t in range(NUM_TYPES) if ctx.cls[t] == s)
        if ln == 0:
            sc += 0.4 * max(0, ntrump - 8)
        elif ln <= 2:
            sc += 0.15 * max(0, ntrump - 8)
    return sc, ntrump


def best_trump(hand, allow_no_trump=True):
    opts = [0, 1, 2, 3] + ([NO_TRUMP] if allow_no_trump else [])
    scored = [(trump_strength(hand, t)[0] - (1.5 if t == NO_TRUMP else 0), t) for t in opts]
    return max(scored)


def card_cost(t, ctx):
    """How valuable a card is to keep (higher = more valuable)."""
    if ctx.cls[t] == TRUMP_CLASS:
        return 20 + (20 - ctx.pos[t])
    return 9 - ctx.pos[t]


class RuleBot:
    name = "rule"

    def __init__(self, seed=None, bid_base=16.0, bid_slope=2.0):
        self.rng = random.Random(seed)
        self.bid_base = bid_base
        self.bid_slope = bid_slope
        self._bury_plan = None

    # ------------------------------------------------------------ bidding --
    def target_bid(self, hand_strength, rules, with_bottom=False):
        base = self.bid_base + (2.5 if with_bottom else 0)
        if getattr(rules, "shuang_jin_dan_chu", False):
            base -= 4.5  # banker wins pay 2x -> bid ~10-15 pts lower aggressively
        if hand_strength < base:
            return None
        steps = int((hand_strength - base) / self.bid_slope)
        return max(rules.min_bid, rules.max_bid - rules.bid_step * steps)

    def act(self, g):
        if g.phase == BID:
            return self._bid(g)
        if g.phase == TRUMP:
            return self._trump(g)
        if g.phase == BURY:
            return self._bury(g)
        if g.phase == PLAY:
            return self._play(g)
        raise RuntimeError("game over")

    def _bid(self, g):
        legal = g.legal_bids()
        vals = [v for v in legal if v != PASS]
        if not vals:
            return PASS
        strength, _ = best_trump(g.hands[g.to_act], g.rules.allow_no_trump)
        target = self.target_bid(strength + 2.0, g.rules)  # +2: expected value of the bottom
        top = max(vals)
        if PASS not in legal:
            return top
        if target is None or top < target:
            return PASS
        return top

    def _trump(self, g):
        hand = g.hands[g.to_act]
        strength, t = best_trump(hand, g.rules.allow_no_trump)
        if SURRENDER in g.legal_trumps():
            target = self.target_bid(strength, g.rules, with_bottom=True)
            if target is None or target > g.contract + 15:
                return SURRENDER
        self._bury_plan = None
        return t

    # -------------------------------------------------------------- bury --
    def _plan_bury(self, g):
        hand = list(g.hands[g.to_act])
        ctx = g.ctx
        lens = {s: sum(hand[t] for t in range(NUM_TYPES) if ctx.cls[t] == s) for s in range(4)}
        has_a = {s: any(hand[t] and ctx.cls[t] == s and rank_of(t) == RA for t in range(NUM_TYPES))
                 for s in range(4)}
        cards = [t for t in range(NUM_TYPES) for _ in range(hand[t])]

        def key(t):
            if ctx.cls[t] == TRUMP_CLASS:
                return (9, 0, 0, 0)
            s = ctx.cls[t]
            keep_high = rank_of(t) in (RA,) or (rank_of(t) == RK and has_a[s])
            return (1 if keep_high else 0, has_a[s], lens[s], POINTS[t] > 0, -ctx.pos[t])

        cards.sort(key=key)
        return cards[:8]

    def _bury(self, g):
        if not self._bury_plan or len(g.buried) == 0:
            self._bury_plan = self._plan_bury(g)
        plan = self._bury_plan
        for t in plan:
            if g.hands[g.to_act][t] > 0:
                plan.remove(t)
                return t
        return g.legal_actions()[0]

    # -------------------------------------------------------------- play --
    def _play(self, g):
        cands = g.legal_plays()
        if len(cands) == 1:
            return cands[0]
        if not g.cur:
            return self._lead(g, cands)
        return self._follow(g, cands)

    def _is_top(self, cards, g, s):
        """Is this single/pair the strongest remaining in its class?"""
        ctx = g.ctx
        unseen = g.unseen_for(s)
        sh = M.analyze(cards, ctx)
        if sh.kind == M.THROW:
            return True
        if sh.kind == M.SINGLE:
            return not any(unseen[u] and ctx.cls[u] == sh.cls and ctx.pos[u] < sh.top for u in range(NUM_TYPES))
        pairs = [u for u in range(NUM_TYPES) if unseen[u] == 2 and ctx.cls[u] == sh.cls]
        if sh.kind == M.PAIR:
            return not any(ctx.pos[u] < sh.top for u in pairs)
        return not any(ctx.pos[tr[0]] < sh.top for tr in M.tractors_from_pairs(pairs, ctx, sh.pairs))

    def _lead(self, g, cands):
        s = g.to_act
        ctx = g.ctx
        banker = s == g.banker
        opp_void = [False] * 5
        for o in range(4):
            if o == s or (not banker and o != g.banker):
                continue
            for c in range(4):
                opp_void[c] = opp_void[c] or g.void[o][c]
        hand = g.hands[s]
        ntrump = sum(hand[t] for t in range(NUM_TYPES) if ctx.cls[t] == TRUMP_CLASS)
        unseen = g.unseen_for(s)
        unseen_trump = sum(unseen[t] for t in range(NUM_TYPES) if ctx.cls[t] == TRUMP_CLASS)

        def score(mv):
            sh = M.analyze(mv, ctx)
            top = self._is_top(mv, g, s)
            v = 0.0
            if sh.kind == M.THROW:
                v += 6 + len(mv)
            if sh.kind in (M.PAIR, M.TRACTOR):
                v += 2 * sh.pairs
            if sh.cls == TRUMP_CLASS:
                if banker and unseen_trump > 0 and ntrump >= 6:
                    v += (4 if top else 1)
                elif banker:
                    v -= 1
                else:
                    v -= 3 if not top else 0
            else:
                if opp_void[sh.cls]:
                    v -= 6
                v += 5 if top else 0
                if not top:
                    v -= M.points_of(mv) * 0.4
                    v -= sum(card_cost(t, ctx) for t in mv) * 0.1
            return v + self.rng.random() * 0.1

        return max(cands, key=score)

    def _follow(self, g, cands):
        s = g.to_act
        ctx = g.ctx
        lead = g.lead_shape
        win_seat = g.current_winner()
        teammate_winning = (win_seat != g.banker) == (s != g.banker)
        last = len(g.cur) == 3
        trick_pts = g.trick_points()
        best_i = M.trick_winner(g.cur, ctx)
        best_pow = M.power(lead, g.cur[best_i][1], ctx, is_lead=(best_i == 0))
        win_secure = teammate_winning and (last or self._is_top(g.cur[best_i][1], g, s)
                                           and lead.cls != TRUMP_CLASS and best_pow[0] == 0
                                           and not any(g.void[o][lead.cls] for o in range(4)
                                                       if o != s and (o == g.banker) != (s == g.banker)))
        if teammate_winning and best_pow[0] == 1 and lead.kind != M.THROW:
            win_secure = win_secure or last or best_pow[1] >= -1

        def cost(mv):
            return sum(card_cost(t, ctx) for t in mv)

        def wins(mv):
            p = M.power(lead, mv, ctx)
            return p is not None and p > best_pow

        if teammate_winning:
            if win_secure:
                return max(cands, key=lambda mv: (M.points_of(mv), -cost(mv)))
            return min(cands, key=lambda mv: (M.points_of(mv), cost(mv)))
        winning = [mv for mv in cands if wins(mv)]
        if winning and (trick_pts >= 10 or last or (s != g.banker and trick_pts >= 5)
                        or (lead.kind != M.SINGLE)):
            if last:
                return max(winning, key=lambda mv: (M.points_of(mv), -cost(mv)))
            return min(winning, key=lambda mv: (cost(mv), -M.points_of(mv)))
        return min(cands, key=lambda mv: (M.points_of(mv), cost(mv)))
