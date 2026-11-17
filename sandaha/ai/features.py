"""Observation / action encoders for the four decision types.

Every feature lies in [0, 1] so samples can be stored as uint8 (x255).
Cards are encoded in a canonical, trump-relative layout (see cards.canonical_index)
as 92 bits: [count>=1 for 46 slots, count==2 for 46 slots].
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .. import moves as M
from ..cards import (BJ, NO_TRUMP, NUM_TYPES, POINTS, SJ, TRUMP_CLASS, canonical_index, is_changzhu,
                     rank_of, R2, R7, suit_of, trump_context)
from ..engine import PASS, SURRENDER

CARD_BITS = 92
PHASES = ("bid", "trump", "bury", "play")

# ---- dimensions ----
BID_STATE = CARD_BITS + 4 + 4 + 2 + 4 + 12 + 4
BID_ACTION = 18
TRUMP_STATE = CARD_BITS + 4
TRUMP_ACTION = CARD_BITS + 3 + 4
BURY_STATE = CARD_BITS * 2 + 6
BURY_ACTION = 46 + 5
PLAY_STATE = CARD_BITS * 10 + 4 + 15 + 32
PLAY_ACTION = CARD_BITS + 12

DIMS = {
    "bid": (BID_STATE, BID_ACTION),
    "trump": (TRUMP_STATE, TRUMP_ACTION),
    "bury": (BURY_STATE, BURY_ACTION),
    "play": (PLAY_STATE, PLAY_ACTION),
}


@lru_cache(maxsize=None)
def _perm(order: tuple) -> np.ndarray:
    return np.array(canonical_index(order), dtype=np.int64)


@lru_cache(maxsize=None)
def _trump_perm(trump: int) -> np.ndarray:
    return np.array(trump_context(trump).canon, dtype=np.int64)


def enc(counts, perm, out=None, off=0):
    if out is None:
        out = np.zeros(CARD_BITS, np.float32)
        off = 0
    c = np.asarray(counts)
    out[off + perm] = c >= 1
    out[off + 46 + perm] = c >= 2
    return out


def counts_from(cards):
    c = np.zeros(NUM_TYPES, np.int8)
    for t in cards:
        c[t] += 1
    return c


def length_order(hand):
    """Suits sorted by length (desc) - canonical order before trump is known."""
    lens = []
    for s in range(4):
        n = sum(hand[s * 11 + r] for r in range(11) if r not in (R2, R7))
        lens.append((-n, s))
    return tuple(s for _, s in sorted(lens))


def suit_slot(ctx):
    """class index -> canonical slot (0..3 suits in canonical order, 4 = trump)."""
    slot = [0] * 5
    for i, s in enumerate(ctx.suit_order):
        slot[s] = i
    slot[TRUMP_CLASS] = 4
    return slot


# ------------------------------------------------------------------ bid --

def bid_state(g, s):
    hand = g.hands[s]
    x = np.zeros(BID_STATE, np.float32)
    order = length_order(hand)
    enc(hand, _perm(order), x, 0)
    o = CARD_BITS
    x[o] = hand[BJ] / 2
    x[o + 1] = hand[SJ] / 2
    x[o + 2] = sum(hand[t * 11 + R7] for t in range(4)) / 8
    x[o + 3] = sum(hand[t * 11 + R2] for t in range(4)) / 8
    o += 4
    for i, su in enumerate(order):
        x[o + i] = sum(hand[su * 11 + r] for r in range(11) if r not in (R2, R7)) / 12
    o += 4
    x[o] = sum(POINTS[t] * hand[t] for t in range(NUM_TYPES)) / 100
    x[o + 1] = g.changzhu_count(s) / 12
    o += 2
    x[o + (g.first_bidder - s) % 4] = 1
    o += 4
    last = {}
    for seat, v in g.bid_history:
        last[seat] = v
    for seat in range(4):
        r = (seat - s) % 4
        x[o + 3 * r] = g.passed[seat]
        x[o + 3 * r + 1] = (last.get(seat) or 0) / 80
        x[o + 3 * r + 2] = g.best_bidder == seat
    o += 12
    x[o] = (g.best_bid or 0) / 80
    x[o + 1] = g._forced(s)
    x[o + 2] = len(g.bid_history) / 12
    x[o + 3] = g.best_bid is None
    return x


def bid_actions(g, acts):
    vals = g.rules.bid_values()
    a = np.zeros((len(acts), BID_ACTION), np.float32)
    for i, v in enumerate(acts):
        if v == PASS:
            a[i, 16] = 1
        else:
            j = vals.index(v) if v in vals else 15
            a[i, min(j, 15)] = 1
            a[i, 17] = v / 80
    return a


# ---------------------------------------------------------------- trump --

def trump_state(g, s):
    hand = g.hands[s]
    x = np.zeros(TRUMP_STATE, np.float32)
    enc(hand, _perm(length_order(hand)), x, 0)
    o = CARD_BITS
    x[o] = g.contract / 80
    x[o + 1] = g.changzhu_count(s) / 16
    x[o + 2] = hand[BJ] / 2
    x[o + 3] = hand[SJ] / 2
    return x


def trump_actions(g, s, acts):
    hand = g.hands[s]
    a = np.zeros((len(acts), TRUMP_ACTION), np.float32)
    for i, t in enumerate(acts):
        if t == SURRENDER:
            a[i, CARD_BITS + 2] = 1
            continue
        enc(hand, _trump_perm(t), a[i], 0)
        ctx = trump_context(t)
        a[i, CARD_BITS + (1 if t == NO_TRUMP else 0)] = 1
        o = CARD_BITS + 3
        ntr = sum(hand[u] for u in range(NUM_TYPES) if ctx.is_trump[u])
        a[i, o] = ntr / 30
        a[i, o + 1] = sum(1 for u in range(NUM_TYPES) if ctx.is_trump[u] and hand[u] == 2) / 10
        voids = sum(1 for c in range(4) if ctx.class_size[c] and
                    not any(hand[u] for u in range(NUM_TYPES) if ctx.cls[u] == c))
        a[i, o + 2] = voids / 4
        a[i, o + 3] = sum(hand[u] for u in range(NUM_TYPES) if ctx.is_trump[u] and ctx.pos[u] <= 5) / 16
    return a


# ----------------------------------------------------------------- bury --

def bury_state(g, s):
    hand = g.hands[s]
    perm = _trump_perm(g.trump)
    x = np.zeros(BURY_STATE, np.float32)
    enc(hand, perm, x, 0)
    enc(counts_from(g.buried), perm, x, CARD_BITS)
    o = 2 * CARD_BITS
    ctx = g.ctx
    x[o] = g.contract / 80
    x[o + 1] = g.trump == NO_TRUMP
    x[o + 2] = len(g.buried) / 8
    x[o + 3] = sum(hand[u] for u in range(NUM_TYPES) if ctx.is_trump[u]) / 30
    x[o + 4] = sum(POINTS[t] for t in g.buried) / 60
    x[o + 5] = sum(hand[t] * POINTS[t] for t in range(NUM_TYPES)) / 120
    return x


def bury_actions(g, s, acts):
    hand = g.hands[s]
    ctx = g.ctx
    perm = ctx.canon
    lens = [sum(hand[u] for u in range(NUM_TYPES) if ctx.cls[u] == c) for c in range(5)]
    a = np.zeros((len(acts), BURY_ACTION), np.float32)
    for i, t in enumerate(acts):
        a[i, perm[t]] = 1
        a[i, 46] = ctx.is_trump[t]
        a[i, 47] = POINTS[t] / 10
        a[i, 48] = lens[ctx.cls[t]] / 20
        a[i, 49] = ctx.pos[t] / 15
        a[i, 50] = hand[t] == 2
    return a


# ----------------------------------------------------------------- play --

def play_state(g, s):
    ctx = g.ctx
    perm = _trump_perm(g.trump)
    x = np.zeros(PLAY_STATE, np.float32)
    enc(g.hands[s], perm, x, 0)
    o = CARD_BITS
    for r in range(4):
        enc(g.played[(s + r) % 4], perm, x, o + r * CARD_BITS)
    o += 4 * CARD_BITS
    cur = {seat: cards for seat, cards in g.cur}
    for r in range(1, 4):
        seat = (s + r) % 4
        if seat in cur:
            enc(counts_from(cur[seat]), perm, x, o + (r - 1) * CARD_BITS)
    o += 3 * CARD_BITS
    unseen = g.unseen_for(s)
    enc(unseen, perm, x, o)
    o += CARD_BITS
    if s == g.banker:
        enc(counts_from(g.buried), perm, x, o)
    o += CARD_BITS
    x[o + (g.banker - s) % 4] = 1
    o += 4
    slot = suit_slot(ctx)
    for r in range(1, 4):
        v = g.void[(s + r) % 4]
        for c in range(5):
            if v[c]:
                x[o + (r - 1) * 5 + slot[c]] = 1
    o += 15
    # scalars (32)
    x[o] = g.contract / 80
    x[o + 1] = min(g.xian_points, 200) / 200
    x[o + 2] = min(g.trick_points(), 100) / 100
    x[o + 3] = len(g.tricks) / 21
    x[o + 4] = g.hand_size(s) / 21
    x[o + 5] = g.trump == NO_TRUMP
    x[o + 6] = (sum(POINTS[t] for t in g.buried) / 60) if s == g.banker else 0
    for r in range(1, 4):
        x[o + 6 + r] = g.hand_size((s + r) % 4) / 21
    need = (g.contract - g.xian_points) / 200
    x[o + 10] = min(max(need, -1), 1) * 0.5 + 0.5
    if g.cur:
        x[o + 11 + (g.cur[0][0] - s) % 4] = 1
        w = g.current_winner()
        x[o + 15 + (w - s) % 4] = 1
        x[o + 19] = (w == g.banker) == (s == g.banker)
        lead = g.lead_shape
        x[o + 20 + slot[lead.cls]] = 1
        x[o + 25 + lead.kind] = 1
        x[o + 29] = lead.n / 12
    x[o + 30] = s == g.banker
    x[o + 31] = len(g.cur) / 3
    return x


def play_actions(g, s, acts):
    ctx = g.ctx
    perm = _trump_perm(g.trump)
    a = np.zeros((len(acts), PLAY_ACTION), np.float32)
    lead = g.lead_shape
    best_pow = None
    if g.cur:
        bi = M.trick_winner(g.cur, ctx)
        best_pow = M.power(lead, g.cur[bi][1], ctx, is_lead=(bi == 0))
    for i, mv in enumerate(acts):
        enc(counts_from(mv), perm, a[i], 0)
        o = CARD_BITS
        sh = M.analyze(mv, ctx)
        a[i, o + sh.kind] = 1
        all_trump = all(ctx.is_trump[t] for t in mv)
        if lead is not None:
            p = M.power(lead, mv, ctx)
            a[i, o + 4] = p is not None and p > best_pow
            a[i, o + 5] = all_trump and lead.cls != TRUMP_CLASS
            a[i, o + 6] = all(ctx.cls[t] == lead.cls for t in mv)
        else:
            a[i, o + 7] = 1
        a[i, o + 8] = M.points_of(mv) / 50
        a[i, o + 9] = len(mv) / 12
        a[i, o + 10] = all_trump
        a[i, o + 11] = sh.top / 15
    return a


def encode(g, s, acts):
    """Return (phase, state_vec, action_matrix) for the decision at hand."""
    ph = g.phase
    if ph == "bid":
        return ph, bid_state(g, s), bid_actions(g, acts)
    if ph == "trump":
        return ph, trump_state(g, s), trump_actions(g, s, acts)
    if ph == "bury":
        return ph, bury_state(g, s), bury_actions(g, s, acts)
    return ph, play_state(g, s), play_actions(g, s, acts)
