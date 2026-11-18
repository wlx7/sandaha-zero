"""Observation / action encoders for the four decision types.

Supports both:
  * ``feat_ver == 1``: legacy encoding used by v1 models (100% bit-compatible)
  * ``feat_ver == 2``: enhanced tactical encoding (pair-void tracking, bidding
    memory & changzhu deductions, settlement margin features, boss-card &
    closing-seat certainty, void-creation & pair-breaking move features, and
    strict [0, 1] clipping with non-overflowing denominators).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .. import moves as M
from ..cards import (BJ, NO_TRUMP, NUM_TYPES, POINTS, RA, RK, R2, R7, SJ, TRUMP_CLASS,
                     canonical_index, is_changzhu, make, rank_of, suit_of, trump_context)
from ..engine import PASS, SURRENDER

CARD_BITS = 92
PHASES = ("bid", "trump", "bury", "play")

# ---- v1 dimensions (legacy compatibility) ----
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

# ---- v2 dimensions ----
BID_STATE_V2 = 137
BID_ACTION_V2 = 20
TRUMP_STATE_V2 = 112
TRUMP_ACTION_V2 = 105
BURY_STATE_V2 = 200
BURY_ACTION_V2 = 58
PLAY_STATE_V2 = 1027
PLAY_ACTION_V2 = 116

DIMS_V2 = {
    "bid": (BID_STATE_V2, BID_ACTION_V2),
    "trump": (TRUMP_STATE_V2, TRUMP_ACTION_V2),
    "bury": (BURY_STATE_V2, BURY_ACTION_V2),
    "play": (PLAY_STATE_V2, PLAY_ACTION_V2),
}

CHANGZHU_TYPES = tuple(t for t in range(NUM_TYPES) if is_changzhu(t))
POINT_TYPES = tuple((t, POINTS[t]) for t in range(NUM_TYPES) if POINTS[t] > 0)
SIDE_NON_CZ = tuple(
    tuple(s * 11 + r for r in range(11) if r not in (R2, R7))
    for s in range(4)
)


def dims_for(feat_ver: int = 1) -> dict[str, tuple[int, int]]:
    return DIMS_V2 if feat_ver >= 2 else DIMS


@lru_cache(maxsize=None)
def _perm(order: tuple) -> np.ndarray:
    return np.array(canonical_index(order), dtype=np.int64)


@lru_cache(maxsize=None)
def _trump_perm(trump: int) -> np.ndarray:
    return np.array(trump_context(trump).canon, dtype=np.int64)


@lru_cache(maxsize=None)
def _ctx_fast(trump: int):
    ctx = trump_context(trump)
    perm = _trump_perm(trump)
    slot = suit_slot(ctx)
    cls_types = tuple(
        tuple(t for t in range(NUM_TYPES) if ctx.cls[t] == c)
        for c in range(5)
    )
    high_trump_types = tuple(t for t in cls_types[TRUMP_CLASS] if ctx.pos[t] <= 5)
    higher_in_cls = {}
    for c in range(5):
        cts = cls_types[c]
        max_p = max((ctx.pos[t] for t in cts), default=0)
        for p in range(max_p + 2):
            higher_in_cls[(c, p)] = tuple(u for u in cts if ctx.pos[u] < p)
    return ctx, perm, slot, cls_types, high_trump_types, higher_in_cls


def enc(counts, perm, out=None, off=0):
    if out is None:
        out = np.zeros(CARD_BITS, np.float32)
        off = 0
    c = np.asarray(counts)
    out[off + perm] = c >= 1
    out[off + 46 + perm] = c >= 2
    return out


def _enc_move(mv, perm, out, off=0):
    """Fast direct card encoding for card tuples/lists (order-independent)."""
    for t in mv:
        p = off + perm[t]
        if out[p] > 0.0:
            out[p + 46] = 1.0
        else:
            out[p] = 1.0


def counts_from(cards):
    c = np.zeros(NUM_TYPES, np.int8)
    for t in cards:
        c[t] += 1
    return c


def length_order(hand):
    """Suits sorted by length (desc) - canonical order before trump is known."""
    lens = []
    for s in range(4):
        n = sum(hand[t] for t in SIDE_NON_CZ[s])
        lens.append((-n, s))
    return tuple(s for _, s in sorted(lens))


def suit_slot(ctx):
    """class index -> canonical slot (0..3 suits in canonical order, 4 = trump)."""
    slot = [0] * 5
    for i, s in enumerate(ctx.suit_order):
        slot[s] = i
    slot[TRUMP_CLASS] = 4
    return slot


def _hand_points(hand) -> int:
    return sum(hand[t] * p for t, p in POINT_TYPES)


# ==================================================================== v1 ==

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


# ==================================================================== v2 ==

def bid_state_v2(g, s):
    hand = g.hands[s]
    x = np.zeros(BID_STATE_V2, np.float32)
    order = length_order(hand)
    enc(hand, _perm(order), x, 0)
    o = CARD_BITS  # 92
    x[o] = hand[BJ] / 2.0
    x[o + 1] = hand[SJ] / 2.0
    x[o + 2] = (hand[R7] + hand[11 + R7] + hand[22 + R7] + hand[33 + R7]) / 8.0
    x[o + 3] = (hand[R2] + hand[11 + R2] + hand[22 + R2] + hand[33 + R2]) / 8.0
    o += 4  # 96
    for i, su in enumerate(order):
        ts = SIDE_NON_CZ[su]
        x[o + i] = sum(hand[t] for t in ts) / 18.0
        x[o + 4 + i] = sum(1 for t in ts if hand[t] == 2) / 9.0
        x[o + 8 + i] = hand[su * 11 + RA] / 2.0
    o += 12  # 108
    cz = sum(hand[t] for t in CHANGZHU_TYPES)
    cz_pairs = sum(1 for t in CHANGZHU_TYPES if hand[t] == 2)
    x[o] = min(_hand_points(hand), 120) / 120.0
    x[o + 1] = cz / 20.0
    x[o + 2] = cz_pairs / 10.0
    x[o + 3] = 1.0 if cz >= g.rules.force_changzhu else 0.0
    o += 4  # 112
    x[o + (g.first_bidder - s) % 4] = 1.0
    o += 4  # 116
    min_bid = getattr(g, "seat_min_bid", [0, 0, 0, 0])
    pu = getattr(g, "passed_unforced", [False, False, False, False])
    for seat in range(4):
        r = (seat - s) % 4
        x[o + 4 * r] = 1.0 if g.passed[seat] else 0.0
        x[o + 4 * r + 1] = min_bid[seat] / 80.0
        x[o + 4 * r + 2] = 1.0 if g.best_bidder == seat else 0.0
        x[o + 4 * r + 3] = 1.0 if pu[seat] else 0.0
    o += 16  # 132
    x[o] = (g.best_bid or 0) / 80.0
    x[o + 1] = 1.0 if g._forced(s) else 0.0
    x[o + 2] = min(len(g.bid_history), 20) / 20.0
    x[o + 3] = 1.0 if g.best_bid is None else 0.0
    x[o + 4] = sum(1 for p in g.passed if not p) / 4.0
    np.clip(x, 0.0, 1.0, out=x)
    return x


def bid_actions_v2(g, acts):
    vals = g.rules.bid_values()
    val_idx = {v: idx for idx, v in enumerate(vals)}
    a = np.zeros((len(acts), BID_ACTION_V2), np.float32)
    ref = g.best_bid if g.best_bid is not None else (g.rules.max_bid + g.rules.bid_step)
    sur_th = g.rules.surrender_threshold
    for i, v in enumerate(acts):
        if v == PASS:
            a[i, 16] = 1.0
        else:
            a[i, min(val_idx.get(v, 15), 15)] = 1.0
            a[i, 17] = v / 80.0
            a[i, 18] = 1.0 if v >= sur_th else 0.0
            a[i, 19] = min(max(ref - v, 0), 80) / 80.0
    np.clip(a, 0.0, 1.0, out=a)
    return a


def trump_state_v2(g, s):
    hand = g.hands[s]
    x = np.zeros(TRUMP_STATE_V2, np.float32)
    enc(hand, _perm(length_order(hand)), x, 0)
    o = CARD_BITS  # 92
    cz = sum(hand[t] for t in CHANGZHU_TYPES)
    x[o] = g.contract / 80.0
    x[o + 1] = 1.0 if g.contract >= g.rules.surrender_threshold else 0.0
    x[o + 2] = cz / 20.0
    x[o + 3] = hand[BJ] / 2.0
    x[o + 4] = hand[SJ] / 2.0
    x[o + 5] = (hand[R7] + hand[11 + R7] + hand[22 + R7] + hand[33 + R7]) / 8.0
    x[o + 6] = (hand[R2] + hand[11 + R2] + hand[22 + R2] + hand[33 + R2]) / 8.0
    x[o + 7] = min(_hand_points(hand), 150) / 150.0
    o += 8  # 100
    min_bid = getattr(g, "seat_min_bid", [0, 0, 0, 0])
    sbids = getattr(g, "seat_bids", [0, 0, 0, 0])
    pu = getattr(g, "passed_unforced", [False, False, False, False])
    for r in range(1, 4):
        seat = (s + r) % 4
        x[o + 4 * (r - 1)] = 1.0 if sbids[seat] > 0 else 0.0
        x[o + 4 * (r - 1) + 1] = min_bid[seat] / 80.0
        x[o + 4 * (r - 1) + 2] = 1.0 if pu[seat] else 0.0
        x[o + 4 * (r - 1) + 3] = min(sbids[seat], 4) / 4.0
    np.clip(x, 0.0, 1.0, out=x)
    return x


def trump_actions_v2(g, s, acts):
    hand = g.hands[s]
    a = np.zeros((len(acts), TRUMP_ACTION_V2), np.float32)
    sur_cost = (1.0 / 3.0) if g.contract >= g.rules.surrender_threshold else (2.0 / 3.0)
    for i, t in enumerate(acts):
        if t == SURRENDER:
            a[i, CARD_BITS + 2] = 1.0
            a[i, CARD_BITS + 3 + 7] = sur_cost
            continue
        ctx, perm, _, cls_types, high_trumps, _ = _ctx_fast(t)
        enc(hand, perm, a[i], 0)
        a[i, CARD_BITS + (1 if t == NO_TRUMP else 0)] = 1.0
        o = CARD_BITS + 3  # 95
        tr_types = cls_types[TRUMP_CLASS]
        ntr = sum(hand[u] for u in tr_types)
        tr_pairs = sum(1 for u in tr_types if hand[u] == 2)
        side_lens = []
        voids_now = 0
        side_aces = 0
        side_pairs = 0
        for c in range(4):
            cts = cls_types[c]
            if not cts:
                continue
            ln = sum(hand[u] for u in cts)
            if ln == 0:
                voids_now += 1
            else:
                # non-Ace cards needed to bury to void this suit
                non_ace = sum(hand[u] for u in cts if rank_of(u) != RA)
                if non_ace == ln:
                    side_lens.append(ln)
            side_aces += hand[make(c, RA)]
            side_pairs += sum(1 for u in cts if hand[u] == 2)
        side_lens.sort()
        rem = 8
        voids_after = voids_now
        for ln in side_lens:
            if ln <= rem:
                rem -= ln
                voids_after += 1
        a[i, o] = ntr / 29.0
        a[i, o + 1] = tr_pairs / 14.0
        a[i, o + 2] = voids_after / 4.0
        a[i, o + 3] = voids_now / 4.0
        a[i, o + 4] = sum(hand[u] for u in high_trumps) / 20.0
        a[i, o + 5] = side_aces / 8.0
        a[i, o + 6] = side_pairs / 10.0
        a[i, o + 7] = 0.0
        if t != NO_TRUMP:
            a[i, o + 8] = hand[make(t, R7)] / 2.0
            a[i, o + 9] = hand[make(t, R2)] / 2.0
    np.clip(a, 0.0, 1.0, out=a)
    return a


def bury_state_v2(g, s):
    hand = g.hands[s]
    ctx, perm, slot, cls_types, _, _ = _ctx_fast(g.trump)
    x = np.zeros(BURY_STATE_V2, np.float32)
    enc(hand, perm, x, 0)
    _enc_move(sorted(g.buried), perm, x, CARD_BITS)
    o = 2 * CARD_BITS  # 184
    tr_types = cls_types[TRUMP_CLASS]
    ntr = sum(hand[u] for u in tr_types)
    tr_pairs = sum(1 for u in tr_types if hand[u] == 2)
    nb = len(g.buried)
    x[o] = g.contract / 80.0
    x[o + 1] = 1.0 if g.contract >= g.rules.surrender_threshold else 0.0
    x[o + 2] = 1.0 if g.trump == NO_TRUMP else 0.0
    x[o + 3] = nb / 8.0
    x[o + 4] = (8 - nb) / 8.0
    x[o + 5] = ntr / 29.0
    x[o + 6] = tr_pairs / 14.0
    x[o + 7] = sum(POINTS[t] for t in g.buried) / 80.0
    x[o + 8] = min(_hand_points(hand), 150) / 150.0
    side_BySlot = [0] * 4
    voids = 0
    for c in range(4):
        cts = cls_types[c]
        if not cts:
            continue
        ln = sum(hand[u] for u in cts)
        side_BySlot[slot[c]] = ln
        if ln == 0:
            voids += 1
    for j in range(4):
        x[o + 9 + j] = side_BySlot[j] / 18.0
    min_bid = getattr(g, "seat_min_bid", [0, 0, 0, 0])
    opp_bids = [min_bid[u] for u in range(4) if u != s and min_bid[u] > 0]
    x[o + 13] = (min(opp_bids) / 80.0) if opp_bids else 1.0
    x[o + 14] = 1.0 if opp_bids else 0.0
    x[o + 15] = voids / 4.0
    np.clip(x, 0.0, 1.0, out=x)
    return x


def bury_actions_v2(g, s, acts):
    hand = g.hands[s]
    ctx, _, _, cls_types, _, _ = _ctx_fast(g.trump)
    canon = ctx.canon
    lens = [sum(hand[u] for u in cls_types[c]) for c in range(5)]
    rem = 8 - len(g.buried)
    has_a = [hand[make(c, RA)] > 0 for c in range(4)]
    a = np.zeros((len(acts), BURY_ACTION_V2), np.float32)
    for i, t in enumerate(acts):
        a[i, canon[t]] = 1.0
        c = ctx.cls[t]
        ist = ctx.is_trump[t]
        pts = POINTS[t]
        ln = lens[c]
        rk = rank_of(t)
        a[i, 46] = 1.0 if ist else 0.0
        a[i, 47] = pts / 10.0
        a[i, 48] = ln / 29.0
        a[i, 49] = 1.0 if (not ist and ln == 1) else 0.0
        a[i, 50] = 1.0 if (not ist and ln <= rem) else 0.0
        a[i, 51] = ctx.pos[t] / 15.0
        a[i, 52] = 1.0 if hand[t] == 2 else 0.0
        a[i, 53] = 1.0 if (not ist and rk == RA) else 0.0
        a[i, 54] = 1.0 if (not ist and rk == RK and has_a[c]) else 0.0
        a[i, 55] = 1.0 if pts == 10 else 0.0
        a[i, 56] = 1.0 if pts == 5 else 0.0
        a[i, 57] = 1.0 if pts == 0 else 0.0
    np.clip(a, 0.0, 1.0, out=a)
    return a


def play_state_v2(g, s):
    ctx, perm, slot, cls_types, _, _ = _ctx_fast(g.trump)
    hand = g.hands[s]
    x = np.zeros(PLAY_STATE_V2, np.float32)
    enc(hand, perm, x, 0)
    o = CARD_BITS  # 92
    for r in range(4):
        enc(g.played[(s + r) % 4], perm, x, o + r * CARD_BITS)
    o += 4 * CARD_BITS  # 460
    cur_map = {seat: cards for seat, cards in g.cur}
    for r in range(1, 4):
        seat = (s + r) % 4
        c_mv = cur_map.get(seat)
        if c_mv is not None:
            _enc_move(c_mv, perm, x, o + (r - 1) * CARD_BITS)
    o += 3 * CARD_BITS  # 736
    unseen = g.unseen_for(s)
    enc(unseen, perm, x, o)
    o += CARD_BITS  # 828
    if s == g.banker and g.buried:
        _enc_move(sorted(g.buried), perm, x, o)
    o += CARD_BITS  # 920
    x[o + (g.banker - s) % 4] = 1.0
    o += 4  # 924
    # suit/trump voids (15)
    for r in range(1, 4):
        v = g.void[(s + r) % 4]
        base = o + (r - 1) * 5
        for c in range(5):
            if v[c]:
                x[base + slot[c]] = 1.0
    o += 15  # 939
    # pair voids (20: r=0 from own hand, r=1..3 from g.pair_void)
    pv = getattr(g, "pair_void", None)
    for c in range(5):
        cts = cls_types[c]
        if cts and not any(hand[u] == 2 for u in cts):
            x[o + slot[c]] = 1.0
    if pv is not None:
        for r in range(1, 4):
            pvr = pv[(s + r) % 4]
            base = o + r * 5
            for c in range(5):
                if pvr[c]:
                    x[base + slot[c]] = 1.0
    o += 20  # 959
    # bidding memory for other 3 seats (12)
    min_bid = getattr(g, "seat_min_bid", [0, 0, 0, 0])
    sbids = getattr(g, "seat_bids", [0, 0, 0, 0])
    pu = getattr(g, "passed_unforced", [False, False, False, False])
    for r in range(1, 4):
        seat = (s + r) % 4
        base = o + (r - 1) * 4
        x[base] = 1.0 if sbids[seat] > 0 else 0.0
        x[base + 1] = min_bid[seat] / 80.0
        x[base + 2] = 1.0 if pu[seat] else 0.0
        x[base + 3] = 1.0 if seat == g.banker else 0.0
    o += 12  # 971
    # 56 rich tactical & threshold scalars -> total 1027
    B = g.contract
    E = g.xian_points
    tp = g.trick_points()
    hsz = g.hand_size(s)
    is_bk = s == g.banker
    x[o] = B / 80.0
    x[o + 1] = min(E, 200) / 200.0
    x[o + 2] = min(tp, 100) / 100.0
    x[o + 3] = len(g.tricks) / 21.0
    x[o + 4] = hsz / 21.0
    x[o + 5] = 1.0 if g.trump == NO_TRUMP else 0.0
    x[o + 6] = (sum(POINTS[t] for t in g.buried) / 80.0) if is_bk else 0.0
    for r in range(1, 4):
        x[o + 6 + r] = g.hand_size((s + r) % 4) / 21.0
    need = (B - E) / 200.0
    x[o + 10] = min(max(need, -1.0), 1.0) * 0.5 + 0.5
    ncur = len(g.cur)
    if ncur:
        x[o + 11 + (g.cur[0][0] - s) % 4] = 1.0
        w = g.current_winner()
        x[o + 15 + (w - s) % 4] = 1.0
        x[o + 19] = 1.0 if ((w == g.banker) == is_bk) else 0.0
        lead = g.lead_shape
        x[o + 20 + slot[lead.cls]] = 1.0
        x[o + 25 + lead.kind] = 1.0
        x[o + 29] = lead.n / 21.0
    x[o + 30] = 1.0 if is_bk else 0.0
    x[o + 31 + min(ncur, 3)] = 1.0  # seat order in trick (0=lead .. 3=closing 4th seat!)
    req_n = g.lead_shape.n if ncur else 1
    x[o + 35] = 1.0 if hsz <= req_n else 0.0
    x[o + 36] = 1.0 if hsz <= 5 else 0.0
    # settlement step-function awareness
    xg_max = g.rules.xiao_guang_max
    xd_m = g.rules.xiao_dao_margin
    dd_m = g.rules.da_dao_margin
    x[o + 37] = 1.0 if E == 0 else 0.0
    x[o + 38] = 1.0 if E <= xg_max else 0.0
    x[o + 39] = 1.0 if E < B else 0.0
    x[o + 40] = 1.0 if E < B + xd_m else 0.0
    x[o + 41] = 1.0 if E < B + dd_m else 0.0
    x[o + 42] = min(max(xg_max + 5 - E, 0), 35) / 35.0
    x[o + 43] = min(max(B - E, 0), 60) / 60.0
    x[o + 44] = min(max(B + xd_m - E, 0), 60) / 60.0
    x[o + 45] = min(max(B + dd_m - E, 0), 60) / 60.0
    my_pts = _hand_points(hand)
    unseen_pts = sum(max(0, unseen[t]) * p for t, p in POINT_TYPES)
    x[o + 46] = min(my_pts, 100) / 100.0
    x[o + 47] = min(unseen_pts, 200) / 200.0
    x[o + 48] = 1.0 if (E + tp + my_pts + unseen_pts) >= B else 0.0
    tr_types = cls_types[TRUMP_CLASS]
    my_tr = sum(hand[u] for u in tr_types)
    my_tr_p = sum(1 for u in tr_types if hand[u] == 2)
    un_tr = sum(max(0, unseen[u]) for u in tr_types)
    un_tr_p = sum(1 for u in tr_types if unseen[u] >= 2)
    x[o + 49] = my_tr / 21.0
    x[o + 50] = my_tr_p / 10.0
    x[o + 51] = un_tr / 38.0
    x[o + 52] = un_tr_p / 19.0
    opps = [u for u in range(4) if u != s and ((u == g.banker) != is_bk)]
    all_opp_tr_void = all(g.void[u][TRUMP_CLASS] or g.hand_size(u) == 0 for u in opps)
    x[o + 53] = 1.0 if (un_tr == 0 or all_opp_tr_void) else 0.0
    rem_seats = [(s + k) % 4 for k in range(1, 4 - ncur)]
    opp_rem = [u for u in rem_seats if (u == g.banker) != is_bk]
    x[o + 54] = 1.0 if opp_rem else 0.0
    if ncur:
        lc = g.lead_shape.cls
        can_contest = any(
            not g.void[u][lc] or (lc != TRUMP_CLASS and not g.void[u][TRUMP_CLASS])
            for u in opp_rem
        )
        x[o + 55] = 1.0 if can_contest else 0.0
    else:
        x[o + 55] = 1.0 if opp_rem else 0.0
    np.clip(x, 0.0, 1.0, out=x)
    return x


def play_actions_v2(g, s, acts):
    ctx, perm, _, cls_types, _, higher_in_cls = _ctx_fast(g.trump)
    hand = g.hands[s]
    unseen = g.unseen_for(s)
    a = np.zeros((len(acts), PLAY_ACTION_V2), np.float32)
    lead = g.lead_shape
    ncur = len(g.cur)
    is_bk = s == g.banker
    hsz = g.hand_size(s)
    tp = g.trick_points()
    best_pow = None
    team_winning_now = False
    if ncur:
        bi = M.trick_winner(g.cur, ctx)
        best_pow = M.power(lead, g.cur[bi][1], ctx, is_lead=(bi == 0))
        win_seat = g.cur[bi][0]
        team_winning_now = (win_seat == g.banker) == is_bk

    rem_seats = [(s + k) % 4 for k in range(1, 4 - ncur)]
    opp_rem = [u for u in rem_seats if (u == g.banker) != is_bk]
    all_opps = [u for u in range(4) if u != s and ((u == g.banker) != is_bk)]
    pv = getattr(g, "pair_void", None)
    cls_hand_len = [sum(hand[u] for u in cls_types[c]) for c in range(5)]

    # Precompute opponent void / pair-void status per class
    opp_any_void = [any(g.void[u][c] for u in all_opps) for c in range(5)]
    opp_all_pvoid = [
        pv is not None and all(pv[u][c] or g.void[u][c] for u in all_opps)
        for c in range(5)
    ]
    opp_rem_can_ruff = any(not g.void[u][TRUMP_CLASS] for u in opp_rem)

    for i, mv in enumerate(acts):
        _enc_move(mv, perm, a[i], 0)
        o = CARD_BITS  # 92
        sh = M.analyze(mv, ctx)
        k = sh.kind
        a[i, o + k] = 1.0
        all_trump = sh.cls == TRUMP_CLASS
        mv_pts = M.points_of(mv)
        n_mv = len(mv)

        beats = False
        if lead is not None:
            p = M.power(lead, mv, ctx)
            beats = p is not None and p > best_pow
            a[i, o + 4] = 1.0 if beats else 0.0
            a[i, o + 5] = 1.0 if (all_trump and lead.cls != TRUMP_CLASS) else 0.0
            a[i, o + 6] = 1.0 if sh.cls == lead.cls else 0.0
        else:
            a[i, o + 7] = 1.0
            beats = True

        a[i, o + 8] = min(mv_pts, 100) / 100.0
        a[i, o + 9] = n_mv / 21.0
        a[i, o + 10] = 1.0 if all_trump else 0.0
        a[i, o + 11] = sh.top / 15.0

        # Boss / higher-unseen analysis in sh.cls
        is_boss = False
        higher_cnt = 0
        if sh.cls >= 0:
            if k == M.THROW:
                is_boss = True
            elif k == M.SINGLE:
                ht = higher_in_cls[(sh.cls, sh.top)]
                higher_cnt = sum(max(0, unseen[u]) for u in ht)
                is_boss = higher_cnt == 0 or all(g.void[u][sh.cls] for u in opp_rem)
            elif k == M.PAIR:
                ht = higher_in_cls[(sh.cls, sh.top)]
                higher_cnt = sum(1 for u in ht if unseen[u] >= 2)
                is_boss = higher_cnt == 0 or (
                    pv is not None and all(pv[u][sh.cls] or g.void[u][sh.cls] for u in opp_rem)
                )
            elif k == M.TRACTOR:
                u_pairs = [u for u in cls_types[sh.cls] if unseen[u] >= 2 and ctx.pos[u] < sh.top + sh.pairs - 1]
                higher_cnt = sum(
                    1 for tr in M.tractors_from_pairs(u_pairs, ctx, sh.pairs) if ctx.pos[tr[0]] < sh.top
                )
                is_boss = higher_cnt == 0 or (
                    pv is not None and all(pv[u][sh.cls] or g.void[u][sh.cls] for u in opp_rem)
                )
        a[i, o + 12] = 1.0 if is_boss else 0.0
        a[i, o + 13] = min(higher_cnt, 10) / 10.0

        # Guaranteed team trick win check
        team_wins_so_far = beats or team_winning_now
        if not opp_rem:
            guar_win = team_wins_so_far
        elif beats and is_boss:
            eff_cls = sh.cls if lead is None else (TRUMP_CLASS if all_trump else lead.cls)
            if eff_cls == TRUMP_CLASS:
                guar_win = True
            else:
                # Side-suit boss can only be beaten if a remaining opponent is void in eff_cls (or has < n_mv cards) and can ruff
                opp_might_ruff = opp_rem_can_ruff and any(
                    g.void[u][eff_cls] or sum(max(0, unseen[t]) for t in cls_types[eff_cls]) < n_mv
                    for u in opp_rem
                )
                guar_win = not opp_might_ruff and not opp_rem_can_ruff
        else:
            guar_win = False

        a[i, o + 14] = 1.0 if guar_win else 0.0
        a[i, o + 15] = (min(tp + mv_pts, 100) / 100.0) if guar_win else 0.0
        a[i, o + 16] = (min(mv_pts, 50) / 50.0) if not team_wins_so_far else 0.0

        # Void creation & pair-breaking checks
        if n_mv == 1:
            t0 = mv[0]
            c0 = ctx.cls[t0]
            creates_void = (c0 != TRUMP_CLASS) and (cls_hand_len[c0] == 1)
            breaks_pair = hand[t0] == 2
            breaks_tr_pair = breaks_pair and (c0 == TRUMP_CLASS)
        else:
            mc = M.counts_of(mv)
            creates_void = False
            cls_used = {}
            for t0, cnt0 in mc.items():
                c0 = ctx.cls[t0]
                if c0 != TRUMP_CLASS:
                    cls_used[c0] = cls_used.get(c0, 0) + cnt0
            for c0, used0 in cls_used.items():
                if used0 == cls_hand_len[c0] and used0 > 0:
                    creates_void = True
                    break
            breaks_pair = any(hand[t0] == 2 and cnt0 == 1 for t0, cnt0 in mc.items())
            breaks_tr_pair = any(ctx.is_trump[t0] and hand[t0] == 2 and cnt0 == 1 for t0, cnt0 in mc.items())

        a[i, o + 17] = 1.0 if creates_void else 0.0
        a[i, o + 18] = 1.0 if breaks_pair else 0.0
        a[i, o + 19] = 1.0 if breaks_tr_pair else 0.0
        a[i, o + 20] = 1.0 if n_mv == hsz else 0.0
        if all_trump:
            eff_sh = sh if lead is None else lead
            if eff_sh.kind == M.SINGLE:
                km = g.rules.koudi_single
            elif eff_sh.kind == M.PAIR:
                km = g.rules.koudi_pair
            elif eff_sh.kind == M.TRACTOR:
                km = g.rules.koudi_per_tractor_pair * eff_sh.pairs
            else:
                km = eff_sh.n if g.rules.koudi_throw_per_card else g.rules.koudi_pair
            a[i, o + 21] = min(km, 8) / 8.0
        if lead is None and sh.cls >= 0 and sh.cls != TRUMP_CLASS:
            a[i, o + 22] = 1.0 if opp_any_void[sh.cls] else 0.0
        if lead is None and sh.cls >= 0 and k in (M.PAIR, M.TRACTOR):
            a[i, o + 23] = 1.0 if opp_all_pvoid[sh.cls] else 0.0

    np.clip(a, 0.0, 1.0, out=a)
    return a


def encode(g, s, acts, feat_ver: int = 1):
    """Return (phase, state_vec, action_matrix) for the decision at hand."""
    ph = g.phase
    if feat_ver >= 2:
        if ph == "bid":
            return ph, bid_state_v2(g, s), bid_actions_v2(g, acts)
        if ph == "trump":
            return ph, trump_state_v2(g, s), trump_actions_v2(g, s, acts)
        if ph == "bury":
            return ph, bury_state_v2(g, s), bury_actions_v2(g, s, acts)
        return ph, play_state_v2(g, s), play_actions_v2(g, s, acts)
    if ph == "bid":
        return ph, bid_state(g, s), bid_actions(g, acts)
    if ph == "trump":
        return ph, trump_state(g, s), trump_actions(g, s, acts)
    if ph == "bury":
        return ph, bury_state(g, s), bury_actions(g, s, acts)
    return ph, play_state(g, s), play_actions(g, s, acts)
