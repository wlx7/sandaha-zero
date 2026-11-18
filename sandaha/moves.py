"""Play shapes, legal-move generation and trick comparison.

A *play* is a sorted tuple of card types.  Hands are lists of 46 counts.
"""
from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations

from .cards import NUM_TYPES, POINTS, TRUMP_CLASS, TrumpContext

SINGLE, PAIR, TRACTOR, THROW = 0, 1, 2, 3
KIND_NAMES = {SINGLE: "单张", PAIR: "对子", TRACTOR: "拖拉机", THROW: "甩牌"}
MOVE_CAP = 96  # max candidate moves handed to the AI for one decision


class Shape:
    __slots__ = ("kind", "cls", "n", "pairs", "top", "components")

    def __init__(self, kind, cls, n, pairs, top, components=None):
        self.kind = kind      # SINGLE / PAIR / TRACTOR / THROW
        self.cls = cls        # class (suit or TRUMP_CLASS), -1 if mixed classes
        self.n = n            # number of cards
        self.pairs = pairs    # number of pairs contained (tractor k, pair 1, throw: total)
        self.top = top        # position of strongest card (0 = strongest)
        self.components = components  # for THROW: list of (kind, cards)

    def __repr__(self):
        return f"Shape({KIND_NAMES[self.kind]}, cls={self.cls}, n={self.n}, pairs={self.pairs}, top={self.top})"


def counts_of(cards) -> dict:
    d = {}
    for t in cards:
        d[t] = d.get(t, 0) + 1
    return d


def tractors_from_pairs(pair_types, ctx: TrumpContext, k: int | None = None):
    """All tractors (tuples of pair types, strongest first) formable from pair_types.

    pair_types must all belong to one class.  If k is given only length-k tractors
    are returned, otherwise every length >= 2.
    """
    bypos = defaultdict(list)
    for t in pair_types:
        bypos[ctx.pos[t]].append(t)
    out = []

    def dfs(p, chain):
        if len(chain) >= 2 and (k is None or len(chain) == k):
            out.append(tuple(chain))
        if k is not None and len(chain) >= k:
            return
        nxt = bypos.get(p + 1)
        if nxt:
            for t in nxt:
                chain.append(t)
                dfs(p + 1, chain)
                chain.pop()

    for p in sorted(bypos):
        for t in bypos[p]:
            dfs(p, [t])
    return out


def decompose(cnt: dict, ctx: TrumpContext):
    """Greedy split of a single-class multiset into tractors, pairs and singles."""
    cnt = dict(cnt)
    comps = []
    while True:
        pairs = [t for t, c in cnt.items() if c == 2]
        trs = tractors_from_pairs(pairs, ctx)
        if not trs:
            break
        best = max(trs, key=lambda tr: (len(tr), -ctx.pos[tr[0]]))
        comps.append((TRACTOR, tuple(sorted(t for t in best for _ in (0, 1)))))
        for t in best:
            del cnt[t]
    for t in sorted(cnt, key=lambda x: ctx.pos[x]):
        if cnt[t] == 2:
            comps.append((PAIR, (t, t)))
    for t in sorted(cnt, key=lambda x: ctx.pos[x]):
        if cnt[t] == 1:
            comps.append((SINGLE, (t,)))
    return comps


def analyze(cards, ctx: TrumpContext) -> Shape:
    n = len(cards)
    classes = {ctx.cls[t] for t in cards}
    cls = classes.pop() if len(classes) == 1 else -1
    top = min(ctx.pos[t] for t in cards)
    if n == 1:
        return Shape(SINGLE, cls, 1, 0, top)
    cnt = counts_of(cards)
    if n == 2 and len(cnt) == 1:
        return Shape(PAIR, cls, 2, 1, top)
    if cls >= 0 and n >= 4 and all(c == 2 for c in cnt.values()):
        ps = sorted(ctx.pos[t] for t in cnt)
        if all(ps[i + 1] == ps[i] + 1 for i in range(len(ps) - 1)):
            return Shape(TRACTOR, cls, n, n // 2, top)
    comps = decompose(cnt, ctx) if cls >= 0 else None
    pairs = sum(1 for c in cnt.values() if c == 2)
    return Shape(THROW, cls, n, pairs, top, comps)


def _hand_class(hand, ctx, c):
    return [t for t in range(NUM_TYPES) if hand[t] and ctx.cls[t] == c]


# ---------------------------------------------------------------- leading ---

def component_beatable(kind, cards, top, hand, ctx) -> bool:
    """Can a player holding ``hand`` beat this trump component?"""
    if kind == SINGLE:
        return any(hand[u] and ctx.cls[u] == TRUMP_CLASS and ctx.pos[u] < top for u in range(NUM_TYPES))
    pairs = [u for u in range(NUM_TYPES) if hand[u] == 2 and ctx.cls[u] == TRUMP_CLASS]
    if kind == PAIR:
        return any(ctx.pos[u] < top for u in pairs)
    k = len(cards) // 2
    return any(ctx.pos[tr[0]] < top for tr in tractors_from_pairs(pairs, ctx, k))


def safe_throw(hand, unseen, ctx, opp_trump_pair: bool = True) -> tuple | None:
    """Largest trump throw whose every component is provably unbeatable given the
    cards opponents could possibly hold (``unseen`` counts, conservative)."""
    cnt = {t: hand[t] for t in range(NUM_TYPES) if hand[t] and ctx.cls[t] == TRUMP_CLASS}
    if not cnt:
        return None
    safe = []
    for kind, cards in decompose(cnt, ctx):
        top = min(ctx.pos[t] for t in cards)
        if kind != SINGLE and not opp_trump_pair:
            safe.append(cards)
        elif not component_beatable(kind, cards, top, unseen, ctx):
            safe.append(cards)
    if len(safe) < 2:
        return None
    return tuple(sorted(t for c in safe for t in c))


def lead_moves(hand, ctx: TrumpContext, unseen=None, allow_throw=True, opp_trump_pair: bool = True):
    moves = []
    for c in range(5):
        hc = _hand_class(hand, ctx, c)
        if not hc:
            continue
        for t in hc:
            moves.append((t,))
        pairs = [t for t in hc if hand[t] == 2]
        for t in pairs:
            moves.append((t, t))
        for tr in tractors_from_pairs(pairs, ctx):
            moves.append(tuple(sorted(t for t in tr for _ in (0, 1))))
    if allow_throw and unseen is not None:
        st = safe_throw(hand, unseen, ctx, opp_trump_pair=opp_trump_pair)
        if st is not None and st not in moves:
            moves.append(st)
    return moves


def check_lead(hand, cards, ctx, allow_throw=True) -> str | None:
    if not cards:
        return "请选择要出的牌"
    cnt = counts_of(cards)
    if any(hand[t] < c for t, c in cnt.items()):
        return "手中没有这些牌"
    sh = analyze(cards, ctx)
    if sh.cls < 0:
        return "首家出牌必须是同一花色（或全为主牌）"
    if sh.kind == THROW:
        if sh.cls != TRUMP_CLASS:
            return "本版本不允许甩副牌，只能出单张、对子或拖拉机"
        if not allow_throw:
            return "本局规则不允许甩牌"
    return None


def throw_result(cards, leader, hands, ctx):
    """Validate a trump throw against the other players' real hands.

    Returns (ok, forced_cards).  On failure the leader must play the failing
    component (the smallest one if several fail)."""
    sh = analyze(cards, ctx)
    failing = []
    for kind, comp in sh.components:
        top = min(ctx.pos[t] for t in comp)
        for s in range(4):
            if s != leader and component_beatable(kind, comp, top, hands[s], ctx):
                failing.append((len(comp), -top, comp))
                break
    if not failing:
        return True, cards
    failing.sort()
    return False, tuple(sorted(failing[0][2]))


# -------------------------------------------------------------- following ---

def _multiset_combos(items, f, limit):
    """Combinations of size f from [(type, count)], or None if more than limit."""
    out = []
    cur = []

    def rec(i, need):
        if need == 0:
            out.append(tuple(cur))
            return len(out) <= limit
        if i >= len(items):
            return True
        t, c = items[i]
        for take in range(min(c, need), -1, -1):
            cur.extend([t] * take)
            ok = rec(i + 1, need - take)
            del cur[len(cur) - take:]
            if not ok:
                return False
        return True

    if not rec(0, f):
        return None
    return out


def _weakness_key(t, ctx):
    # prefer to give away: side cards before trump, non-points before points, low before high
    return (ctx.is_trump[t], POINTS[t] > 0, -ctx.pos[t])


def _greedy_fill(items, f, key):
    pool = []
    for t, c in items:
        pool += [t] * c
    pool.sort(key=key)
    return tuple(pool[:f])


def _sample_fills(items, f, rng, n, ctx):
    pool = []
    cnt_map = {}
    cls_len = [0] * 5
    for t, c in items:
        pool += [t] * c
        cnt_map[t] = c
        cls_len[ctx.cls[t]] += c
    res = {_greedy_fill(items, f, lambda t: _weakness_key(t, ctx))}
    res.add(_greedy_fill(items, f, lambda t: (-POINTS[t], ctx.is_trump[t], -ctx.pos[t])))
    res.add(_greedy_fill(items, f, lambda t: (not ctx.is_trump[t], ctx.pos[t])))
    # tactical fills: preserve pairs, create short side-suit voids, feed points without breaking trump pairs
    res.add(_greedy_fill(items, f, lambda t: (ctx.is_trump[t], POINTS[t] > 0, cnt_map[t] == 2, -ctx.pos[t])))
    res.add(_greedy_fill(items, f, lambda t: (
        ctx.is_trump[t], POINTS[t] > 0, ctx.pos[t] == 0, cls_len[ctx.cls[t]], cnt_map[t] == 2, -ctx.pos[t]
    )))
    res.add(_greedy_fill(items, f, lambda t: (-POINTS[t], ctx.is_trump[t], cnt_map[t] == 2, -ctx.pos[t])))
    for _ in range(n * 3):
        if len(res) >= n:
            break
        res.add(tuple(rng.sample(pool, f)))
    return list(res)


def follow_moves(hand, lead: Shape, ctx: TrumpContext, rng=None, cap=MOVE_CAP):
    rng = rng or random
    c, n = lead.cls, lead.n
    hc = _hand_class(hand, ctx, c)
    nc = sum(hand[t] for t in hc)
    moves = []
    if nc >= n:
        pairs = [t for t in hc if hand[t] == 2]
        if lead.kind == SINGLE:
            return [(t,) for t in hc]
        if lead.kind == PAIR:
            if pairs:
                return [(t, t) for t in pairs]
            return [tuple(sorted(p)) for p in combinations(hc, 2)]
        if lead.kind == TRACTOR:
            k = lead.pairs
            trs = tractors_from_pairs(pairs, ctx, k)
            if trs:
                return [tuple(sorted(t for t in tr for _ in (0, 1))) for tr in trs]
            m = min(k, len(pairs))
        else:  # THROW
            m = min(lead.pairs, len(pairs))
        f = n - 2 * m
        seen = set()
        pair_choices = list(combinations(pairs, m))
        per = max(4, cap // max(1, len(pair_choices)))
        for chosen in pair_choices:
            rest = dict((t, hand[t]) for t in hc)
            for t in chosen:
                del rest[t]
            items = sorted(rest.items())
            fills = _multiset_combos(items, f, per)
            if fills is None:
                fills = _sample_fills(items, f, rng, per, ctx)
            base = [t for t in chosen for _ in (0, 1)]
            for fl in fills:
                mv = tuple(sorted(base + list(fl)))
                if mv not in seen:
                    seen.add(mv)
                    moves.append(mv)
        return moves
    # not enough cards of the led class: play all of them plus anything else
    base = [t for t in hc for _ in range(hand[t])]
    f = n - nc
    others = sorted((t, hand[t]) for t in range(NUM_TYPES) if hand[t] and ctx.cls[t] != c)
    fills = _multiset_combos(others, f, cap)
    if fills is None:
        fills = _sample_fills(others, f, rng, cap, ctx)
        # make sure every proper trump "毙" with the lead's shape is offered
        if nc == 0 and c != TRUMP_CLASS:
            tp = [t for t, cnt in others if ctx.cls[t] == TRUMP_CLASS and cnt == 2]
            if lead.kind == PAIR:
                fills += [(t, t) for t in tp]
            elif lead.kind == TRACTOR:
                fills += [tuple(t for t in tr for _ in (0, 1)) for tr in tractors_from_pairs(tp, ctx, lead.pairs)]
    seen = set()
    for fl in fills:
        mv = tuple(sorted(base + list(fl)))
        if mv not in seen:
            seen.add(mv)
            moves.append(mv)
    return moves


def check_follow(hand, lead: Shape, cards, ctx) -> str | None:
    if len(cards) != lead.n:
        return f"必须出 {lead.n} 张牌"
    cnt = counts_of(cards)
    if any(hand[t] < k for t, k in cnt.items()):
        return "手中没有这些牌"
    c = lead.cls
    hc = _hand_class(hand, ctx, c)
    nc = sum(hand[t] for t in hc)
    in_cls = sum(k for t, k in cnt.items() if ctx.cls[t] == c)
    if nc < lead.n:
        if in_cls != nc:
            return "必须先跟出所有同门牌" if c != TRUMP_CLASS else "必须先跟出所有主牌"
        return None
    if in_cls != lead.n:
        return "有同门牌必须跟同门" if c != TRUMP_CLASS else "有主牌必须跟主牌"
    pairs = [t for t in hc if hand[t] == 2]
    played_pairs = sum(1 for k in cnt.values() if k == 2)
    if lead.kind == PAIR and pairs and played_pairs < 1:
        return "有对子必须跟对子"
    if lead.kind == TRACTOR:
        if tractors_from_pairs(pairs, ctx, lead.pairs):
            sh = analyze(cards, ctx)
            if sh.kind != TRACTOR:
                return "有拖拉机必须跟拖拉机"
        elif played_pairs < min(lead.pairs, len(pairs)):
            return "有对子必须跟对子"
    if lead.kind == THROW and played_pairs < min(lead.pairs, len(pairs)):
        return "有对子必须跟对子"
    return None


# ---------------------------------------------------------------- tricks ---

def power(lead: Shape, cards, ctx, is_lead=False):
    """Comparable strength of a play inside a trick, or None if it cannot win.

    Returns (category, -top): higher tuple wins; ties go to the earlier play.
    """
    if is_lead:
        return (0, -lead.top) if lead.kind != THROW else (9, 0)
    if lead.kind == THROW:
        return None
    sh = analyze(cards, ctx)
    if sh.kind != lead.kind or sh.n != lead.n:
        return None
    if sh.cls == lead.cls:
        return (0, -sh.top)
    if sh.cls == TRUMP_CLASS and lead.cls != TRUMP_CLASS:
        return (1, -sh.top)
    return None


def trick_winner(plays, ctx):
    """plays: list of (seat, cards) in play order.  Returns index into plays."""
    lead = analyze(plays[0][1], ctx)
    best_i, best_p = 0, power(lead, plays[0][1], ctx, is_lead=True)
    for i in range(1, len(plays)):
        p = power(lead, plays[i][1], ctx)
        if p is not None and p > best_p:
            best_i, best_p = i, p
    return best_i


def points_of(cards) -> int:
    return sum(POINTS[t] for t in cards)
