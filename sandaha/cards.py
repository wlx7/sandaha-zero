"""Card constants and trump-dependent orderings.

Two decks with all 3s and 4s removed: 92 cards, 46 distinct card *types*, two
copies of each.  A hand is represented as a list of 46 counts (0, 1 or 2).

Type id: ``suit * 11 + rank`` for suited cards, 44 = small joker, 45 = big joker.
"""
from __future__ import annotations

from functools import lru_cache

SUITS = ("S", "H", "C", "D")
SUIT_SYMBOLS = ("♠", "♥", "♣", "♦")
SUIT_NAMES_ZH = ("黑桃", "红桃", "梅花", "方块")
RANKS = ("2", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
R2, R5, R6, R7, R8, R9, R10, RJ, RQ, RK, RA = range(11)
SJ = 44  # 小王
BJ = 45  # 大王
NUM_TYPES = 46
NO_TRUMP = 4  # 无主
TRUMP_CLASS = 4  # "suit class" of every trump card

# Side-suit order, strongest first (7 and 2 are always trump, 3/4 removed).
SIDE_ORDER = (RA, RK, RQ, RJ, R10, R9, R8, R6, R5)
SIDE_POS = {r: i for i, r in enumerate(SIDE_ORDER)}


def suit_of(t: int) -> int:
    return -1 if t >= 44 else t // 11


def rank_of(t: int) -> int:
    return -1 if t >= 44 else t % 11


def make(suit: int, rank: int) -> int:
    return suit * 11 + rank


POINTS = [0] * NUM_TYPES
for _s in range(4):
    POINTS[make(_s, R5)] = 5
    POINTS[make(_s, R10)] = 10
    POINTS[make(_s, RK)] = 10
TOTAL_POINTS = 2 * sum(POINTS)  # 200


def card_name(t: int) -> str:
    if t == BJ:
        return "大王"
    if t == SJ:
        return "小王"
    return SUIT_SYMBOLS[suit_of(t)] + RANKS[rank_of(t)]


def card_code(t: int) -> str:
    """Short ASCII code used by the web client, e.g. 'S10', 'HA', 'LJ' (小王), 'BJ' (大王)."""
    if t == BJ:
        return "BJ"
    if t == SJ:
        return "LJ"
    return SUITS[suit_of(t)] + RANKS[rank_of(t)]


CODE_TO_TYPE = {card_code(t): t for t in range(NUM_TYPES)}


def full_deck() -> list[int]:
    deck = []
    for t in range(NUM_TYPES):
        deck += [t, t]
    return deck


def is_changzhu(t: int) -> bool:
    """常主: jokers, all 7s and all 2s."""
    return t >= 44 or rank_of(t) in (R7, R2)


class TrumpContext:
    """Per-trump lookup tables.

    cls[t]  : TRUMP_CLASS for trump cards, otherwise the suit
    pos[t]  : position inside its class, 0 = strongest; equal positions are
              equal strength (e.g. two different 副7)
    canon[t]: canonical feature index (trump suit first) used by the networks
    """

    __slots__ = ("trump", "cls", "pos", "canon", "suit_order", "is_trump", "class_size")

    def __init__(self, trump: int, suit_order: tuple[int, ...] | None = None):
        self.trump = trump
        cls = [0] * NUM_TYPES
        pos = [0] * NUM_TYPES
        for t in range(NUM_TYPES):
            if t == BJ:
                cls[t], pos[t] = TRUMP_CLASS, 0
            elif t == SJ:
                cls[t], pos[t] = TRUMP_CLASS, 1
            else:
                s, r = suit_of(t), rank_of(t)
                if trump == NO_TRUMP:
                    if r == R7:
                        cls[t], pos[t] = TRUMP_CLASS, 2
                    elif r == R2:
                        cls[t], pos[t] = TRUMP_CLASS, 3
                    else:
                        cls[t], pos[t] = s, SIDE_POS[r]
                else:
                    if r == R7:
                        cls[t], pos[t] = TRUMP_CLASS, (2 if s == trump else 3)
                    elif r == R2:
                        cls[t], pos[t] = TRUMP_CLASS, (4 if s == trump else 5)
                    elif s == trump:
                        cls[t], pos[t] = TRUMP_CLASS, 6 + SIDE_POS[r]
                    else:
                        cls[t], pos[t] = s, SIDE_POS[r]
        self.cls = cls
        self.pos = pos
        self.is_trump = [c == TRUMP_CLASS for c in cls]
        self.class_size = [sum(1 for t in range(NUM_TYPES) if cls[t] == c) for c in range(5)]
        if suit_order is None:
            if trump == NO_TRUMP:
                suit_order = (0, 1, 2, 3)
            else:
                suit_order = (trump,) + tuple(s for s in range(4) if s != trump)
        self.suit_order = suit_order
        self.canon = canonical_index(suit_order)


def canonical_index(suit_order: tuple[int, ...]) -> list[int]:
    """Map card types to a fixed 46-slot layout.

    [大王, 小王, 7(s0), 7(s1..s3), 2(s0), 2(s1..s3), s0:A..5, s1:A..5, s2:A..5, s3:A..5]
    where s0 is the trump suit (or the first suit of ``suit_order``).
    """
    idx = [0] * NUM_TYPES
    idx[BJ] = 0
    idx[SJ] = 1
    for i, s in enumerate(suit_order):
        idx[make(s, R7)] = 2 + i
        idx[make(s, R2)] = 6 + i
        for r in SIDE_ORDER:
            idx[make(s, r)] = 10 + 9 * i + SIDE_POS[r]
    return idx


@lru_cache(maxsize=None)
def trump_context(trump: int) -> TrumpContext:
    return TrumpContext(trump)


def sort_key(ctx: TrumpContext | None, t: int):
    """Display sort key: trump first (strongest first), then suits."""
    if ctx is None:
        if t >= 44:
            return (0, 45 - t)
        r = rank_of(t)
        if r in (R7, R2):
            return (1, 0 if r == R7 else 1, suit_of(t))
        return (2 + suit_of(t), SIDE_POS[r])
    c = ctx.cls[t]
    return (0 if c == TRUMP_CLASS else 1 + c, ctx.pos[t], suit_of(t))
