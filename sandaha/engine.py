"""三打哈 game engine (one hand: bid -> trump -> bury -> play -> settle).

Seats 0..3 in play order (counter-clockwise); ``(s + 1) % 4`` is the 下家.
"""
from __future__ import annotations

import random

from . import moves as M
from .cards import (BJ, NO_TRUMP, NUM_TYPES, POINTS, SJ, TRUMP_CLASS, card_name, full_deck,
                    is_changzhu, trump_context)
from .rules import Rules

BID, TRUMP, BURY, PLAY, OVER = "bid", "trump", "bury", "play", "over"
PASS = 0
SURRENDER = 5
BURY_COUNT = 8
HAND_SIZE = 21

TRUMP_LABELS = ["♠ 黑桃", "♥ 红桃", "♣ 梅花", "♦ 方块", "无主", "投降"]


class IllegalMove(Exception):
    pass


class Game:
    def __init__(self, rules: Rules | None = None, seed=None, first_bidder: int = 0,
                 hands=None, bottom=None):
        self.rules = rules or Rules()
        self.rng = random.Random(seed)
        if hands is None:
            deck = full_deck()
            self.rng.shuffle(deck)
            hands = [deck[i * HAND_SIZE:(i + 1) * HAND_SIZE] for i in range(4)]
            bottom = deck[4 * HAND_SIZE:]
        self.initial_hands = [sorted(h) for h in hands]
        self.initial_bottom = sorted(bottom)
        self.hands = [[0] * NUM_TYPES for _ in range(4)]
        for s in range(4):
            for t in hands[s]:
                self.hands[s][t] += 1
        self.bottom = list(bottom)
        self.phase = BID
        self.first_bidder = first_bidder
        self.to_act = first_bidder
        self.passed = [False] * 4
        self.bid_history: list[tuple[int, int]] = []  # (seat, value) value 0 = pass
        self.best_bid: int | None = None
        self.best_bidder: int | None = None
        self.banker: int | None = None
        self.contract: int | None = None
        self.trump: int | None = None
        self.ctx = None
        self.buried: list[int] = []
        self.tricks: list[dict] = []
        self.cur: list[tuple[int, tuple]] = []
        self.lead_shape = None
        self.leader: int | None = None
        self.xian_points = 0
        self.played = [[0] * NUM_TYPES for _ in range(4)]
        self.played_total = [0] * NUM_TYPES
        self.void = [[False] * 5 for _ in range(4)]
        self.result: dict | None = None
        self.events: list[dict] = []   # human-readable log for the UI

    # ------------------------------------------------------------ helpers --
    def hand_list(self, s):
        h = self.hands[s]
        return [t for t in range(NUM_TYPES) for _ in range(h[t])]

    def hand_size(self, s):
        return sum(self.hands[s])

    def changzhu_count(self, s):
        h = self.hands[s]
        return sum(h[t] for t in range(NUM_TYPES) if is_changzhu(t))

    def is_xian(self, s):
        return self.banker is not None and s != self.banker

    def unseen_for(self, s):
        """Card counts seat s has not seen (other hands + unknown bottom)."""
        u = [2 - self.hands[s][t] - self.played_total[t] for t in range(NUM_TYPES)]
        if s == self.banker:
            for t in self.buried:
                u[t] -= 1
        return u

    def trick_points(self):
        return sum(M.points_of(c) for _, c in self.cur)

    def current_winner(self):
        if not self.cur:
            return None
        return self.cur[M.trick_winner(self.cur, self.ctx)][0]

    # ------------------------------------------------------------ bidding --
    def _forced(self, s):
        r = self.rules
        return (r.force_enabled and self.best_bid is not None and self.best_bidder != s
                and self.best_bid > r.force_bid and self.changzhu_count(s) >= r.force_changzhu)

    def legal_bids(self, s=None):
        s = self.to_act if s is None else s
        r = self.rules
        top = r.max_bid if self.best_bid is None else self.best_bid - r.bid_step
        vals = [v for v in r.bid_values() if v <= top]
        if not self._forced(s) or not vals:
            vals = vals + [PASS]
        return vals

    def _next_bidder(self, s):
        for i in range(1, 5):
            n = (s + i) % 4
            if not self.passed[n]:
                return n
        return None

    def _bid(self, s, v):
        if v not in self.legal_bids(s):
            raise IllegalMove("非法叫分")
        self.bid_history.append((s, v))
        if v == PASS:
            self.passed[s] = True
            self.events.append({"type": "bid", "seat": s, "value": 0})
        else:
            self.best_bid, self.best_bidder = v, s
            self.events.append({"type": "bid", "seat": s, "value": v})
        active = [i for i in range(4) if not self.passed[i]]
        if not active:
            self._set_banker(self.first_bidder, self.rules.max_bid)
        elif self.best_bid is not None and (self.best_bid <= self.rules.min_bid
                                            or active == [self.best_bidder]):
            self._set_banker(self.best_bidder, self.best_bid)
        else:
            self.to_act = self._next_bidder(s)

    def _set_banker(self, s, v):
        self.banker, self.contract = s, v
        for t in self.bottom:
            self.hands[s][t] += 1
        self.phase = TRUMP
        self.to_act = s
        self.events.append({"type": "banker", "seat": s, "value": v})

    # -------------------------------------------------------------- trump --
    def legal_trumps(self):
        opts = [0, 1, 2, 3]
        if self.rules.allow_no_trump:
            opts.append(NO_TRUMP)
        if self.rules.allow_surrender:
            opts.append(SURRENDER)
        return opts

    def _trump(self, s, a):
        if a not in self.legal_trumps():
            raise IllegalMove("非法定主")
        if a == SURRENDER:
            self.events.append({"type": "surrender", "seat": s})
            self._settle(surrender=True)
            return
        self.trump = a
        self.ctx = trump_context(a)
        self.phase = BURY
        self.events.append({"type": "trump", "seat": s, "value": a})

    # --------------------------------------------------------------- bury --
    def _bury(self, s, t):
        if self.hands[s][t] <= 0:
            raise IllegalMove("没有这张牌")
        self.hands[s][t] -= 1
        self.buried.append(t)
        if len(self.buried) == BURY_COUNT:
            self.phase = PLAY
            self.leader = self.to_act = self.banker
            self.events.append({"type": "buried", "seat": s})

    # --------------------------------------------------------------- play --
    def legal_plays(self, s=None):
        s = self.to_act if s is None else s
        if not self.cur:
            unseen = self.unseen_for(s) if self.rules.allow_trump_throw else None
            return M.lead_moves(self.hands[s], self.ctx, unseen, self.rules.allow_trump_throw)
        return M.follow_moves(self.hands[s], self.lead_shape, self.ctx, self.rng)

    def check_play(self, s, cards) -> str | None:
        if self.phase != PLAY or s != self.to_act:
            return "还没轮到你出牌"
        cards = tuple(sorted(cards))
        if not self.cur:
            return M.check_lead(self.hands[s], cards, self.ctx, self.rules.allow_trump_throw)
        return M.check_follow(self.hands[s], self.lead_shape, cards, self.ctx)

    def _play(self, s, cards):
        cards = tuple(sorted(cards))
        err = self.check_play(s, cards)
        if err:
            raise IllegalMove(err)
        if not self.cur:
            sh = M.analyze(cards, self.ctx)
            if sh.kind == M.THROW:
                ok, forced = M.throw_result(cards, s, self.hands, self.ctx)
                if not ok:
                    self.events.append({"type": "throw_fail", "seat": s, "tried": list(cards),
                                        "forced": list(forced)})
                    cards = forced
                    sh = M.analyze(cards, self.ctx)
            self.lead_shape = sh
        else:
            lc = self.lead_shape.cls
            if any(self.ctx.cls[t] != lc for t in cards):
                self.void[s][lc] = True
        for t in cards:
            self.hands[s][t] -= 1
            self.played[s][t] += 1
            self.played_total[t] += 1
        self.cur.append((s, cards))
        self.events.append({"type": "play", "seat": s, "cards": list(cards)})
        if len(self.cur) < 4:
            self.to_act = (s + 1) % 4
            return
        wi = M.trick_winner(self.cur, self.ctx)
        winner = self.cur[wi][0]
        pts = self.trick_points()
        if winner != self.banker:
            self.xian_points += pts
        trick = {"leader": self.cur[0][0], "plays": [(p, list(c)) for p, c in self.cur],
                 "winner": winner, "points": pts}
        self.tricks.append(trick)
        self.events.append({"type": "trick", "winner": winner, "points": pts})
        last_shape = self.lead_shape
        win_cards = self.cur[wi][1]
        self.cur = []
        self.lead_shape = None
        if self.hand_size(winner) == 0:
            self._koudi(winner, last_shape, win_cards)
            self._settle()
        else:
            self.leader = self.to_act = winner

    def _koudi(self, winner, lead_shape, win_cards):
        bp = M.points_of(self.buried)
        info = {"bottom_points": bp, "mult": 0, "gained": 0, "winner": winner}
        if winner != self.banker and all(self.ctx.is_trump[t] for t in win_cards):
            r = self.rules
            k = lead_shape.kind
            if k == M.SINGLE:
                mult = r.koudi_single
            elif k == M.PAIR:
                mult = r.koudi_pair
            elif k == M.TRACTOR:
                mult = r.koudi_per_tractor_pair * lead_shape.pairs
            else:
                mult = lead_shape.n if r.koudi_throw_per_card else r.koudi_pair
            info["mult"] = mult
            info["gained"] = bp * mult
            self.xian_points += bp * mult
        self.koudi = info
        self.events.append({"type": "koudi", **info})

    # ------------------------------------------------------------- settle --
    def _settle(self, surrender=False):
        r = self.rules
        E, B = self.xian_points, self.contract
        if surrender:
            if B >= r.surrender_threshold:
                tier, m, xian_win = "投降(下庄)", r.kua_zhuang_xian, True
            else:
                tier, m, xian_win = "投降(小倒)", r.xiao_guang_xian, True
        elif E >= B:
            xian_win = True
            if E >= B + r.da_dao_margin:
                tier, m = "大倒", r.da_guang_xian
            elif E >= B + r.xiao_dao_margin:
                tier, m = "小倒", r.xiao_guang_xian
            else:
                tier, m = "下庄", r.kua_zhuang_xian
        else:
            xian_win = False
            if E == 0:
                tier, m = "大光", r.da_guang_xian
            elif E <= r.xiao_guang_max:
                tier, m = "小光", r.xiao_guang_xian
            else:
                tier, m = "保庄", r.bao_zhuang_xian
            if r.shuang_jin_dan_chu:
                m *= 2
        scores = [0] * 4
        for s in range(4):
            if s == self.banker:
                scores[s] = -3 * m if xian_win else 3 * m
            else:
                scores[s] = m if xian_win else -m
        self.result = {"tier": tier, "mult": m, "xian_win": xian_win, "xian_points": E,
                       "contract": B, "banker": self.banker, "scores": scores,
                       "surrender": surrender, "buried": list(self.buried),
                       "koudi": getattr(self, "koudi", None)}
        self.phase = OVER
        self.to_act = None
        self.events.append({"type": "result", **self.result})

    # ---------------------------------------------------------- dispatch --
    def legal_actions(self):
        if self.phase == BID:
            return self.legal_bids()
        if self.phase == TRUMP:
            return self.legal_trumps()
        if self.phase == BURY:
            h = self.hands[self.to_act]
            return [t for t in range(NUM_TYPES) if h[t]]
        if self.phase == PLAY:
            return self.legal_plays()
        return []

    def step(self, action):
        s = self.to_act
        if self.phase == BID:
            self._bid(s, action)
        elif self.phase == TRUMP:
            self._trump(s, action)
        elif self.phase == BURY:
            self._bury(s, action)
        elif self.phase == PLAY:
            self._play(s, action)
        else:
            raise IllegalMove("本局已结束")

    @property
    def over(self):
        return self.phase == OVER


def describe_action(phase, a):
    if phase == BID:
        return "不叫" if a == PASS else f"叫 {a} 分"
    if phase == TRUMP:
        return TRUMP_LABELS[a]
    if phase == BURY:
        return "埋 " + card_name(a)
    if phase == PLAY:
        return " ".join(card_name(t) for t in a)
    return str(a)
