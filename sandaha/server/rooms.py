"""Game rooms: seating, turn flow, timers, AI moves and per-player views."""
from __future__ import annotations

import asyncio
import copy
import random
import time
import uuid

from .. import moves as M
from ..ai import AIAgent
from ..cards import CODE_TO_TYPE, NUM_TYPES, card_code, sort_key, trump_context
from ..engine import (BID, BURY, BURY_COUNT, OVER, PASS, PLAY, SURRENDER, TRUMP, TRUMP_LABELS, Game,
                      IllegalMove, describe_action)
from ..rules import Rules
from . import registry

BATTLE, TEACH = "battle", "teach"
TURN_SECONDS = 30


def codes(cards, ctx=None):
    return [card_code(t) for t in sorted(cards, key=lambda t: sort_key(ctx, t))]


def hand_codes(counts, ctx=None):
    return codes([t for t in range(NUM_TYPES) for _ in range(counts[t])], ctx)


class SeatInfo:
    def __init__(self, kind="open", agent_spec=None):
        self.kind = kind            # open / human / ai
        self.client_id = None
        self.name = ""
        self.agent_spec = agent_spec
        self.agent = None

    def public(self, clients):
        d = {"kind": self.kind, "name": self.name}
        if self.kind == "ai":
            d["model"] = self.agent_spec or "rule"
            d["name"] = self.name or registry.agent_label(self.agent_spec)
        if self.kind == "human":
            c = clients.get(self.client_id)
            d["connected"] = bool(c and c.ws is not None)
        return d


class Room:
    def __init__(self, hub, host_id, name, mode, seats, rules=None, coach=None, ai_delay=0.9,
                 turn_seconds=TURN_SECONDS):
        self.hub = hub
        self.id = uuid.uuid4().hex[:6]
        self.name = name or f"房间 {self.id}"
        self.mode = mode
        self.host_id = host_id
        self.seats = [SeatInfo() for _ in range(4)]
        self.rules = Rules.from_dict(rules)
        self.coach_spec = coach or registry.default_model_path()
        self.ai_delay = float(ai_delay)
        self.turn_seconds = int(turn_seconds)
        self.members: set[str] = set()
        self.status = "waiting"     # waiting / playing / hand_over
        self.game: Game | None = None
        self.hand_no = 0
        self.totals = [0, 0, 0, 0]
        self.past_hands: list[dict] = []
        self.comments: list[dict] = []
        self.deadline = None
        self._task = None
        self._next_first = random.randrange(4)
        self.lock = asyncio.Lock()
        self.created = time.time()
        for i, s in enumerate(seats or []):
            self.configure_seat(i, s)

    # ------------------------------------------------------------ seating --
    def configure_seat(self, i, spec):
        kind = spec.get("kind", "open")
        seat = self.seats[i]
        if kind == "ai":
            seat.kind, seat.client_id = "ai", None
            seat.agent_spec = spec.get("model") or "rule"
            seat.name = f"{registry.agent_label(seat.agent_spec)}·{i + 1}"
            seat.agent = registry.make_agent(seat.agent_spec, seed=random.randrange(1 << 30))
        elif kind == "human" and spec.get("client_id"):
            seat.kind, seat.client_id = "human", spec["client_id"]
            c = self.hub.clients.get(seat.client_id)
            seat.name = c.name if c else "玩家"
            seat.agent = None
        else:
            seat.kind, seat.client_id, seat.agent, seat.name = "open", None, None, ""

    def seat_of(self, client_id):
        for i, s in enumerate(self.seats):
            if s.kind == "human" and s.client_id == client_id:
                return i
        return None

    def coach(self):
        if not self.coach_spec:
            return None
        try:
            return AIAgent(registry.load_model(self.coach_spec))
        except Exception:
            return None

    def summary(self):
        return {"id": self.id, "name": self.name, "mode": self.mode, "status": self.status,
                "seats": [s.public(self.hub.clients) for s in self.seats], "hand_no": self.hand_no,
                "host": self.hub.clients[self.host_id].name if self.host_id in self.hub.clients else ""}

    # --------------------------------------------------------------- flow --
    async def start_hand(self):
        if any(s.kind == "open" for s in self.seats):
            raise IllegalMove("还有空座位")
        self.game = Game(self.rules, seed=random.randrange(1 << 30), first_bidder=self._next_first)
        self.hand_no += 1
        self.status = "playing"
        self.comments = []
        for s in self.seats:
            if s.kind == "ai":
                s.agent = registry.make_agent(s.agent_spec, seed=random.randrange(1 << 30))
        await self.advance()

    async def advance(self):
        g = self.game
        if self._task and not self._task.done() and self._task is not asyncio.current_task():
            self._task.cancel()
        self._task = None
        self.deadline = None
        if g is None:
            await self.broadcast()
            return
        if g.over:
            self._finish_hand()
            if all(x.kind == "ai" for x in self.seats):
                self._task = asyncio.create_task(self._auto_next(self.hand_no))
            await self.broadcast()
            return
        seat = self.seats[g.to_act]
        if seat.kind == "ai":
            self._task = asyncio.create_task(self._ai_turn())
        elif self.mode == BATTLE and self.turn_seconds > 0:
            self.deadline = time.time() + self.turn_seconds
            self._task = asyncio.create_task(self._timeout(self.deadline))
        await self.broadcast()

    def _finish_hand(self):
        if self.status == "hand_over":
            return
        g = self.game
        r = g.result
        for i in range(4):
            self.totals[i] += r["scores"][i]
        self.status = "hand_over"
        self.past_hands.append({"hand_no": self.hand_no, "banker": r["banker"], "contract": r["contract"],
                                "trump": g.trump, "tier": r["tier"], "xian_points": r["xian_points"],
                                "scores": r["scores"], "tricks": self._tricks_view(g),
                                "buried": codes(g.buried, g.ctx), "bids": g.bid_history,
                                "koudi": r.get("koudi")})
        self.past_hands = self.past_hands[-30:]
        b = r["banker"]
        self._next_first = b if not r["xian_win"] else (b + 1) % 4

    async def _ai_turn(self):
        try:
            await asyncio.sleep(self.ai_delay if self.game.phase in (PLAY, BID) else self.ai_delay * 0.6)
            async with self.lock:
                g = self.game
                if g is None or g.over:
                    return
                seat = self.seats[g.to_act]
                if seat.kind != "ai":
                    return
                s = g.to_act
                if g.phase == BURY:
                    cards = []
                    comment = self._comment(g, s, None) if self.mode == TEACH else None
                    while g.phase == BURY:
                        a = await asyncio.to_thread(seat.agent.act, g)
                        cards.append(a)
                        g.step(a)
                    if comment:
                        comment["chosen"] = "埋 " + " ".join(describe_action(PLAY, (t,)) for t in cards)
                        self.comments.append(comment)
                else:
                    a = await asyncio.to_thread(seat.agent.act, g)
                    if self.mode == TEACH:
                        c = self._comment(g, s, a)
                        if c:
                            self.comments.append(c)
                    g.step(a)
                await self.advance()
        except asyncio.CancelledError:
            pass

    async def _auto_next(self, hand_no):
        """All-AI rooms (spectating) continue on their own."""
        try:
            await asyncio.sleep(4 + self.ai_delay * 3)
            async with self.lock:
                if self.hand_no == hand_no and self.status == "hand_over" and self.members:
                    await self.start_hand()
        except asyncio.CancelledError:
            pass

    async def _timeout(self, deadline):
        try:
            await asyncio.sleep(max(0, deadline - time.time()))
            async with self.lock:
                g = self.game
                if g is None or g.over or self.deadline != deadline:
                    return
                s = g.to_act
                rng = random.Random()
                if g.phase == BURY:
                    while g.phase == BURY:
                        g.step(rng.choice(g.legal_actions()))
                else:
                    acts = g.legal_actions()
                    if g.phase == BID and PASS in acts:
                        acts = [PASS]  # an expired bid counts as 不叫
                    if g.phase == TRUMP and len(acts) > 1:
                        acts = [a for a in acts if a != SURRENDER]
                    g.step(rng.choice(acts))
                g.events.append({"type": "timeout", "seat": s})
                await self.advance()
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------- human input --
    async def human_action(self, client_id, msg):
        async with self.lock:
            g = self.game
            s = self.seat_of(client_id)
            if g is None or g.over or s is None or s != g.to_act:
                raise IllegalMove("还没轮到你")
            kind = msg.get("kind")
            comment = None
            if kind == "bid" and g.phase == BID:
                v = int(msg.get("value", 0))
                if self.mode == TEACH:
                    comment = self._comment(g, s, v)
                g.step(v)
            elif kind == "trump" and g.phase == TRUMP:
                v = int(msg.get("value"))
                if self.mode == TEACH:
                    comment = self._comment(g, s, v)
                g.step(v)
            elif kind == "bury" and g.phase == BURY:
                cards = [CODE_TO_TYPE[c] for c in msg.get("cards", [])]
                need = BURY_COUNT - len(g.buried)
                if len(cards) != need:
                    raise IllegalMove(f"请选择 {need} 张牌埋底")
                h = list(g.hands[s])
                for t in cards:
                    h[t] -= 1
                if min(h) < 0:
                    raise IllegalMove("手中没有这些牌")
                if self.mode == TEACH:
                    comment = self._comment(g, s, None)
                    if comment:
                        comment["chosen"] = "埋 " + " ".join(describe_action(PLAY, (t,)) for t in cards)
                for t in cards:
                    g.step(t)
            elif kind == "play" and g.phase == PLAY:
                cards = tuple(sorted(CODE_TO_TYPE[c] for c in msg.get("cards", [])))
                err = g.check_play(s, cards)
                if err:
                    raise IllegalMove(err)
                if self.mode == TEACH:
                    comment = self._comment(g, s, cards)
                g.step(cards)
            else:
                raise IllegalMove("无效操作")
            if comment:
                self.comments.append(comment)
            await self.advance()

    # ---------------------------------------------------------- coaching --
    def _comment(self, g, s, chosen):
        """AI evaluation of the current decision (teaching mode)."""
        coach = self.coach()
        if coach is None:
            return None
        try:
            if g.phase == BURY:
                sim = copy.deepcopy(g)
                picks = []
                while sim.phase == BURY:
                    a = coach.act(sim)
                    picks.append(a)
                    sim.step(a)
                return {"seat": s, "phase": g.phase, "hand": self.hand_no, "trick": 0,
                        "suggest": "埋 " + " ".join(describe_action(PLAY, (t,)) for t in picks),
                        "codes": [card_code(t) for t in picks], "top": [], "chosen": None}
            rows = coach.evaluate(g)
            if chosen is not None and not any(r["action"] == chosen for r in rows):
                extra = self._eval_extra(coach, g, chosen)
                if extra:
                    rows.append(extra)
                    rows.sort(key=lambda r: -r["exp_score"])
            rank = next((i for i, r in enumerate(rows) if r["action"] == chosen), None)
            best = rows[0] if rows else None
            c = {"seat": s, "phase": g.phase, "hand": self.hand_no, "trick": len(g.tricks) + 1,
                 "n_options": len(rows),
                 "top": [self._row(r, g.phase) for r in rows[:3]],
                 "chosen": describe_action(g.phase, chosen) if chosen is not None else None,
                 "chosen_rank": (rank + 1) if rank is not None else None}
            if rank is not None and best:
                c["chosen_eval"] = self._row(rows[rank], g.phase)
                c["regret"] = round(best["exp_score"] - rows[rank]["exp_score"], 3)
            return c
        except Exception as e:  # never let coaching break the game
            return {"seat": s, "phase": g.phase, "error": str(e)}

    def _eval_extra(self, coach, g, chosen):
        from ..ai import features as F
        import numpy as np
        phase, st, A = F.encode(g, g.to_act, [chosen], coach.model.feat_ver)
        out = coach.model.q(phase, st, A)[0]
        return {"action": chosen, "label": describe_action(phase, chosen), "exp_score": float(out[0] * 9),
                "win_prob": float(1 / (1 + np.exp(-out[1]))), "xian_points": float(out[2] * 400)}

    @staticmethod
    def _row(r, phase=None):
        d = {"label": r["label"], "exp_score": round(r["exp_score"], 3),
             "win_prob": round(r["win_prob"], 3), "xian_points": round(r["xian_points"], 1)}
        if phase == PLAY:
            d["codes"] = [card_code(t) for t in r["action"]]
        else:
            d["value"] = r["action"]
        return d

    async def hint(self, client_id):
        g = self.game
        s = self.seat_of(client_id)
        if self.mode != TEACH or g is None or g.over or s != g.to_act:
            raise IllegalMove("只有教学模式下轮到你时可以请求提示")
        c = self._comment(g, s, None)
        return c

    # -------------------------------------------------------------- views --
    def _tricks_view(self, g):
        return [{"leader": t["leader"], "winner": t["winner"], "points": t["points"],
                 "plays": [{"seat": p, "cards": codes(c, g.ctx)} for p, c in t["plays"]]}
                for t in g.tricks]

    def view(self, client_id):
        me = self.seat_of(client_id)
        g = self.game
        d = {"op": "state", "room": self.summary(), "my_seat": me, "mode": self.mode,
             "is_host": client_id == self.host_id, "totals": self.totals,
             "past_hands": [{k: v for k, v in h.items() if k != "tricks"} for h in self.past_hands],
             "turn_seconds": self.turn_seconds, "now": time.time()}
        if g is None:
            return d
        ctx = g.ctx
        reveal_all = (self.mode == TEACH and me is not None) or g.over
        d.update({
            "phase": g.phase, "to_act": g.to_act, "deadline": self.deadline,
            "first_bidder": g.first_bidder,
            "bids": [{"seat": s, "value": v} for s, v in g.bid_history],
            "best_bid": g.best_bid, "best_bidder": g.best_bidder, "passed": g.passed,
            "banker": g.banker, "contract": g.contract, "trump": g.trump,
            "trump_label": TRUMP_LABELS[g.trump] if g.trump is not None else None,
            "hand_counts": [g.hand_size(i) for i in range(4)],
            "cur": [{"seat": p, "cards": codes(c, ctx)} for p, c in g.cur],
            "tricks": self._tricks_view(g),
            "xian_points": g.xian_points, "void": g.void,
            "result": g.result, "hand_no": self.hand_no,
            "events": g.events[-12:],
            "lead": ({"kind": g.lead_shape.kind, "n": g.lead_shape.n, "cls": g.lead_shape.cls}
                     if g.lead_shape else None),
        })
        if me is not None:
            d["hand"] = hand_codes(g.hands[me], ctx)
            if g.to_act == me:
                if g.phase == BID:
                    d["legal_bids"] = g.legal_bids(me)
                elif g.phase == TRUMP:
                    d["legal_trumps"] = g.legal_trumps()
                elif g.phase == BURY:
                    d["bury_need"] = BURY_COUNT - len(g.buried)
            if me == g.banker and g.phase in (TRUMP, BURY):
                d["bottom"] = codes(g.bottom)
            if me == g.banker and g.buried:
                d["buried"] = codes(g.buried, ctx)
            unseen = g.unseen_for(me)
        else:
            unseen = [2 - g.played_total[t] for t in range(NUM_TYPES)]
        if g.over:
            d["buried"] = codes(g.buried, ctx)
            d["bottom"] = codes(g.bottom)
        if reveal_all:
            d["all_hands"] = [hand_codes(g.hands[i], ctx) for i in range(4)]
            if g.over:
                d["initial_hands"] = [codes(h, ctx) for h in g.initial_hands]
        d["counter"] = {card_code(t): max(0, unseen[t]) for t in range(NUM_TYPES)}
        if self.mode == TEACH:
            d["comments"] = self.comments[-60:]
        return d

    async def broadcast(self):
        for cid in list(self.members):
            c = self.hub.clients.get(cid)
            if c and c.ws is not None:
                await c.send(self.view(cid))

    def cancel(self):
        if self._task and not self._task.done():
            self._task.cancel()
