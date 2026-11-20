"""Room turn flow must never freeze: failing AI, lost timers and expired forced bids all recover."""
import asyncio
import time

from sandaha.cards import NUM_TYPES, is_changzhu
from sandaha.engine import BID, PASS
from sandaha.server.rooms import BATTLE, Room


class _Client:
    def __init__(self, cid):
        self.id, self.name, self.ws, self.room_id = cid, "H", object(), None
        self.sent = []

    async def send(self, obj):
        self.sent.append(obj)


class _Hub:
    def __init__(self):
        self.clients, self.rooms = {}, {}

    async def push_rooms(self):
        pass


def _room(seats, turn_seconds=30):
    hub = _Hub()
    hub.clients["h"] = _Client("h")
    room = Room(hub, "h", "t", BATTLE, seats, None, None, 0.0, turn_seconds)
    room.members.add("h")
    return room, hub.clients["h"]


async def _play_out(room, limit=20.0):
    t0 = time.time()
    while not room.game.over:
        assert time.time() - t0 < limit, f"hand froze in phase {room.game.phase}"
        await asyncio.sleep(0.01)


class _Broken:
    name = "broken"

    def act(self, g):
        raise RuntimeError("model exploded")


def test_failing_ai_does_not_freeze_the_hand():
    async def run():
        room, _ = _room([{"kind": "ai", "model": "rule"}] * 4)
        async with room.lock:
            await room.start_hand()
        room.seats[1].agent = _Broken()
        room.seats[2].agent = _Broken()
        await _play_out(room)
        assert room.game.result is not None
        room.cancel()

    asyncio.run(run())


def test_lost_turn_task_is_revived_by_kick():
    async def run():
        room, client = _room([{"kind": "ai", "model": "rule"}] * 4)
        async with room.lock:
            await room.start_hand()
        room._task.cancel()          # simulate the scheduled AI move being lost
        await asyncio.sleep(0.05)
        assert room.stalled()
        rev = room.rev
        await room.kick()
        assert not room.stalled() and room.rev > rev
        await _play_out(room)
        room.cancel()

    asyncio.run(run())


def test_expired_forced_bid_takes_the_easiest_contract():
    async def run():
        room, _ = _room([{"kind": "human", "client_id": "h"}] + [{"kind": "ai", "model": "rule"}] * 3)
        async with room.lock:
            await room.start_hand()
        room.cancel()
        g = room.game
        # Put the human on turn holding 8 常主 while another seat stands at 80: passing is not legal.
        cz = [t for t in range(NUM_TYPES) if is_changzhu(t)]
        hand = [0] * NUM_TYPES
        for t in (cz * 2)[:8]:
            hand[t] += 1
        g.hands[0] = hand
        g.bid_history, g.passed = [(1, 80)], [False] * 4
        g.best_bid, g.best_bidder, g.to_act, g.phase = 80, 1, 0, BID
        assert PASS not in g.legal_bids(0)
        room._force_move(g, 0)
        assert g.bid_history[-1] == (0, 75)

    asyncio.run(run())


def test_expired_bid_counts_as_pass():
    async def run():
        room, _ = _room([{"kind": "human", "client_id": "h"}] + [{"kind": "ai", "model": "rule"}] * 3, turn_seconds=1)
        async with room.lock:
            await room.start_hand()
        t0 = time.time()
        while not any(s == 0 for s, _ in room.game.bid_history) and room.game.phase == BID:
            assert time.time() - t0 < 10, "bidding froze waiting for the human"
            await asyncio.sleep(0.02)
        mine = [v for s, v in room.game.bid_history if s == 0]
        # the timer answered for the human: 不叫, unless the 常主 rule made a bid compulsory
        assert mine and (mine[0] == PASS or room.game.changzhu_count(0) >= room.rules.force_changzhu)
        room.cancel()

    asyncio.run(run())
