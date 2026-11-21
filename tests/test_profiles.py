"""Player save slots: storage, name rules, one-device-at-a-time binding and hand recording."""
import asyncio
import json
import os
import time

import pytest

from sandaha.engine import IllegalMove
from sandaha.server import profiles, registry
from sandaha.server.rooms import BATTLE, Room


@pytest.fixture(autouse=True)
def tmp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "ROOT", str(tmp_path))
    profiles.reset_cache()
    yield tmp_path
    profiles.reset_cache()


class _Client:
    def __init__(self, cid, online=True):
        self.id, self.name, self.room_id, self.profile_id = cid, "H", None, None
        self.ws = object() if online else None
        self.sent = []

    async def send(self, obj):
        self.sent.append(obj)


class _Hub:
    def __init__(self, *clients):
        self.clients = {c.id: c for c in clients}
        self.rooms = {}

    async def push_rooms(self):
        pass


def test_create_rename_delete_and_persist(tmp_root):
    a = profiles.create("  小  明 ")
    assert a["name"] == "小 明"
    b = profiles.create("Bob")
    with pytest.raises(IllegalMove):
        profiles.create("bob")              # names are unique, case-insensitive
    with pytest.raises(IllegalMove):
        profiles.create("   ")
    with pytest.raises(IllegalMove):
        profiles.create("x" * 13)
    profiles.rename(b["id"], "Bobby")
    with pytest.raises(IllegalMove):
        profiles.rename(b["id"], "小 明")
    profiles.reset_cache()                  # survives a restart
    names = {p["name"] for p in profiles.list_profiles()}
    assert names == {"小 明", "Bobby"}
    profiles.delete(a["id"])
    assert [p["name"] for p in profiles.list_profiles()] == ["Bobby"]
    assert json.load(open(os.path.join(tmp_root, "data", "profiles.json"), encoding="utf-8"))["profiles"][0]["name"] == "Bobby"


def test_record_hand_stats():
    p = profiles.create("A")["id"]
    for score, banker in ((3, False), (6, True), (-1, False), (2, False), (1, False)):
        profiles.record_hand(p, "battle", score, banker, 70, "保庄", 40)
    profiles.record_hand(p, "teach", -9, True, 50, "下庄", 60)
    d = profiles.detail(p)
    s = d["stats"]["battle"]
    assert (s["hands"], s["wins"], s["losses"], s["score"]) == (5, 4, 1, 11)
    assert (s["banker_hands"], s["banker_wins"], s["best"], s["worst"]) == (1, 1, 6, -1)
    assert (s["streak"], s["best_streak"]) == (2, 2)
    assert d["stats"]["teach"]["hands"] == 1 and d["stats"]["teach"]["score"] == -9
    assert d["recent"][0]["score"] == -9 and len(d["recent"]) == 6       # newest first
    profiles.record_hand("nope", "battle", 3, False, 70, "保庄", 40)      # unknown id: ignored


def test_broken_file_is_set_aside(tmp_root):
    d = os.path.join(tmp_root, "data")
    os.makedirs(d)
    with open(os.path.join(d, "profiles.json"), "w") as f:
        f.write("{not json")
    assert profiles.list_profiles() == []
    profiles.create("A")
    assert any(n.startswith("profiles.json.broken-") for n in os.listdir(d))


def test_one_profile_per_online_device():
    a, b, gone = _Client("a"), _Client("b"), _Client("g", online=False)
    hub = _Hub(a, b, gone)
    pid = profiles.create("小明")["id"]
    profiles.bind(hub, a, pid)
    assert a.name == "小明"
    with pytest.raises(IllegalMove):
        profiles.bind(hub, b, pid)          # in use on another connected device
    gone.profile_id = pid                   # an offline device does not block it
    a.profile_id = None
    profiles.bind(hub, b, pid)
    # reconnecting device gets its profile back; an unknown one is dropped
    assert profiles.on_hello(hub, a, {"profile_id": pid}) is None and a.profile_id is None
    assert profiles.on_hello(hub, b, {"profile_id": pid})["id"] == pid
    assert profiles.on_hello(hub, b, {"profile_id": "missing"})["id"] == pid   # keeps what it had


def test_ops_select_rename_delete():
    async def run():
        a, b = _Client("a"), _Client("b")
        hub = _Hub(a, b)
        await profiles.handle_op(hub, a, {"op": "profile_create", "name": "甲"})
        pid = a.profile_id
        assert a.sent[0] == {"op": "profile", "profile": profiles.summary(pid)}
        await profiles.handle_op(hub, b, {"op": "profile_list"})
        assert b.sent[-1]["in_use"] == {pid: True}
        with pytest.raises(IllegalMove):
            await profiles.handle_op(hub, b, {"op": "profile_delete", "id": pid})
        await profiles.handle_op(hub, b, {"op": "profile_rename", "id": pid, "name": "甲甲"})
        assert a.name == "甲甲" and a.sent[-1]["profile"]["name"] == "甲甲"
        await profiles.handle_op(hub, a, {"op": "profile_delete", "id": pid})
        assert a.profile_id is None and {"op": "profile", "profile": None} in a.sent

    asyncio.run(run())


def test_finished_hand_is_recorded_for_human_seats():
    async def run():
        me = _Client("h")
        hub = _Hub(me)
        pid = profiles.create("我")["id"]
        profiles.bind(hub, me, pid)
        seats = [{"kind": "human", "client_id": "h"}] + [{"kind": "ai", "model": "rule"}] * 3
        room = Room(hub, "h", "t", BATTLE, seats, None, None, 0.0, 1)
        room.members.add("h")
        async with room.lock:
            await room.start_hand()
        t0 = time.time()
        while not room.game.over:           # the 1 s turn timer plays for the human
            assert time.time() - t0 < 120
            await asyncio.sleep(0.02)
        room.cancel()
        d = profiles.detail(pid)
        assert d["stats"]["battle"]["hands"] == 1
        assert d["recent"][0]["score"] == room.game.result["scores"][0]
        assert d["recent"][0]["banker"] == (room.game.result["banker"] == 0)

    asyncio.run(run())
