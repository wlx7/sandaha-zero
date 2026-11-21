"""Player profiles: named save slots ("谁在玩？") that keep each player's record.

Profiles live on the machine that runs the server (the PC for the EXE / source tree, the phone
for the APK), in ``<root>/data/profiles.json``. LAN guests pick or create a profile on the host,
so the host keeps everybody's record. One profile can be used by one connected device at a time.

Every finished hand of a human seat that has a profile is recorded: the stats per mode (对战 /
教学) and the most recent hands. A guest (游客) plays under a free-text name without a profile:
nothing is saved, the score only lives in the room's running total.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

from ..engine import IllegalMove
from . import registry

NAME_MAX = 12
MAX_PROFILES = 50
RECENT_KEEP = 40
MODES = ("battle", "teach")

_lock = threading.RLock()
_cache: tuple[str, dict] | None = None


# ----------------------------------------------------------------- storage --
def _path() -> str:
    return os.path.join(registry.ROOT, "data", "profiles.json")


def _load() -> dict:
    global _cache
    p = _path()
    if _cache is not None and _cache[0] == p:
        return _cache[1]
    data = {"version": 1, "profiles": []}
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("profiles"), list):
            data = d
        else:
            raise ValueError("unexpected layout")
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        # Never overwrite a file we could not read: set it aside and start a fresh one.
        try:
            os.replace(p, f"{p}.broken-{int(time.time())}")
        except OSError:
            pass
    _cache = (p, data)
    return data


def _save(data: dict):
    p = _path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def reset_cache():
    """Forget the in-memory copy (tests, or after the root directory changed)."""
    global _cache
    with _lock:
        _cache = None


# ------------------------------------------------------------------ model --
def _blank_stats() -> dict:
    return {"hands": 0, "wins": 0, "losses": 0, "score": 0,
            "banker_hands": 0, "banker_wins": 0, "best": None, "worst": None,
            "streak": 0, "best_streak": 0}


def clean_name(name) -> str:
    name = " ".join(str(name or "").split())
    name = "".join(ch for ch in name if ch.isprintable())
    if not name:
        raise IllegalMove("请输入用户名")
    if len(name) > NAME_MAX:
        raise IllegalMove(f"用户名最多 {NAME_MAX} 个字")
    return name


def _find(data, pid):
    for p in data["profiles"]:
        if p["id"] == pid:
            return p
    return None


def _check_unique(data, name, skip_id=None):
    key = name.casefold()
    for p in data["profiles"]:
        if p["id"] != skip_id and p["name"].casefold() == key:
            raise IllegalMove(f"已经有叫「{name}」的用户了")


def _summary(p) -> dict:
    return {"id": p["id"], "name": p["name"], "created": p["created"], "last_played": p.get("last_played"),
            "stats": {m: dict(p["stats"].get(m) or _blank_stats()) for m in MODES}}


def summary(pid) -> dict | None:
    if not pid:
        return None
    with _lock:
        p = _find(_load(), pid)
        return _summary(p) if p else None


def detail(pid) -> dict:
    with _lock:
        p = _find(_load(), pid)
        if p is None:
            raise IllegalMove("该用户不存在")
        d = _summary(p)
        d["recent"] = list(reversed(p.get("recent", [])))
        return d


def list_profiles() -> list[dict]:
    with _lock:
        ps = [_summary(p) for p in _load()["profiles"]]
    # Most recently used first, like a save-slot list
    ps.sort(key=lambda p: -(p["last_played"] or p["created"]))
    return ps


def create(name) -> dict:
    name = clean_name(name)
    with _lock:
        data = _load()
        _check_unique(data, name)
        if len(data["profiles"]) >= MAX_PROFILES:
            raise IllegalMove(f"最多 {MAX_PROFILES} 个用户，请先删除不用的")
        p = {"id": uuid.uuid4().hex[:12], "name": name, "created": time.time(), "last_played": None,
             "stats": {m: _blank_stats() for m in MODES}, "recent": []}
        data["profiles"].append(p)
        _save(data)
        return _summary(p)


def rename(pid, name) -> dict:
    name = clean_name(name)
    with _lock:
        data = _load()
        p = _find(data, pid)
        if p is None:
            raise IllegalMove("该用户不存在")
        _check_unique(data, name, skip_id=pid)
        p["name"] = name
        _save(data)
        return _summary(p)


def delete(pid):
    with _lock:
        data = _load()
        p = _find(data, pid)
        if p is None:
            raise IllegalMove("该用户不存在")
        data["profiles"].remove(p)
        _save(data)


def record_hand(pid, mode, score, is_banker, contract, tier, xian_points, room="", hand_no=0):
    """Add one finished hand to a profile's record (no-op for an unknown profile)."""
    mode = mode if mode in MODES else "battle"
    score = int(score)
    with _lock:
        data = _load()
        p = _find(data, pid)
        if p is None:
            return
        st = p["stats"].setdefault(mode, _blank_stats())
        st["hands"] += 1
        st["score"] += score
        if score > 0:
            st["wins"] += 1
            st["streak"] = st["streak"] + 1 if st["streak"] > 0 else 1
        elif score < 0:
            st["losses"] += 1
            st["streak"] = st["streak"] - 1 if st["streak"] < 0 else -1
        st["best_streak"] = max(st["best_streak"], st["streak"])
        if is_banker:
            st["banker_hands"] += 1
            st["banker_wins"] += score > 0
        st["best"] = score if st["best"] is None else max(st["best"], score)
        st["worst"] = score if st["worst"] is None else min(st["worst"], score)
        now = time.time()
        p["last_played"] = now
        rec = p.setdefault("recent", [])
        rec.append({"t": now, "mode": mode, "banker": bool(is_banker), "contract": contract, "tier": tier,
                    "xian_points": xian_points, "score": score, "room": room, "hand_no": hand_no})
        del rec[:-RECENT_KEEP]
        _save(data)


# -------------------------------------------------------- client binding --
def _users(hub, pid, exclude=None):
    """Connected clients (other than ``exclude``) currently playing as profile ``pid``."""
    return [c for c in hub.clients.values()
            if c is not exclude and c.ws is not None and getattr(c, "profile_id", None) == pid]


def _seated_in_running_hand(hub, client):
    room = hub.rooms.get(client.room_id) if client.room_id else None
    return room is not None and room.status == "playing" and room.seat_of(client.id) is not None


def _apply_name(hub, client):
    """Seats keep a copy of the player's name: refresh it after a switch or a rename."""
    rooms = []
    for room in hub.rooms.values():
        hit = False
        for s in room.seats:
            if s.kind == "human" and s.client_id == client.id:
                s.name = client.name
                hit = True
        if hit:
            rooms.append(room)
    return rooms


def bind(hub, client, pid) -> dict:
    """Make ``client`` play as profile ``pid``; returns the profile summary."""
    prof = summary(pid)
    if prof is None:
        raise IllegalMove("该用户不存在")
    if getattr(client, "profile_id", None) != pid:
        if _users(hub, pid, exclude=client):
            raise IllegalMove(f"「{prof['name']}」正在另一台设备上使用")
        if _seated_in_running_hand(hub, client):
            raise IllegalMove("对局进行中不能切换用户")
    client.profile_id = pid
    client.name = prof["name"]
    return prof


def on_hello(hub, client, msg) -> dict | None:
    """Restore the profile a reconnecting device remembers (silently dropped when unavailable)."""
    current = getattr(client, "profile_id", None)
    if msg.get("guest"):
        # This device chose 游客模式: drop the profile, unless it is in the middle of a recorded hand
        if current and _seated_in_running_hand(hub, client):
            prof = summary(current)
            if prof:
                client.name = prof["name"]
                return prof
        client.profile_id = None
        return None
    pid = msg.get("profile_id") or current
    if not pid:
        return None
    try:
        return bind(hub, client, pid)
    except IllegalMove:
        # Keep what this connection already plays as (e.g. mid-hand); otherwise it has to choose again
        prof = summary(current) if current else None
        if prof is None:
            client.profile_id = None
        else:
            client.name = prof["name"]
        return prof


def _list_msg(hub, client) -> dict:
    in_use = {}
    for c in hub.clients.values():
        pid = getattr(c, "profile_id", None)
        if c is not client and c.ws is not None and pid:
            in_use[pid] = True
    return {"op": "profiles", "profiles": list_profiles(), "current": getattr(client, "profile_id", None),
            "in_use": in_use}


async def _refresh_rooms(hub, rooms):
    for room in rooms:
        await room.broadcast()
    if rooms:
        await hub.push_rooms()


async def handle_op(hub, client, msg):
    """WebSocket ops: profile_list / _create / _select / _guest / _rename / _delete / _get."""
    op = msg["op"]
    if op == "profile_list":
        await client.send(_list_msg(hub, client))
    elif op == "profile_get":
        await client.send({"op": "profile_detail", "profile": detail(msg.get("id"))})
    elif op == "profile_create":
        if _seated_in_running_hand(hub, client):
            raise IllegalMove("对局进行中不能切换用户")
        prof = create(msg.get("name"))
        bind(hub, client, prof["id"])
        await client.send({"op": "profile", "profile": prof})
        await client.send(_list_msg(hub, client))
        await _refresh_rooms(hub, _apply_name(hub, client))
    elif op == "profile_guest":
        # 游客模式: play under a free-text name, nothing is saved
        name = clean_name(msg.get("name"))
        if getattr(client, "profile_id", None) and _seated_in_running_hand(hub, client):
            raise IllegalMove("对局进行中不能切换用户")
        client.profile_id = None
        client.name = name
        await client.send({"op": "profile", "profile": None, "guest": True, "name": name})
        await _refresh_rooms(hub, _apply_name(hub, client))
    elif op == "profile_select":
        prof = bind(hub, client, msg.get("id"))
        await client.send({"op": "profile", "profile": prof})
        await _refresh_rooms(hub, _apply_name(hub, client))
    elif op == "profile_rename":
        pid = msg.get("id")
        prof = rename(pid, msg.get("name"))
        rooms = []
        for c in list(hub.clients.values()):
            if getattr(c, "profile_id", None) == pid:
                c.name = prof["name"]
                rooms += _apply_name(hub, c)
                if c is not client:
                    await c.send({"op": "profile", "profile": prof})
        if getattr(client, "profile_id", None) == pid:
            await client.send({"op": "profile", "profile": prof})
        await client.send(_list_msg(hub, client))
        await _refresh_rooms(hub, list(dict.fromkeys(rooms)))
    elif op == "profile_delete":
        pid = msg.get("id")
        if _users(hub, pid, exclude=client):
            raise IllegalMove("这个用户正在另一台设备上使用，不能删除")
        if getattr(client, "profile_id", None) == pid and _seated_in_running_hand(hub, client):
            raise IllegalMove("对局进行中不能删除当前用户")
        delete(pid)
        for c in hub.clients.values():
            if getattr(c, "profile_id", None) == pid:
                c.profile_id = None
                if c is not client:
                    await c.send({"op": "profile", "profile": None})
        if getattr(client, "profile_id", None) is None:
            await client.send({"op": "profile", "profile": None})
        await client.send(_list_msg(hub, client))
    else:
        raise IllegalMove(f"未知操作 {op}")
