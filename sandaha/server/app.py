"""FastAPI application: static web client, REST API and the game WebSocket."""
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..engine import IllegalMove
from ..rules import Rules
from . import registry
from .rooms import BATTLE, TEACH, Room, destroy_room

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")


def lan_addresses():
    """LAN IPv4 addresses, the one on the default route first."""
    primary = None
    for probe in ("8.8.8.8", "10.255.255.255"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((probe, 80))  # UDP connect sends nothing; it just picks the route
            primary = s.getsockname()[0]
            s.close()
            break
        except OSError:
            continue
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and not ip.startswith("169.254."):
                ips.add(ip)
    except OSError:
        pass
    if primary and not primary.startswith("127."):
        ips.add(primary)

    def rank(ip):
        # virtual adapters (VMware, WSL, VPN/TUN) usually own the .1 address of their subnet
        virtual = ip.endswith(".1")
        home = ip.startswith("192.168.") or ip.startswith("10.")
        return (virtual, not home, ip != primary, ip)

    return sorted(ips, key=rank)


class Client:
    def __init__(self, cid, name):
        self.id = cid
        self.name = name
        self.ws: WebSocket | None = None
        self.room_id = None
        self.last_seen = time.time()

    async def send(self, obj):
        if self.ws is None:
            return
        try:
            await self.ws.send_text(json.dumps(obj, ensure_ascii=False))
        except Exception:
            self.ws = None


class Hub:
    def __init__(self):
        self.clients: dict[str, Client] = {}
        self.rooms: dict[str, Room] = {}

    def room_list(self):
        return [r.summary() for r in sorted(self.rooms.values(), key=lambda r: r.created)]

    async def push_rooms(self):
        msg = {"op": "rooms", "rooms": self.room_list()}
        for c in list(self.clients.values()):
            if c.ws is not None and c.room_id is None:
                await c.send(msg)

    def gc(self):
        now = time.time()
        for rid, r in list(self.rooms.items()):
            humans = [s for s in r.seats if s.kind == "human"]
            online = [cid for cid in r.members if cid in self.clients and self.clients[cid].ws is not None]
            if not online and now - r.created > 30:
                stale = all(now - self.clients[s.client_id].last_seen > 600
                            for s in humans if s.client_id in self.clients)
                if stale:
                    r.cancel()
                    del self.rooms[rid]


hub = Hub()
app = FastAPI(title="SanDaHa Zero", version=__version__)


@app.middleware("http")
async def no_cache(request, call_next):
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(os.path.join(WEB_DIR, "icon.svg"), media_type="image/svg+xml")


@app.get("/api/info")
async def info():
    port = int(os.environ.get("SANDAHA_PORT", "8765"))
    standalone = registry.is_standalone()
    torch_st = {"available": False}
    if not standalone:
        from . import training
        torch_st = training.torch_info()
    return {
        "version": __version__,
        "lan": [f"http://{ip}:{port}" for ip in lan_addresses()],
        "port": port,
        "default_model": registry.default_model_path(),
        "rules": Rules().to_dict(),
        "standalone": standalone,
        "mobile": os.environ.get("SANDAHA_MOBILE") == "1",
        "can_train": not standalone,
        "torch": torch_st,
    }


@app.get("/api/models")
async def models():
    return {"models": registry.list_models(), "default": registry.default_model_path()}


@app.get("/api/qr")
async def qr(url: str):
    import segno
    import io
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="svg", scale=5, border=2, dark="#1b1b1b", light="#ffffff")
    return Response(buf.getvalue(), media_type="image/svg+xml")


# ---------------------------------------------------------------- training --

def _require_train():
    if registry.is_standalone():
        raise HTTPException(404, "独立客户端已移除训练模块")
    from . import training
    return training


def _wrap(fn, *a):
    try:
        return fn(*a)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/train/presets")
async def train_presets():
    training = _require_train()
    from ..train.config import TrainConfig
    return {"presets": training.presets(), "defaults": TrainConfig().__dict__,
            "torch": training.torch_info(), "cpus": os.cpu_count()}


@app.get("/api/train/runs")
async def train_runs():
    training = _require_train()
    return {"runs": training.list_runs()}


@app.get("/api/train/run/{name}")
async def train_run(name: str):
    training = _require_train()
    return _wrap(training.run_detail, name)


@app.post("/api/train/start")
async def train_start(cfg: dict = Body(...)):
    training = _require_train()
    return _wrap(training.start, cfg)


@app.post("/api/train/stop/{name}")
async def train_stop(name: str):
    training = _require_train()
    return _wrap(training.stop, name)


@app.post("/api/train/delete/{name}")
async def train_delete(name: str):
    training = _require_train()
    return _wrap(training.delete, name)


@app.post("/api/train/publish/{name}")
async def train_publish(name: str, body: dict = Body(...)):
    training = _require_train()
    return _wrap(training.publish, name, body.get("source"), body.get("target"))


@app.post("/api/train/install")
async def train_install():
    training = _require_train()
    return training.install_torch()


@app.get("/api/train/install")
async def train_install_status():
    training = _require_train()
    return training.install_status()


@app.post("/api/eval")
async def quick_eval(body: dict = Body(...)):
    training = _require_train()
    a, b = body.get("a"), body.get("b")
    deals = min(500, int(body.get("deals", 100)))
    try:
        return await asyncio.to_thread(training.quick_eval, a, b, deals)
    except Exception as e:
        raise HTTPException(400, str(e))


# --------------------------------------------------------------- websocket --

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    client = None
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            op = msg.get("op")
            if op == "hello":
                cid = msg.get("client_id") or uuid.uuid4().hex
                client = hub.clients.get(cid)
                if client is None:
                    client = Client(cid, msg.get("name") or "玩家")
                    hub.clients[cid] = client
                if msg.get("name"):
                    client.name = msg["name"][:16]
                if client.ws is not None and client.ws is not ws:
                    try:
                        await client.ws.close()
                    except Exception:
                        pass
                client.ws = ws
                client.last_seen = time.time()
                await client.send({"op": "welcome", "client_id": cid, "name": client.name})
                room = hub.rooms.get(client.room_id) if client.room_id else None
                if room:
                    for s in room.seats:
                        if s.client_id == cid:
                            s.name = client.name
                    await client.send(room.view(cid))
                else:
                    client.room_id = None
                    await client.send({"op": "rooms", "rooms": hub.room_list()})
                continue
            if client is None:
                await ws.send_text(json.dumps({"op": "error", "msg": "请先登录"}))
                continue
            client.last_seen = time.time()
            try:
                await handle(client, msg)
            except IllegalMove as e:
                await client.send({"op": "error", "msg": str(e)})
            except (KeyError, ValueError, TypeError) as e:
                await client.send({"op": "error", "msg": f"请求错误: {e}"})
    except WebSocketDisconnect:
        pass
    finally:
        if client is not None and client.ws is ws:
            client.ws = None
            room = hub.rooms.get(client.room_id) if client.room_id else None
            if room:
                await room.broadcast()
        hub.gc()


async def handle(client: Client, msg: dict):
    op = msg["op"]
    room = hub.rooms.get(client.room_id) if client.room_id else None
    if op == "list_rooms":
        await client.send({"op": "rooms", "rooms": hub.room_list()})
    elif op == "sync":
        if room:
            await client.send(room.view(client.id))
        else:
            await client.send({"op": "rooms", "rooms": hub.room_list()})
    elif op == "create_room":
        if room:
            await leave(client, room)
        mode = TEACH if msg.get("mode") == TEACH else BATTLE
        seats = msg.get("seats") or []
        specs = []
        for i in range(4):
            s = seats[i] if i < len(seats) else {"kind": "open"}
            if s.get("kind") == "me":
                s = {"kind": "human", "client_id": client.id}
            elif s.get("kind") == "human":
                s = {"kind": "open"}
            specs.append(s)
        if mode == TEACH:
            # teaching: this player + three AIs, no time limit
            mine = next((i for i, s in enumerate(specs) if s.get("kind") == "human"), 0)
            for i in range(4):
                if i == mine:
                    specs[i] = {"kind": "human", "client_id": client.id}
                elif specs[i].get("kind") != "ai":
                    specs[i] = {"kind": "ai", "model": msg.get("coach") or registry.default_model_path() or "rule"}
        room = Room(hub, client.id, msg.get("name"), mode, specs, msg.get("rules"), msg.get("coach"),
                    msg.get("ai_delay", 0.9), msg.get("turn_seconds", 30) if mode == BATTLE else 0)
        hub.rooms[room.id] = room
        room.members.add(client.id)
        client.room_id = room.id
        await room.broadcast()
        await hub.push_rooms()
        if all(s.kind != "open" for s in room.seats) and msg.get("autostart", True):
            async with room.lock:
                await room.start_hand()
    elif op == "join_room":
        r = hub.rooms.get(msg.get("room"))
        if r is None:
            raise IllegalMove("房间不存在")
        if room and room is not r:
            await leave(client, room)
        r.members.add(client.id)
        client.room_id = r.id
        await r.broadcast()
        await hub.push_rooms()
    elif op == "close_room":
        target = hub.rooms.get(msg.get("room")) if msg.get("room") else room
        if target is None:
            await client.send({"op": "rooms", "rooms": hub.room_list()})
            return
        if not target.can_manage(client.id):
            raise IllegalMove("只有房主可以结束该房间")
        await destroy_room(hub, target, client.id)
    elif op == "close_all_rooms":
        targets = [r for r in list(hub.rooms.values()) if r.can_manage(client.id)]
        for t in targets:
            await destroy_room(hub, t, client.id)
        await client.send({"op": "rooms", "rooms": hub.room_list()})
    elif room is None:
        raise IllegalMove("你不在房间中")
    elif op == "leave_room":
        await leave(client, room)
        await client.send({"op": "rooms", "rooms": hub.room_list()})
    elif op == "take_seat":
        i = int(msg["seat"])
        if room.seats[i].kind != "open":
            raise IllegalMove("该座位已被占用")
        old = room.seat_of(client.id)
        if old is not None:
            if room.status == "playing":
                raise IllegalMove("对局进行中不能换座")
            room.configure_seat(old, {"kind": "open"})
        room.configure_seat(i, {"kind": "human", "client_id": client.id})
        await room.broadcast()
        await hub.push_rooms()
    elif op == "leave_seat":
        old = room.seat_of(client.id)
        if old is not None:
            if room.status == "playing":
                raise IllegalMove("对局进行中不能离座")
            room.configure_seat(old, {"kind": "open"})
        await room.broadcast()
        await hub.push_rooms()
    elif op == "set_seat":
        if not room.can_manage(client.id):
            raise IllegalMove("只有房主可以设置座位")
        if room.status == "playing":
            raise IllegalMove("对局进行中不能修改座位")
        i = int(msg["seat"])
        kind = msg.get("kind")
        if kind == "ai":
            room.configure_seat(i, {"kind": "ai", "model": msg.get("model") or "rule"})
        elif kind == "me":
            old = room.seat_of(client.id)
            if old is not None:
                room.configure_seat(old, {"kind": "open"})
            room.configure_seat(i, {"kind": "human", "client_id": client.id})
        else:
            room.configure_seat(i, {"kind": "open"})
        await room.broadcast()
        await hub.push_rooms()
    elif op == "start" or op == "next_hand":
        if room.status == "playing":
            raise IllegalMove("对局进行中")
        if op == "start" and not room.can_manage(client.id):
            raise IllegalMove("只有房主可以开始")
        async with room.lock:
            await room.start_hand()
        await hub.push_rooms()
    elif op == "restart_hand":
        if not room.can_manage(client.id):
            raise IllegalMove("只有房主可以重开本局")
        async with room.lock:
            await room.restart_hand()
        await hub.push_rooms()
    elif op == "abort_hand":
        if not room.can_manage(client.id):
            raise IllegalMove("只有房主可以结束本局")
        async with room.lock:
            await room.abort_hand()
        await hub.push_rooms()
    elif op == "reset_scores":
        if not room.can_manage(client.id):
            raise IllegalMove("只有房主可以清零总分")
        async with room.lock:
            await room.reset_scores()
    elif op == "action":
        await room.human_action(client.id, msg)
    elif op == "hint":
        c = await room.hint(client.id)
        await client.send({"op": "hint", "comment": c})
    elif op == "settings":
        if not room.can_manage(client.id):
            raise IllegalMove("只有房主可以修改设置")
        if "ai_delay" in msg:
            room.ai_delay = max(0.0, min(5.0, float(msg["ai_delay"])))
        await room.broadcast()
    else:
        raise IllegalMove(f"未知操作 {op}")


async def leave(client, room):
    s = room.seat_of(client.id)
    if s is not None and room.status != "playing":
        room.configure_seat(s, {"kind": "open"})
    room.members.discard(client.id)
    client.room_id = None
    if not room.members and s is not None and room.status != "playing":
        pass
    if not any(cid in hub.clients and hub.clients[cid].ws for cid in room.members) and \
            all(x.kind != "human" for x in room.seats):
        room.cancel()
        hub.rooms.pop(room.id, None)
    else:
        await room.broadcast()
    await hub.push_rooms()
