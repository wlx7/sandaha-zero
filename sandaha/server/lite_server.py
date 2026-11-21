"""Zero-FastAPI pure-stdlib asyncio HTTP + RFC-6455 WebSocket server.

Used by the standalone Android APK (Chaquopy, where only ``numpy`` and ``segno``
are bundled — no Rust ``pydantic-core`` needed) and as a lightweight fallback
when ``sandaha --lite`` is passed.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import mimetypes
import os
import socket
import struct
import threading
import time
import urllib.parse
import uuid

from .. import __version__
from ..engine import IllegalMove
from ..rules import Rules
from . import profiles, registry
from .rooms import BATTLE, TEACH, Room, destroy_room

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

MIME_OVERRIDES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".ico": "image/x-icon",
}


def lan_addresses():
    primary = None
    for probe in ("8.8.8.8", "10.255.255.255"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((probe, 80))
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
        virtual = ip.endswith(".1")
        home = ip.startswith("192.168.") or ip.startswith("10.")
        return (virtual, not home, ip != primary, ip)

    return sorted(ips, key=rank)


class LiteWebSocket:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self.closed = False

    async def send_text(self, text: str):
        if self.closed:
            raise ConnectionError("ws closed")
        data = text.encode("utf-8")
        n = len(data)
        header = bytearray([0x81])
        if n < 126:
            header.append(n)
        elif n < 65536:
            header.append(126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(127)
            header.extend(struct.pack("!Q", n))
        self.writer.write(bytes(header) + data)
        await self.writer.drain()

    async def receive_text(self) -> str:
        fragments = bytearray()
        while True:
            hdr = await self.reader.readexactly(2)
            b0, b1 = hdr[0], hdr[1]
            fin = (b0 & 0x80) != 0
            opcode = b0 & 0x0F
            masked = (b1 & 0x80) != 0
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack("!H", await self.reader.readexactly(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", await self.reader.readexactly(8))[0]
            if length > 4 * 1024 * 1024:
                raise ConnectionError("frame too large")
            mask_key = await self.reader.readexactly(4) if masked else b""
            payload = bytearray(await self.reader.readexactly(length))
            if masked:
                for i in range(length):
                    payload[i] ^= mask_key[i & 3]
            if opcode == 0x8:  # CLOSE
                await self.close()
                raise EOFError("ws closed by peer")
            if opcode == 0x9:  # PING -> PONG
                pong = bytearray([0x8A, len(payload)]) + payload
                self.writer.write(bytes(pong))
                await self.writer.drain()
                continue
            if opcode == 0xA:  # PONG
                continue
            if opcode in (0x1, 0x0):
                fragments.extend(payload)
                if fin:
                    return fragments.decode("utf-8", errors="replace")

    async def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.writer.write(b"\x88\x00")
                await self.writer.drain()
            except Exception:
                pass
            try:
                self.writer.close()
            except Exception:
                pass


class Client:
    def __init__(self, cid, name):
        self.id = cid
        self.name = name
        self.ws: LiteWebSocket | None = None
        self.room_id = None
        self.profile_id = None    # save slot this device plays as
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
                stale = all(
                    now - self.clients[s.client_id].last_seen > 600
                    for s in humans if s.client_id in self.clients
                )
                if stale:
                    r.cancel()
                    del self.rooms[rid]


hub = Hub()


async def leave(client: Client, room: Room):
    s = room.seat_of(client.id)
    if s is not None and room.status != "playing":
        room.configure_seat(s, {"kind": "open"})
    room.members.discard(client.id)
    client.room_id = None
    if not any(cid in hub.clients and hub.clients[cid].ws for cid in room.members) and \
            all(x.kind != "human" for x in room.seats):
        room.cancel()
        hub.rooms.pop(room.id, None)
    else:
        await room.broadcast()
    await hub.push_rooms()


async def handle(client: Client, msg: dict):
    op = msg["op"]
    room = hub.rooms.get(client.room_id) if client.room_id else None
    if op == "list_rooms":
        await client.send({"op": "rooms", "rooms": hub.room_list()})
    elif op == "ping":
        # Heartbeat: proves the socket is alive, revives a stalled hand, and tells the client
        # which state revision it should be showing.
        if room:
            await room.kick()
        await client.send({"op": "pong", "rev": room.rev if room else None})
    elif op == "sync":
        if room:
            await room.kick()
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
            mine = next((i for i, s in enumerate(specs) if s.get("kind") == "human"), 0)
            for i in range(4):
                if i == mine:
                    specs[i] = {"kind": "human", "client_id": client.id}
                elif specs[i].get("kind") != "ai":
                    specs[i] = {"kind": "ai", "model": msg.get("coach") or registry.default_agent_spec() or "rule"}
        room = Room(
            hub, client.id, msg.get("name"), mode, specs, msg.get("rules"), msg.get("coach"),
            msg.get("ai_delay", 0.9), msg.get("turn_seconds", 30) if mode == BATTLE else 0
        )
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
    elif op.startswith("profile_"):
        await profiles.handle_op(hub, client, msg)
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
        c = await room.hint(client_id=client.id)
        await client.send({"op": "hint", "comment": c})
    elif op == "settings":
        if not room.can_manage(client.id):
            raise IllegalMove("只有房主可以修改设置")
        if "ai_delay" in msg:
            room.ai_delay = max(0.0, min(5.0, float(msg["ai_delay"])))
        await room.broadcast()
    else:
        raise IllegalMove(f"未知操作 {op}")


async def _ws_session(ws: LiteWebSocket):
    client = None
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            op = msg.get("op")
            if op == "hello":
                cid = msg.get("client_id") or uuid.uuid4().hex
                client = hub.clients.get(cid)
                if client is None:
                    client = Client(cid, msg.get("name") or "玩家")
                    hub.clients[cid] = client
                if msg.get("name"):
                    client.name = msg["name"][:16]
                prof = profiles.on_hello(hub, client, msg)   # a profile name wins over the free-text name
                if client.ws is not None and client.ws is not ws:
                    try:
                        await client.ws.close()
                    except Exception:
                        pass
                client.ws = ws
                client.last_seen = time.time()
                await client.send({"op": "welcome", "client_id": cid, "name": client.name, "profile": prof})
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
                await ws.send_text(json.dumps({"op": "error", "msg": "请先登录"}, ensure_ascii=False))
                continue
            client.last_seen = time.time()
            try:
                await handle(client, msg)
            except IllegalMove as e:
                await client.send({"op": "error", "msg": str(e)})
            except (KeyError, ValueError, TypeError, FileNotFoundError) as e:
                await client.send({"op": "error", "msg": f"请求错误: {e}"})
            except Exception as e:
                await client.send({"op": "error", "msg": f"操作失败: {e}"})
    except (EOFError, ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        if client is not None and client.ws is ws:
            client.ws = None
            room = hub.rooms.get(client.room_id) if client.room_id else None
            if room:
                await room.broadcast()
        hub.gc()


def _http_response(writer: asyncio.StreamWriter, status: int, content_type: str, body: bytes, no_cache=False):
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"}.get(status, "OK")
    hdrs = [
        f"HTTP/1.1 {status} {reason}",
        f"Content-Type: {content_type}",
        f"Content-Length: {len(body)}",
        "Connection: close",
    ]
    if no_cache:
        hdrs.append("Cache-Control: no-store, no-cache, must-revalidate")
    writer.write(("\r\n".join(hdrs) + "\r\n\r\n").encode("latin-1") + body)


def _json_response(writer: asyncio.StreamWriter, status: int, obj: dict):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    _http_response(writer, status, "application/json; charset=utf-8", data, no_cache=True)


async def _handle_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        req_line = await asyncio.wait_for(reader.readline(), timeout=10.0)
        if not req_line:
            writer.close()
            return
        parts = req_line.decode("latin-1", errors="replace").strip().split(" ")
        if len(parts) < 2:
            writer.close()
            return
        method, raw_target = parts[0].upper(), parts[1]
        headers: dict[str, str] = {}
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            if not line or line in (b"\r\n", b"\n"):
                break
            s = line.decode("latin-1", errors="replace").strip()
            if ":" in s:
                k, v = s.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        parsed = urllib.parse.urlsplit(raw_target)
        path = parsed.path or "/"
        query = dict(urllib.parse.parse_qsl(parsed.query))

        # WebSocket Upgrade
        if path == "/ws" and headers.get("upgrade", "").lower() == "websocket":
            ws_key = headers.get("sec-websocket-key", "")
            accept = base64.b64encode(
                hashlib.sha1((ws_key + WS_GUID).encode("ascii")).digest()
            ).decode("ascii")
            resp = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            )
            writer.write(resp.encode("ascii"))
            await writer.drain()
            await _ws_session(LiteWebSocket(reader, writer))
            return

        body_bytes = b""
        clen = int(headers.get("content-length", "0") or "0")
        if clen > 0:
            body_bytes = await asyncio.wait_for(reader.readexactly(clen), timeout=15.0)

        # Static files
        if method == "GET" and (path == "/" or path == "/favicon.ico" or path.startswith("/static/")):
            if path == "/":
                fpath = os.path.join(WEB_DIR, "index.html")
            elif path == "/favicon.ico":
                fpath = os.path.join(WEB_DIR, "icon.svg")
            else:
                rel = os.path.normpath(urllib.parse.unquote(path[len("/static/"):]))
                fpath = os.path.abspath(os.path.join(WEB_DIR, rel))
                if not fpath.startswith(os.path.abspath(WEB_DIR)):
                    _http_response(writer, 404, "text/plain", b"Not Found")
                    await writer.drain()
                    writer.close()
                    return
            if os.path.isfile(fpath):
                ext = os.path.splitext(fpath)[1].lower()
                ctype = MIME_OVERRIDES.get(ext) or mimetypes.guess_type(fpath)[0] or "application/octet-stream"
                with open(fpath, "rb") as f:
                    data = f.read()
                _http_response(writer, 200, ctype, data, no_cache=True)
            else:
                _http_response(writer, 404, "text/plain", b"Not Found")
            await writer.drain()
            writer.close()
            return

        # REST API
        port = int(os.environ.get("SANDAHA_PORT", "8765"))
        standalone = registry.is_standalone()
        if method == "GET" and path == "/api/info":
            torch_st = {"available": False}
            if not standalone:
                from . import training
                torch_st = training.torch_info()
            _json_response(writer, 200, {
                "version": __version__,
                "lan": [f"http://{ip}:{port}" for ip in lan_addresses()],
                "port": port,
                "default_model": registry.default_model_path(),
                "rules": Rules().to_dict(),
                "standalone": standalone,
                "mobile": os.environ.get("SANDAHA_MOBILE") == "1",
                "can_train": not standalone,
                "torch": torch_st,
            })
        elif method == "GET" and path == "/api/models":
            _json_response(writer, 200, {
                "models": registry.list_models(),
                "default": registry.default_agent_spec(),
            })
        elif method == "GET" and path == "/api/qr":
            url = query.get("url", "")
            try:
                import segno
                buf = io.BytesIO()
                segno.make(url, error="m").save(buf, kind="svg", scale=5, border=2, dark="#1b1b1b", light="#ffffff")
                _http_response(writer, 200, "image/svg+xml", buf.getvalue())
            except Exception:
                _http_response(writer, 200, "image/svg+xml", b"<svg xmlns='http://www.w3.org/2000/svg'/>")
        elif standalone and (path.startswith("/api/train") or path == "/api/eval"):
            _json_response(writer, 404, {"detail": "独立客户端已移除训练模块"})
        elif method == "GET" and path == "/api/train/presets":
            from . import training
            from ..train.config import TrainConfig
            _json_response(writer, 200, {
                "presets": training.presets(),
                "defaults": TrainConfig().__dict__,
                "torch": training.torch_info(),
                "cpus": os.cpu_count(),
            })
        elif method == "GET" and path == "/api/train/runs":
            from . import training
            _json_response(writer, 200, {"runs": training.list_runs()})
        elif method == "GET" and path.startswith("/api/train/run/"):
            from . import training
            name = urllib.parse.unquote(path[len("/api/train/run/"):])
            _json_response(writer, 200, training.run_detail(name))
        elif method == "GET" and path == "/api/train/install":
            from . import training
            _json_response(writer, 200, training.install_status())
        elif method == "POST" and path == "/api/eval":
            from . import training
            payload = json.loads(body_bytes.decode("utf-8") or "{}")
            a, b = payload.get("a"), payload.get("b")
            deals = min(500, int(payload.get("deals", 100)))
            res = await asyncio.to_thread(training.quick_eval, a, b, deals)
            _json_response(writer, 200, res)
        elif method == "POST" and path == "/api/train/start":
            from . import training
            payload = json.loads(body_bytes.decode("utf-8") or "{}")
            _json_response(writer, 200, training.start(payload))
        else:
            _json_response(writer, 404, {"detail": "Not Found"})
    except (ValueError, FileNotFoundError) as e:
        try:
            _json_response(writer, 400, {"detail": str(e)})
        except Exception:
            pass
    except Exception:
        pass
    finally:
        try:
            await writer.drain()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass


async def serve_async(host: str = "0.0.0.0", port: int = 8765):
    os.environ["SANDAHA_PORT"] = str(port)
    srv = await asyncio.start_server(_handle_conn, host=host, port=port)
    async with srv:
        await srv.serve_forever()


def run_lite_server(host: str = "0.0.0.0", port: int = 8765):
    asyncio.run(serve_async(host=host, port=port))


_bg_port: int | None = None
_bg_lock = threading.Lock()


def start_android_server(port: int = 8765, root_dir: str | None = None) -> int:
    """Entry point called from Android MainActivity via Chaquopy."""
    global _bg_port, WEB_DIR
    with _bg_lock:
        if _bg_port is not None:
            return _bg_port
        os.environ["SANDAHA_MOBILE"] = "1"
        if root_dir:
            registry.set_root(root_dir)
            cand_web = os.path.join(root_dir, "web")
            if os.path.isdir(cand_web):
                WEB_DIR = cand_web
        # Pick free port if 8765 is busy
        chosen = port
        for p in (port, 18765, 28765, 0):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", p))
                chosen = s.getsockname()[1]
                s.close()
                break
            except OSError:
                continue
        os.environ["SANDAHA_PORT"] = str(chosen)

        ready = threading.Event()

        def _worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _main():
                srv = await asyncio.start_server(_handle_conn, host="0.0.0.0", port=chosen)
                ready.set()
                async with srv:
                    await srv.serve_forever()

            loop.run_until_complete(_main())

        t = threading.Thread(target=_worker, daemon=True, name="sandaha-lite-server")
        t.start()
        ready.wait(timeout=8.0)
        _bg_port = chosen
        return chosen
