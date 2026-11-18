"""Test the pure-stdlib HTTP + RFC-6455 WebSocket server (used by Android APK & --lite)."""
import base64
import json
import os
import socket
import struct
import urllib.request

from sandaha.server.lite_server import start_android_server


def _ws_send_masked(sock: socket.socket, obj: dict):
    payload = json.dumps(obj).encode("utf-8")
    mask = b"\x12\x34\x56\x78"
    masked = bytes(payload[i] ^ mask[i & 3] for i in range(len(payload)))
    hdr = bytearray([0x81])
    n = len(payload)
    if n < 126:
        hdr.append(0x80 | n)
    else:
        hdr.append(0x80 | 126)
        hdr.extend(struct.pack("!H", n))
    sock.sendall(bytes(hdr) + mask + masked)


def _ws_recv(sock: socket.socket) -> dict:
    def exact(k):
        buf = bytearray()
        while len(buf) < k:
            chunk = sock.recv(k - len(buf))
            assert chunk
            buf.extend(chunk)
        return bytes(buf)

    b0, b1 = exact(2)
    length = b1 & 0x7F
    if length == 126:
        length = struct.unpack("!H", exact(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", exact(8))[0]
    data = exact(length)
    return json.loads(data.decode("utf-8"))


def test_lite_server_http_and_websocket():
    port = start_android_server(port=0)
    base = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    with opener.open(base + "/") as r:
        html = r.read().decode("utf-8")
        assert "三打哈 AI" in html

    with opener.open(base + "/api/info") as r:
        info = json.loads(r.read().decode("utf-8"))
        assert "version" in info and "rules" in info

    with opener.open(base + "/api/models") as r:
        mods = json.loads(r.read().decode("utf-8"))
        assert len(mods["models"]) >= 1

    # Raw RFC-6455 WebSocket handshake + hello + create_room
    s = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    try:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            f"GET /ws HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        s.sendall(req.encode("ascii"))
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += s.recv(1024)
        header_part, leftover = resp.split(b"\r\n\r\n", 1)
        assert b"101 Switching Protocols" in header_part
        # Put leftover bytes back if any via buffered check; here server sends nothing until 'hello'
        assert leftover == b""

        _ws_send_masked(s, {"op": "hello", "name": "Tester"})
        m1 = _ws_recv(s)
        assert m1["op"] == "welcome" and m1["name"] == "Tester"
        m2 = _ws_recv(s)
        assert m2["op"] == "rooms"
    finally:
        s.close()
