"""One-click launcher: starts the local server and opens the browser."""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import threading
import time
import webbrowser


def main(argv=None):
    multiprocessing.freeze_support()
    if sys.stdout and getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="sandaha", description="三打哈 AI 本地客户端")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = 允许局域网连接")
    ap.add_argument("--port", type=int, default=int(os.environ.get("SANDAHA_PORT", 8765)))
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--lite", action="store_true", help="使用纯标准库轻量 HTTP/WS 服务（无需 FastAPI）")
    args = ap.parse_args(argv)
    os.environ["SANDAHA_PORT"] = str(args.port)

    use_lite = args.lite
    if not use_lite:
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401
        except ImportError:
            use_lite = True

    if use_lite:
        from .server.lite_server import lan_addresses, run_lite_server
    else:
        import uvicorn
        from .server.app import app, lan_addresses

    url = f"http://127.0.0.1:{args.port}"
    print("=" * 56, flush=True)
    print("  三打哈 AI  SanDaHa-Zero", flush=True)
    print(f"  本机访问:   {url}", flush=True)
    if args.host == "0.0.0.0":
        for ip in lan_addresses():
            print(f"  局域网访问: http://{ip}:{args.port}", flush=True)
    print("  按 Ctrl+C 退出", flush=True)
    print("=" * 56, flush=True)
    if not args.no_browser:
        def open_later():
            time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=open_later, daemon=True).start()
    if use_lite:
        run_lite_server(host=args.host, port=args.port)
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
