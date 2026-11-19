"""Build standalone one-click Windows executable (SanDaHa-Zero-Windows-x64.exe).

Usage:
    python scripts/build_windows.py
Output:
    dist/SanDaHa-Zero-Windows-x64.exe
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import zlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _make_png_icon(size: int = 64) -> bytes:
    """Create a crisp RGBA PNG icon (emerald felt + gold-bordered card) using pure stdlib."""
    pixels = bytearray()
    for y in range(size):
        pixels.append(0)  # filter byte 0
        for x in range(size):
            # Rounded square background (#0f3d2e emerald with #e3b341 gold border)
            dx = min(x, size - 1 - x)
            dy = min(y, size - 1 - y)
            r_corner = 10
            if dx < r_corner and dy < r_corner and (r_corner - dx) ** 2 + (r_corner - dy) ** 2 > r_corner ** 2:
                pixels.extend((0, 0, 0, 0))
                continue
            border = dx < 3 or dy < 3
            if border:
                pixels.extend((227, 179, 65, 255))
                continue
            # Center white playing card [16..48, 10..54]
            if 15 <= x <= 48 and 9 <= y <= 54:
                if x in (15, 48) or y in (9, 54):
                    pixels.extend((227, 179, 65, 255))
                else:
                    # Red diamond / heart pip in center of card
                    cx, cy = 31.5, 31.5
                    if abs(x - cx) + abs(y - cy) <= 9.5:
                        pixels.extend((198, 40, 40, 255))
                    else:
                        pixels.extend((255, 253, 248, 255))
            else:
                pixels.extend((15, 61, 46, 255))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = chunk(b"IHDR", struct.pack("!IIBBBBB", size, size, 8, 6, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(bytes(pixels), 9))
    iend = chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def make_ico(path: str):
    png = _make_png_icon(64)
    # ICO header (1 image, 64x64, PNG-compressed)
    ico_hdr = struct.pack("<HHH", 0, 1, 1)
    ico_dir = struct.pack("<BBBBHHII", 64, 64, 0, 0, 1, 32, len(png), 6 + 16)
    with open(path, "wb") as f:
        f.write(ico_hdr + ico_dir + png)


def main():
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    ico_path = os.path.join(ROOT, "build", "sandaha.ico")
    make_ico(ico_path)

    sep = ";" if os.name == "nt" else ":"
    add_data = [
        "--add-data", f"{os.path.join(ROOT, 'sandaha', 'web')}{sep}sandaha/web",
        "--add-data", f"{os.path.join(ROOT, 'models', 'sandaha-best.npz')}{sep}models",
    ]
    lite_model = os.path.join(ROOT, "models", "sandaha-lite.npz")
    if os.path.exists(lite_model):
        add_data += ["--add-data", f"{lite_model}{sep}models"]

    excludes = []
    for mod in (
        "sandaha.train",
        "sandaha.server.training",
        "torch", "torchvision", "torchaudio", "triton",
        "pytest", "scipy", "matplotlib", "pandas",
    ):
        excludes += ["--exclude-module", mod]

    hidden = []
    for mod in (
        "sandaha.server.app",
        "sandaha.server.lite_server",
        "sandaha.ai.search",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "segno",
    ):
        hidden += ["--hidden-import", mod]

    entry_wrapper = os.path.join(ROOT, "build", "_exe_entry.py")
    with open(entry_wrapper, "w", encoding="utf-8") as f:
        f.write("from sandaha.launcher import main\nif __name__ == '__main__':\n    main()\n")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name", "SanDaHa-Zero-Windows-x64",
        "--specpath", os.path.join(ROOT, "build"),
        "--icon", ico_path,
        "--paths", ROOT,
        *add_data,
        *excludes,
        *hidden,
        entry_wrapper,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)
    out_exe = os.path.join(ROOT, "dist", "SanDaHa-Zero-Windows-x64.exe")
    print(f"Built: {out_exe} ({os.path.getsize(out_exe) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
