"""Discover and cache playable models (supports source tree, PyInstaller exe, and Android)."""
from __future__ import annotations

import glob
import json
import os
import sys
import threading

from ..ai import AIAgent, Model
from ..bots import RandomBot, RuleBot

BUNDLE_ROOT = os.path.abspath(
    getattr(sys, "_MEIPASS", os.path.join(os.path.dirname(__file__), "..", ".."))
)
if getattr(sys, "frozen", False):
    ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    ROOT = BUNDLE_ROOT

MODELS_DIR = os.path.join(ROOT, "models")
BUNDLE_MODELS_DIR = os.path.join(BUNDLE_ROOT, "models")
RUNS_DIR = os.path.join(ROOT, "runs")

_cache: dict[str, Model] = {}
_lock = threading.Lock()


def is_standalone() -> bool:
    """Return True when running inside standalone Windows EXE or Android APK (training disabled)."""
    return bool(
        getattr(sys, "frozen", False)
        or os.environ.get("SANDAHA_MOBILE") == "1"
        or os.environ.get("SANDAHA_NO_TRAIN") == "1"
    )


def set_root(root_dir: str):
    """Override ROOT / MODELS_DIR / RUNS_DIR (used by embedded Android runtime)."""
    global ROOT, MODELS_DIR, RUNS_DIR
    ROOT = os.path.abspath(root_dir)
    MODELS_DIR = os.path.join(ROOT, "models")
    RUNS_DIR = os.path.join(ROOT, "runs")


def _allowed_roots() -> tuple[str, ...]:
    if ROOT == BUNDLE_ROOT:
        return (ROOT,)
    return (ROOT, BUNDLE_ROOT)


def rel(path):
    ap = os.path.abspath(path)
    for base in _allowed_roots():
        if ap.startswith(base):
            try:
                return os.path.relpath(ap, base).replace("\\", "/")
            except ValueError:
                pass
    return path.replace("\\", "/")


def _resolve_model_file(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return os.path.abspath(path)
    for base in _allowed_roots():
        cand = os.path.abspath(os.path.join(base, path))
        if os.path.exists(cand):
            return cand
    return os.path.abspath(os.path.join(ROOT, path))


def list_models():
    out = []
    seen_names = set()
    for mdir in (MODELS_DIR,) if MODELS_DIR == BUNDLE_MODELS_DIR else (MODELS_DIR, BUNDLE_MODELS_DIR):
        for p in sorted(glob.glob(os.path.join(mdir, "*.npz"))):
            bn = os.path.basename(p)
            if bn in seen_names:
                continue
            seen_names.add(bn)
            out.append(_info(p, "models"))
    for p in sorted(glob.glob(os.path.join(RUNS_DIR, "*", "final.npz"))):
        out.append(_info(p, "run"))
    for p in sorted(glob.glob(os.path.join(RUNS_DIR, "*", "checkpoints", "*.npz")))[-200:]:
        out.append(_info(p, "checkpoint"))
    return out


def _info(path, kind):
    info = {"path": rel(path), "kind": kind, "size": os.path.getsize(path),
            "mtime": os.path.getmtime(path)}
    run = os.path.basename(os.path.dirname(path if kind == "run" else os.path.dirname(path)))
    if kind == "models":
        info["label"] = os.path.splitext(os.path.basename(path))[0]
    elif kind == "run":
        info["label"] = f"{run} / final"
    else:
        info["label"] = f"{os.path.basename(os.path.dirname(os.path.dirname(path)))} / " \
                        f"{os.path.splitext(os.path.basename(path))[0]}"
    side = os.path.splitext(path)[0] + ".json"
    if os.path.exists(side):
        try:
            with open(side, encoding="utf-8") as f:
                info["card"] = json.load(f)
        except Exception:
            pass
    return info


def default_model_path():
    for mdir in (MODELS_DIR, BUNDLE_MODELS_DIR):
        pref = os.path.join(mdir, "sandaha-best.npz")
        if os.path.exists(pref):
            return rel(pref)
    for mdir in (MODELS_DIR, BUNDLE_MODELS_DIR):
        ms = sorted(glob.glob(os.path.join(mdir, "*.npz")), key=os.path.getmtime)
        if ms:
            return rel(ms[-1])
    return None


def load_model(path) -> Model:
    path = path[:-7] if path.endswith("?search") else path
    full = _resolve_model_file(path)
    if not any(full.startswith(base) for base in _allowed_roots()) or not full.endswith(".npz"):
        raise ValueError("invalid model path")
    with _lock:
        m = _cache.get(full)
        if m is None or m.meta.get("_mtime") != os.path.getmtime(full):
            m = Model.load(full)
            m.meta["_mtime"] = os.path.getmtime(full)
            _cache[full] = m
        return m


def make_agent(spec: str, seed=None):
    """spec: 'rule', 'random' or a model path."""
    if spec in (None, "", "rule"):
        return RuleBot(seed=seed)
    if spec == "random":
        return RandomBot(seed=seed)
    try:
        if spec.endswith("?search"):
            from ..ai.search import SearchAgent
            return SearchAgent(load_model(spec[:-7]), seed=seed)
        return AIAgent(load_model(spec), seed=seed)
    except (FileNotFoundError, ValueError, OSError):
        fallback = default_model_path()
        if fallback and fallback != spec and fallback + "?search" != spec:
            try:
                return AIAgent(load_model(fallback), seed=seed)
            except Exception:
                pass
        return RuleBot(seed=seed)


def agent_label(spec: str):
    if spec in (None, "", "rule"):
        return "规则AI"
    if spec == "random":
        return "随机AI"
    if spec.endswith("?search"):
        return os.path.splitext(os.path.basename(spec[:-7]))[0] + "+搜索"
    return os.path.splitext(os.path.basename(spec))[0]
