"""Discover and cache playable models."""
from __future__ import annotations

import glob
import json
import os
import threading
import time

from ..ai import AIAgent, Model
from ..bots import RandomBot, RuleBot

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(ROOT, "models")
RUNS_DIR = os.path.join(ROOT, "runs")

_cache: dict[str, Model] = {}
_lock = threading.Lock()


def rel(path):
    try:
        return os.path.relpath(path, ROOT).replace("\\", "/")
    except ValueError:
        return path


def list_models():
    out = []
    for p in sorted(glob.glob(os.path.join(MODELS_DIR, "*.npz"))):
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
    pref = os.path.join(MODELS_DIR, "sandaha-best.npz")
    if os.path.exists(pref):
        return rel(pref)
    ms = sorted(glob.glob(os.path.join(MODELS_DIR, "*.npz")), key=os.path.getmtime)
    return rel(ms[-1]) if ms else None


def load_model(path) -> Model:
    path = path[:-7] if path.endswith("?search") else path
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    full = os.path.abspath(full)
    if not full.startswith(ROOT) or not full.endswith(".npz"):
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
    if spec.endswith("?search"):
        from ..ai.search import SearchAgent
        return SearchAgent(load_model(spec[:-7]), seed=seed)
    return AIAgent(load_model(spec), seed=seed)


def agent_label(spec: str):
    if spec in (None, "", "rule"):
        return "规则AI"
    if spec == "random":
        return "随机AI"
    if spec.endswith("?search"):
        return os.path.splitext(os.path.basename(spec[:-7]))[0] + "+搜索"
    return os.path.splitext(os.path.basename(spec))[0]
