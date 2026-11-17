"""Launch / monitor training runs as subprocesses (used by the web dashboard)."""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from ..ai.model import PRESETS, param_count
from ..train.config import TrainConfig
from .registry import MODELS_DIR, ROOT, RUNS_DIR, rel

_procs: dict[str, subprocess.Popen] = {}
_install = {"state": "idle", "log": ""}


def torch_info():
    try:
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            return {"available": False}
        import torch
        return {"available": True, "version": torch.__version__, "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as e:
        return {"available": False, "error": str(e)}


def presets():
    return {k: {"arch": v, "params": param_count(v)} for k, v in PRESETS.items()}


def _read_jsonl(path, limit=5000):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if len(out) > limit:
        step = len(out) / limit
        out = [out[int(i * step)] for i in range(limit)] + [out[-1]]
    return out


def _status(run_dir):
    p = os.path.join(run_dir, "status.json")
    st = {}
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                st = json.load(f)
        except Exception:
            st = {}
    name = os.path.basename(run_dir)
    proc = _procs.get(name)
    if st.get("state") in ("running", "starting", "evaluating"):
        alive = proc is not None and proc.poll() is None
        if proc is None:
            # started elsewhere (CLI / previous session): trust a fresh heartbeat
            m = os.path.join(run_dir, "metrics.jsonl")
            alive = os.path.exists(m) and time.time() - os.path.getmtime(m) < 60
        if not alive:
            st["state"] = "stale"
    return st


def list_runs():
    runs = []
    for d in sorted(glob.glob(os.path.join(RUNS_DIR, "*")), key=os.path.getmtime, reverse=True):
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "config.json")):
            continue
        st = _status(d)
        evals = _read_jsonl(os.path.join(d, "evals.jsonl"))
        last = evals[-1] if evals else None
        runs.append({"name": os.path.basename(d), "status": st,
                     "last_eval": last, "has_final": os.path.exists(os.path.join(d, "final.npz"))})
    return runs


def run_detail(name):
    d = _run_dir(name)
    cfg = TrainConfig.load(os.path.join(d, "config.json"))
    ckpts = [rel(p) for p in sorted(glob.glob(os.path.join(d, "checkpoints", "*.npz")))]
    log = ""
    lp = os.path.join(d, "train.log")
    if os.path.exists(lp):
        with open(lp, encoding="utf-8", errors="replace") as f:
            log = f.read()[-6000:]
    return {"name": name, "config": cfg.__dict__, "status": _status(d),
            "metrics": _read_jsonl(os.path.join(d, "metrics.jsonl")),
            "evals": _read_jsonl(os.path.join(d, "evals.jsonl")),
            "checkpoints": ckpts, "log": log,
            "params": param_count(cfg.arch())}


def _run_dir(name):
    if not re.fullmatch(r"[\w\-.]+", name or ""):
        raise ValueError("非法的训练名称")
    return os.path.join(RUNS_DIR, name)


def start(cfg_dict):
    cfg = TrainConfig.from_dict(cfg_dict)
    if not cfg.name:
        cfg.name = time.strftime("run-%Y%m%d-%H%M%S")
    d = _run_dir(cfg.name)
    if os.path.exists(os.path.join(d, "status.json")) and _status(d).get("state") in ("running", "starting"):
        raise ValueError("该训练正在运行")
    if not torch_info().get("available"):
        raise ValueError("未安装 PyTorch，请先点击“安装训练依赖”")
    for k in ("baseline", "init_from"):
        v = getattr(cfg, k)
        if v and not os.path.isabs(v):
            setattr(cfg, k, os.path.join(ROOT, v))
    os.makedirs(d, exist_ok=True)
    cfg.save(os.path.join(d, "config.json"))
    log = open(os.path.join(d, "train.log"), "a", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    p = subprocess.Popen([sys.executable, "-m", "sandaha.train.cli", "train", "--config",
                          os.path.join(d, "config.json")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         env=env, creationflags=flags)
    _procs[cfg.name] = p
    return {"name": cfg.name, "pid": p.pid}


def stop(name):
    d = _run_dir(name)
    with open(os.path.join(d, "STOP"), "w") as f:
        f.write("stop")
    return {"ok": True}


def delete(name):
    d = _run_dir(name)
    if _status(d).get("state") in ("running", "starting", "evaluating"):
        raise ValueError("请先停止训练")
    shutil.rmtree(d)
    return {"ok": True}


def publish(name, src, target):
    """Copy a run's model into models/ so it shows up as a named AI."""
    if not re.fullmatch(r"[\w\-.]+", target or ""):
        raise ValueError("非法的模型名称")
    d = _run_dir(name)
    src_path = os.path.join(ROOT, src) if src else os.path.join(d, "final.npz")
    src_path = os.path.abspath(src_path)
    if not src_path.startswith(os.path.abspath(d)) or not os.path.exists(src_path):
        raise ValueError("找不到模型文件")
    os.makedirs(MODELS_DIR, exist_ok=True)
    dst = os.path.join(MODELS_DIR, target + ".npz")
    shutil.copyfile(src_path, dst)
    evals = _read_jsonl(os.path.join(d, "evals.jsonl"))
    card = {"run": name, "source": rel(src_path), "published": time.time(),
            "last_eval": evals[-1] if evals else None}
    with open(os.path.join(MODELS_DIR, target + ".json"), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=1)
    return {"path": rel(dst)}


# ------------------------------------------------------------ installer --

def install_status():
    return dict(_install, torch=torch_info())


def install_torch():
    if _install["state"] == "running":
        return install_status()

    def work():
        _install.update(state="running", log="")
        has_gpu = shutil.which("nvidia-smi") is not None
        index = "https://download.pytorch.org/whl/cu124" if has_gpu else "https://download.pytorch.org/whl/cpu"
        uv = shutil.which("uv")
        if uv:
            cmd = [uv, "pip", "install", "--python", sys.executable, "torch", "--index-url", index]
        else:
            subprocess.run([sys.executable, "-m", "ensurepip"], capture_output=True)
            cmd = [sys.executable, "-m", "pip", "install", "torch", "--index-url", index]
        _install["log"] = "$ " + " ".join(cmd) + "\n"
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace")
            for line in p.stdout:
                _install["log"] = (_install["log"] + line)[-8000:]
            p.wait()
            _install["state"] = "done" if p.returncode == 0 else "failed"
        except Exception as e:
            _install["log"] += f"\n{e}"
            _install["state"] = "failed"

    threading.Thread(target=work, daemon=True).start()
    return install_status()


def quick_eval(a, b, deals=100):
    from ..arena import duel
    from .registry import make_agent
    r = duel(make_agent(a, 1), make_agent(b, 2), int(deals), seed=4242)
    return r
