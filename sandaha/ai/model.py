"""Pure-numpy inference for the Q-networks (no PyTorch needed to play).

File format (.npz): for each phase p and layer i: ``p.W{i}`` (in, out) and
``p.b{i}``, plus ``__meta__`` holding a JSON string.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .features import DIMS, PHASES

PRESETS = {
    "tiny": {"play": [256, 256], "aux": [128, 64]},
    "small": {"play": [512, 256, 256], "aux": [256, 128]},
    "medium": {"play": [768, 512, 256], "aux": [256, 128]},
    "large": {"play": [1024, 1024, 512, 256], "aux": [512, 256]},
    "xlarge": {"play": [2048, 1024, 1024, 512], "aux": [512, 256]},
}
OUT = 3  # [q (expected score / 9), win logit, final 闲 points / 400]


def hidden_for(phase, arch):
    return arch["play"] if phase == "play" else arch["aux"]


def param_count(arch):
    n = 0
    for p in PHASES:
        dims = [sum(DIMS[p])] + list(hidden_for(p, arch)) + [OUT]
        n += sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
    return n


class NumpyNet:
    def __init__(self, Ws, bs, s_dim):
        self.W0s = np.ascontiguousarray(Ws[0][:s_dim])
        self.W0a = np.ascontiguousarray(Ws[0][s_dim:])
        self.Ws = Ws[1:]
        self.b0 = bs[0]
        self.bs = bs[1:]

    def __call__(self, state, actions):
        h = state @ self.W0s + self.b0
        h = actions @ self.W0a + h
        np.maximum(h, 0, out=h)
        return self._tail(h)

    def _tail(self, h):
        n = len(self.Ws)
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            h = h @ W + b
            if i < n - 1:
                np.maximum(h, 0, out=h)
        return h

    def batch(self, states, actions_list):
        """Evaluate many decisions at once (one weight pass for all of them).

        states: (n, s_dim); actions_list: n matrices (m_i, a_dim).
        Returns a list of (m_i, OUT) arrays."""
        sizes = [len(a) for a in actions_list]
        hs = states @ self.W0s + self.b0                      # (n, H)
        A = np.concatenate(actions_list, 0)                    # (M, a_dim)
        h = A @ self.W0a + np.repeat(hs, sizes, axis=0)
        np.maximum(h, 0, out=h)
        out = self._tail(h)
        return np.split(out, np.cumsum(sizes)[:-1])


class Model:
    def __init__(self, params: dict, meta: dict):
        self.meta = meta
        self.params = params
        self.nets = {}
        for p in PHASES:
            Ws, bs = [], []
            i = 0
            while f"{p}.W{i}" in params:
                Ws.append(params[f"{p}.W{i}"].astype(np.float32))
                bs.append(params[f"{p}.b{i}"].astype(np.float32))
                i += 1
            self.nets[p] = NumpyNet(Ws, bs, DIMS[p][0])

    @property
    def name(self):
        return self.meta.get("name", "model")

    def q(self, phase, state, actions):
        return self.nets[phase](state, actions)

    def q_batch(self, phase, states, actions_list):
        return self.nets[phase].batch(np.stack(states), actions_list)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            params = {k: z[k] for k in z.files if k != "__meta__"}
            meta = json.loads(str(z["__meta__"])) if "__meta__" in z.files else {}
        meta.setdefault("name", os.path.splitext(os.path.basename(path))[0])
        meta["path"] = os.path.abspath(path)
        return cls(params, meta)

    @classmethod
    def random(cls, arch=None, seed=0, name="random-init"):
        arch = arch or PRESETS["small"]
        rng = np.random.default_rng(seed)
        params = {}
        for p in PHASES:
            dims = [sum(DIMS[p])] + list(hidden_for(p, arch)) + [OUT]
            for i in range(len(dims) - 1):
                params[f"{p}.W{i}"] = (rng.standard_normal((dims[i], dims[i + 1])) *
                                       np.sqrt(2 / dims[i])).astype(np.float32)
                params[f"{p}.b{i}"] = np.zeros(dims[i + 1], np.float32)
        return cls(params, {"name": name, "arch": arch})


def replace_retry(src, dst, tries=50):
    """os.replace that tolerates Windows sharing violations (reader holding dst)."""
    import time
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.02 * (i + 1))


def save_params(path, params: dict, meta: dict):
    tmp = path + ".tmp.npz"
    np.savez(tmp, __meta__=np.array(json.dumps(meta, ensure_ascii=False)), **params)
    replace_retry(tmp, path)
