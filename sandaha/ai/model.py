"""Pure-numpy inference for the Q-networks (no PyTorch needed to play).

Supports both:
  * Legacy ``v1`` MLP models (``arch_type == "mlp_v1"``)
  * ``dueling_film_v2`` models: Pre-LN Residual State Tower + Value/Advantage
    Dueling decomposition + FiLM multiplicative State-Action gating + LeakyReLU.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .features import DIMS, PHASES, dims_for

PRESETS = {
    "tiny": {"play": [256, 192], "aux": [160, 128]},
    "small": {"play": [512, 384, 192], "aux": [256, 160]},
    "medium": {"play": [768, 512, 256], "aux": [384, 256]},
    "large": {"play": [1024, 768, 384], "aux": [512, 256]},
    "xlarge": {"play": [1536, 1024, 512], "aux": [768, 384]},
}
OUT = 3  # [q (expected score / 9), win logit, final 闲 points / 400]
LRELU_SLOPE = np.float32(0.05)
LN_EPS = np.float32(1e-5)


def hidden_for(phase, arch):
    return arch["play"] if phase == "play" else arch["aux"]


def _ln(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = x.mean(axis=-1, keepdims=True)
    d = x - m
    inv = np.float32(1.0) / np.sqrt((d * d).mean(axis=-1, keepdims=True) + LN_EPS)
    return d * (inv * g) + b


def _lrelu_inplace(x: np.ndarray) -> np.ndarray:
    np.maximum(x, LRELU_SLOPE * x, out=x)
    return x


def param_count(arch, arch_type: str = "dueling_film_v2", feat_ver: int = 2) -> int:
    dm = dims_for(feat_ver)
    if arch_type != "dueling_film_v2":
        n = 0
        for p in PHASES:
            dims = [sum(dm[p])] + list(hidden_for(p, arch)) + [OUT]
            n += sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
        return n
    n = 0
    for p in PHASES:
        s_dim, a_dim = dm[p]
        dims = list(hidden_for(p, arch))
        sh = dims[0]
        j_dims = dims[1:] if len(dims) > 1 else [sh]
        j0 = j_dims[0]
        # State stem + Pre-LN ResBlock + 4 LayerNorms (ln0, ln1, ln2, ln3)
        n += (s_dim * sh + sh) + 2 * (sh * sh + sh) + 4 * (2 * sh)
        # Value head + FiLM shift/gate + Action stem
        n += (sh * OUT + OUT) + 2 * (sh * j0 + j0) + (a_dim * j0 + j0)
        # Joint tail layers + LayerNorms + Advantage head
        for k in range(len(j_dims) - 1):
            d_in, d_out = j_dims[k], j_dims[k + 1]
            n += (d_in * d_out + d_out) + 2 * d_out
        n += j_dims[-1] * OUT + OUT
    return n


class NumpyNet:
    """Legacy v1 ReLU MLP."""

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
        sizes = [len(a) for a in actions_list]
        hs = states @ self.W0s + self.b0
        A = np.concatenate(actions_list, 0)
        h = A @ self.W0a + np.repeat(hs, sizes, axis=0)
        np.maximum(h, 0, out=h)
        out = self._tail(h)
        return np.split(out, np.cumsum(sizes)[:-1])


class DuelingNumpyNet:
    """v2 Pre-LN Residual State Tower + FiLM-Gated Dueling Advantage Network."""

    def __init__(self, params: dict, prefix: str):
        def g(k):
            return np.ascontiguousarray(params[f"{prefix}.{k}"].astype(np.float32))

        self.Ws0, self.bs0 = g("Ws0"), g("bs0")
        self.ln0_g, self.ln0_b = g("ln0_g"), g("ln0_b")
        self.ln1_g, self.ln1_b = g("ln1_g"), g("ln1_b")
        self.Wsr1, self.bsr1 = g("Wsr1"), g("bsr1")
        self.ln2_g, self.ln2_b = g("ln2_g"), g("ln2_b")
        self.Wsr2, self.bsr2 = g("Wsr2"), g("bsr2")
        self.ln3_g, self.ln3_b = g("ln3_g"), g("ln3_b")
        self.Wv, self.bv = g("Wv"), g("bv")
        Wss, bss = g("Wss"), g("bss")
        Wsg, bsg = g("Wsg"), g("bsg")
        self.j0 = Wss.shape[1]
        # Fuse Wss and Wsg into a single matrix multiply on the state vector
        self.Wssg = np.ascontiguousarray(np.concatenate([Wss, Wsg], axis=1))
        self.bssg = np.ascontiguousarray(np.concatenate([bss, bsg], axis=0))
        self.Wa0, self.ba0 = g("Wa0"), g("ba0")
        self.tail = []
        k = 0
        while f"{prefix}.Wjt{k}" in params:
            self.tail.append((
                g(f"Wjt{k}"), g(f"bjt{k}"),
                g(f"lnjt{k}_g"), g(f"lnjt{k}_b"),
            ))
            k += 1
        self.Wadv, self.badv = g("Wadv"), g("badv")

    def _state_forward(self, s: np.ndarray):
        h = _lrelu_inplace(_ln(s @ self.Ws0 + self.bs0, self.ln0_g, self.ln0_b))
        r = _lrelu_inplace(_ln(h, self.ln1_g, self.ln1_b))
        r = r @ self.Wsr1 + self.bsr1
        r = _lrelu_inplace(_ln(r, self.ln2_g, self.ln2_b))
        r = r @ self.Wsr2 + self.bsr2
        h = h + r
        hn = _ln(h, self.ln3_g, self.ln3_b)
        v = hn @ self.Wv + self.bv
        ssg = hn @ self.Wssg + self.bssg
        j0 = self.j0
        s_shift = ssg[..., :j0]
        s_gate = np.tanh(ssg[..., j0:])
        return v, s_shift, s_gate

    def _joint_tail(self, j: np.ndarray) -> np.ndarray:
        _lrelu_inplace(j)
        for Wt, bt, lg, lb in self.tail:
            j = _lrelu_inplace(_ln(j @ Wt + bt, lg, lb))
        return j @ self.Wadv + self.badv

    def __call__(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        v, s_shift, s_gate = self._state_forward(state)
        a_emb = actions @ self.Wa0 + self.ba0
        j = a_emb * (np.float32(1.0) + s_gate) + s_shift
        adv = self._joint_tail(j)
        return adv + v

    def batch(self, states: np.ndarray, actions_list: list[np.ndarray]) -> list[np.ndarray]:
        sizes = [len(a) for a in actions_list]
        v, s_shift, s_gate = self._state_forward(states)
        A = np.concatenate(actions_list, 0)
        a_emb = A @ self.Wa0 + self.ba0
        gate_rep = np.repeat(np.float32(1.0) + s_gate, sizes, axis=0)
        shift_rep = np.repeat(s_shift, sizes, axis=0)
        j = a_emb * gate_rep + shift_rep
        adv = self._joint_tail(j)
        out = adv + np.repeat(v, sizes, axis=0)
        return np.split(out, np.cumsum(sizes)[:-1])


class Model:
    def __init__(self, params: dict, meta: dict):
        self.meta = meta
        self.params = params
        self.arch_type = meta.get("arch_type") or (
            "dueling_film_v2" if "play.Ws0" in params else "mlp_v1"
        )
        self.feat_ver = int(meta.get("feat_ver", 2 if self.arch_type == "dueling_film_v2" else 1))
        dm = dims_for(self.feat_ver)
        self.nets = {}
        if self.arch_type == "dueling_film_v2":
            for p in PHASES:
                self.nets[p] = DuelingNumpyNet(params, p)
        else:
            for p in PHASES:
                Ws, bs = [], []
                i = 0
                while f"{p}.W{i}" in params:
                    Ws.append(params[f"{p}.W{i}"].astype(np.float32))
                    bs.append(params[f"{p}.b{i}"].astype(np.float32))
                    i += 1
                self.nets[p] = NumpyNet(Ws, bs, dm[p][0])

    @property
    def name(self):
        return self.meta.get("name", "model")

    def q(self, phase, state, actions):
        return self.nets[phase](state, actions)

    def q_batch(self, phase, states, actions_list):
        return self.nets[phase].batch(np.asarray(states, dtype=np.float32), actions_list)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            params = {k: z[k] for k in z.files if k != "__meta__"}
            meta = json.loads(str(z["__meta__"])) if "__meta__" in z.files else {}
        meta.setdefault("name", os.path.splitext(os.path.basename(path))[0])
        meta["path"] = os.path.abspath(path)
        return cls(params, meta)

    @classmethod
    def random(cls, arch=None, seed=0, name="random-init", arch_type="dueling_film_v2", feat_ver=2):
        arch = arch or PRESETS["small"]
        rng = np.random.default_rng(seed)
        dm = dims_for(feat_ver)
        params = {}
        if arch_type != "dueling_film_v2":
            for p in PHASES:
                dims = [sum(dm[p])] + list(hidden_for(p, arch)) + [OUT]
                for i in range(len(dims) - 1):
                    params[f"{p}.W{i}"] = (rng.standard_normal((dims[i], dims[i + 1])) *
                                           np.sqrt(2 / dims[i])).astype(np.float32)
                    params[f"{p}.b{i}"] = np.zeros(dims[i + 1], np.float32)
            return cls(params, {"name": name, "arch": arch, "arch_type": "mlp_v1", "feat_ver": feat_ver})

        def lin(d_in, d_out, scale=1.0):
            w = (rng.standard_normal((d_in, d_out)) * np.sqrt(2.0 / d_in) * scale).astype(np.float32)
            b = np.zeros(d_out, np.float32)
            return w, b

        for p in PHASES:
            s_dim, a_dim = dm[p]
            dims = list(hidden_for(p, arch))
            sh = dims[0]
            j_dims = dims[1:] if len(dims) > 1 else [sh]
            j0 = j_dims[0]
            params[f"{p}.Ws0"], params[f"{p}.bs0"] = lin(s_dim, sh)
            for lnk in ("ln0", "ln1", "ln2", "ln3"):
                params[f"{p}.{lnk}_g"] = np.ones(sh, np.float32)
                params[f"{p}.{lnk}_b"] = np.zeros(sh, np.float32)
            params[f"{p}.Wsr1"], params[f"{p}.bsr1"] = lin(sh, sh)
            params[f"{p}.Wsr2"], params[f"{p}.bsr2"] = lin(sh, sh, scale=0.2)
            params[f"{p}.Wv"], params[f"{p}.bv"] = lin(sh, OUT, scale=0.1)
            params[f"{p}.Wss"], params[f"{p}.bss"] = lin(sh, j0)
            params[f"{p}.Wsg"], params[f"{p}.bsg"] = lin(sh, j0, scale=0.2)
            params[f"{p}.Wa0"], params[f"{p}.ba0"] = lin(a_dim, j0)
            for k in range(len(j_dims) - 1):
                d_in, d_out = j_dims[k], j_dims[k + 1]
                params[f"{p}.Wjt{k}"], params[f"{p}.bjt{k}"] = lin(d_in, d_out)
                params[f"{p}.lnjt{k}_g"] = np.ones(d_out, np.float32)
                params[f"{p}.lnjt{k}_b"] = np.zeros(d_out, np.float32)
            params[f"{p}.Wadv"], params[f"{p}.badv"] = lin(j_dims[-1], OUT, scale=0.1)
        return cls(params, {"name": name, "arch": arch, "arch_type": "dueling_film_v2", "feat_ver": feat_ver})


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
