from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import nn

from ..ai.features import PHASES, dims_for
from ..ai.model import LRELU_SLOPE, LN_EPS, OUT, hidden_for

SLOPE = float(LRELU_SLOPE)
EPS = float(LN_EPS)


class QNet(nn.Module):
    """Legacy v1 ReLU MLP."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        dims = [in_dim] + list(hidden) + [OUT]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def forward(self, s, a=None):
        x = s if a is None else torch.cat([s, a], dim=-1)
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x, None


class DuelingQNet(nn.Module):
    """v2 Pre-LN Residual State Tower + FiLM-Gated Dueling Advantage Network."""

    def __init__(self, s_dim: int, a_dim: int, hidden: list[int]):
        super().__init__()
        dims = list(hidden)
        sh = dims[0]
        j_dims = dims[1:] if len(dims) > 1 else [sh]
        j0 = j_dims[0]

        self.s0 = nn.Linear(s_dim, sh)
        self.ln0 = nn.LayerNorm(sh, eps=EPS)
        self.ln1 = nn.LayerNorm(sh, eps=EPS)
        self.sr1 = nn.Linear(sh, sh)
        self.ln2 = nn.LayerNorm(sh, eps=EPS)
        self.sr2 = nn.Linear(sh, sh)
        self.ln3 = nn.LayerNorm(sh, eps=EPS)

        self.v_head = nn.Linear(sh, OUT)
        self.s_shift = nn.Linear(sh, j0)
        self.s_gate = nn.Linear(sh, j0)
        self.a0 = nn.Linear(a_dim, j0)

        self.j_linears = nn.ModuleList()
        self.j_lns = nn.ModuleList()
        for k in range(len(j_dims) - 1):
            self.j_linears.append(nn.Linear(j_dims[k], j_dims[k + 1]))
            self.j_lns.append(nn.LayerNorm(j_dims[k + 1], eps=EPS))
        self.adv_head = nn.Linear(j_dims[-1], OUT)
        self._init_weights()

    def _init_weights(self):
        for m in (self.s0, self.sr1, self.s_shift, self.a0, *self.j_linears):
            nn.init.kaiming_normal_(m.weight, a=SLOPE, nonlinearity="leaky_relu")
            nn.init.zeros_(m.bias)
        for m, scale in ((self.sr2, 0.2), (self.s_gate, 0.2), (self.v_head, 0.1), (self.adv_head, 0.1)):
            nn.init.normal_(m.weight, mean=0.0, std=scale * math.sqrt(2.0 / m.weight.shape[1]))
            nn.init.zeros_(m.bias)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        h = Fn.leaky_relu(self.ln0(self.s0(s)), negative_slope=SLOPE)
        r = Fn.leaky_relu(self.ln1(h), negative_slope=SLOPE)
        r = self.sr1(r)
        r = Fn.leaky_relu(self.ln2(r), negative_slope=SLOPE)
        r = self.sr2(r)
        h = h + r
        hn = self.ln3(h)
        v = self.v_head(hn)
        shift = self.s_shift(hn)
        gate = torch.tanh(self.s_gate(hn))
        j = Fn.leaky_relu(self.a0(a) * (1.0 + gate) + shift, negative_slope=SLOPE)
        for lin, ln in zip(self.j_linears, self.j_lns):
            j = Fn.leaky_relu(ln(lin(j)), negative_slope=SLOPE)
        adv = self.adv_head(j)
        return v + adv, v


class SanDaHaNet(nn.Module):
    def __init__(self, arch, arch_type: str = "dueling_film_v2", feat_ver: int = 2):
        super().__init__()
        self.arch = arch
        self.arch_type = arch_type
        self.feat_ver = feat_ver
        dm = dims_for(feat_ver)
        if arch_type == "dueling_film_v2":
            self.nets = nn.ModuleDict({
                p: DuelingQNet(dm[p][0], dm[p][1], hidden_for(p, arch))
                for p in PHASES
            })
        else:
            self.nets = nn.ModuleDict({
                p: QNet(sum(dm[p]), hidden_for(p, arch))
                for p in PHASES
            })

    def forward(self, phase, s, a=None):
        q, _ = self.nets[phase](s, a)
        return q

    def forward_train(self, phase, s, a):
        return self.nets[phase](s, a)

    def to_numpy(self) -> dict:
        params = {}
        if self.arch_type != "dueling_film_v2":
            for p in PHASES:
                for i, l in enumerate(self.nets[p].layers):
                    params[f"{p}.W{i}"] = l.weight.detach().cpu().numpy().T.astype(np.float32).copy()
                    params[f"{p}.b{i}"] = l.bias.detach().cpu().numpy().astype(np.float32).copy()
            return params

        def put_lin(key_w, key_b, mod: nn.Linear):
            params[key_w] = mod.weight.detach().cpu().numpy().T.astype(np.float32).copy()
            params[key_b] = mod.bias.detach().cpu().numpy().astype(np.float32).copy()

        def put_ln(key_g, key_b, mod: nn.LayerNorm):
            params[key_g] = mod.weight.detach().cpu().numpy().astype(np.float32).copy()
            params[key_b] = mod.bias.detach().cpu().numpy().astype(np.float32).copy()

        for p in PHASES:
            m: DuelingQNet = self.nets[p]  # type: ignore[assignment]
            put_lin(f"{p}.Ws0", f"{p}.bs0", m.s0)
            put_ln(f"{p}.ln0_g", f"{p}.ln0_b", m.ln0)
            put_ln(f"{p}.ln1_g", f"{p}.ln1_b", m.ln1)
            put_lin(f"{p}.Wsr1", f"{p}.bsr1", m.sr1)
            put_ln(f"{p}.ln2_g", f"{p}.ln2_b", m.ln2)
            put_lin(f"{p}.Wsr2", f"{p}.bsr2", m.sr2)
            put_ln(f"{p}.ln3_g", f"{p}.ln3_b", m.ln3)
            put_lin(f"{p}.Wv", f"{p}.bv", m.v_head)
            put_lin(f"{p}.Wss", f"{p}.bss", m.s_shift)
            put_lin(f"{p}.Wsg", f"{p}.bsg", m.s_gate)
            put_lin(f"{p}.Wa0", f"{p}.ba0", m.a0)
            for k, (lin, ln) in enumerate(zip(m.j_linears, m.j_lns)):
                put_lin(f"{p}.Wjt{k}", f"{p}.bjt{k}", lin)
                put_ln(f"{p}.lnjt{k}_g", f"{p}.lnjt{k}_b", ln)
            put_lin(f"{p}.Wadv", f"{p}.badv", m.adv_head)
        return params

    def load_numpy(self, params: dict):
        with torch.no_grad():
            if self.arch_type != "dueling_film_v2":
                for p in PHASES:
                    for i, l in enumerate(self.nets[p].layers):
                        W = params.get(f"{p}.W{i}")
                        if W is None or W.T.shape != tuple(l.weight.shape):
                            raise ValueError(f"shape mismatch for {p}.W{i}")
                        l.weight.copy_(torch.from_numpy(np.ascontiguousarray(W.T)))
                        l.bias.copy_(torch.from_numpy(params[f"{p}.b{i}"]))
                return

            def load_lin(key_w, key_b, mod: nn.Linear):
                W = params[key_w]
                if W.T.shape != tuple(mod.weight.shape):
                    raise ValueError(f"shape mismatch for {key_w}: {W.T.shape} vs {tuple(mod.weight.shape)}")
                mod.weight.copy_(torch.from_numpy(np.ascontiguousarray(W.T)))
                mod.bias.copy_(torch.from_numpy(params[key_b]))

            def load_ln(key_g, key_b, mod: nn.LayerNorm):
                mod.weight.copy_(torch.from_numpy(params[key_g]))
                mod.bias.copy_(torch.from_numpy(params[key_b]))

            for p in PHASES:
                m: DuelingQNet = self.nets[p]  # type: ignore[assignment]
                load_lin(f"{p}.Ws0", f"{p}.bs0", m.s0)
                load_ln(f"{p}.ln0_g", f"{p}.ln0_b", m.ln0)
                load_ln(f"{p}.ln1_g", f"{p}.ln1_b", m.ln1)
                load_lin(f"{p}.Wsr1", f"{p}.bsr1", m.sr1)
                load_ln(f"{p}.ln2_g", f"{p}.ln2_b", m.ln2)
                load_lin(f"{p}.Wsr2", f"{p}.bsr2", m.sr2)
                load_ln(f"{p}.ln3_g", f"{p}.ln3_b", m.ln3)
                load_lin(f"{p}.Wv", f"{p}.bv", m.v_head)
                load_lin(f"{p}.Wss", f"{p}.bss", m.s_shift)
                load_lin(f"{p}.Wsg", f"{p}.bsg", m.s_gate)
                load_lin(f"{p}.Wa0", f"{p}.ba0", m.a0)
                for k, (lin, ln) in enumerate(zip(m.j_linears, m.j_lns)):
                    load_lin(f"{p}.Wjt{k}", f"{p}.bjt{k}", lin)
                    load_ln(f"{p}.lnjt{k}_g", f"{p}.lnjt{k}_b", ln)
                load_lin(f"{p}.Wadv", f"{p}.badv", m.adv_head)
