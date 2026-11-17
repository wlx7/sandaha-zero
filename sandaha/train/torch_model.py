from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..ai.features import DIMS, PHASES
from ..ai.model import OUT, hidden_for


class QNet(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        dims = [in_dim] + list(hidden) + [OUT]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class SanDaHaNet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.nets = nn.ModuleDict({p: QNet(sum(DIMS[p]), hidden_for(p, arch)) for p in PHASES})

    def forward(self, phase, x):
        return self.nets[phase](x)

    def to_numpy(self) -> dict:
        params = {}
        for p in PHASES:
            for i, l in enumerate(self.nets[p].layers):
                params[f"{p}.W{i}"] = l.weight.detach().cpu().numpy().T.astype(np.float32).copy()
                params[f"{p}.b{i}"] = l.bias.detach().cpu().numpy().astype(np.float32).copy()
        return params

    def load_numpy(self, params: dict):
        with torch.no_grad():
            for p in PHASES:
                for i, l in enumerate(self.nets[p].layers):
                    W = params.get(f"{p}.W{i}")
                    if W is None or W.T.shape != tuple(l.weight.shape):
                        raise ValueError(f"shape mismatch for {p}.W{i}")
                    l.weight.copy_(torch.from_numpy(np.ascontiguousarray(W.T)))
                    l.bias.copy_(torch.from_numpy(params[f"{p}.b{i}"]))
