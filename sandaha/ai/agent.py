"""Neural agent: picks the action with the highest predicted score."""
from __future__ import annotations

import random

import numpy as np

from ..engine import describe_action
from . import features as F
from .model import Model


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class AIAgent:
    def __init__(self, model: Model, epsilon: float = 0.0, seed=None, name=None):
        self.model = model
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.name = name or model.name
        self.record = None  # set to a list to capture (phase, state, action) samples

    def _scores(self, g):
        s = g.to_act
        acts = g.legal_actions()
        phase, st, A = F.encode(g, s, acts, self.model.feat_ver)
        out = self.model.q(phase, st, A)
        return acts, phase, st, A, out

    def act(self, g):
        acts, phase, st, A, out = self._scores(g)
        if len(acts) > 1 and self.epsilon > 0 and self.rng.random() < self.epsilon:
            i = self.rng.randrange(len(acts))
        else:
            i = int(np.argmax(out[:, 0]))
        if self.record is not None:
            self.record.append((g.to_act, phase, st, A[i]))
        return acts[i]

    def evaluate(self, g, top=None):
        """Ranked list of candidate actions with the network's predictions."""
        acts, phase, st, A, out = self._scores(g)
        rows = []
        for i, a in enumerate(acts):
            rows.append({
                "action": a,
                "label": describe_action(phase, a),
                "exp_score": float(out[i, 0] * 9),
                "win_prob": float(_sigmoid(out[i, 1])),
                "xian_points": float(out[i, 2] * 400),
            })
        rows.sort(key=lambda r: -r["exp_score"])
        return rows[:top] if top else rows
