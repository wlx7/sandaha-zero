from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class TrainConfig:
    name: str = "run"
    preset: str = "medium"              # tiny / small / medium / large / xlarge / custom
    play_hidden: list = field(default_factory=lambda: [768, 512, 256])
    aux_hidden: list = field(default_factory=lambda: [256, 128])
    num_actors: int = 0                 # 0 = auto (cpu_count - 2)
    device: str = "auto"                # auto / cpu / cuda
    learner_threads: int = 0            # torch threads for a CPU learner (0 = auto)
    lr: float = 3e-4
    batch_size: int = 1024
    buffer_size: int = 300_000          # play samples kept in replay (others scaled)
    replay_ratio: float = 2.0           # times each sample is used on average
    epsilon_start: float = 0.10
    epsilon_end: float = 0.01
    epsilon_decay_games: int = 300_000
    max_games: int = 2_000_000
    max_hours: float = 0.0              # 0 = unlimited
    export_every: int = 50              # learner steps between weight pushes to actors
    eval_every_minutes: float = 10.0
    eval_deals: int = 200               # duplicate deals (x2 games) per evaluation
    baseline: str = ""                  # path to an older model (.npz) to compare against
    init_from: str = ""                 # warm start from a checkpoint (.pt or .npz)
    seed: int = 0
    # surrender curriculum: early on an untrained banker always loses, so the
    # agent would learn "always surrender" and never learn to play as banker.
    no_surrender_games: int = 300_000   # surrender disabled for the first N games
    no_surrender_frac: float = 0.3      # afterwards, fraction of games without surrender
    loss_win_weight: float = 0.3
    loss_pts_weight: float = 0.3
    rules: dict = field(default_factory=dict)

    def arch(self):
        from ..ai.model import PRESETS
        if self.preset in PRESETS:
            return {"play": list(PRESETS[self.preset]["play"]), "aux": list(PRESETS[self.preset]["aux"])}
        return {"play": list(self.play_hidden), "aux": list(self.aux_hidden)}

    def resolved_actors(self):
        if self.num_actors > 0:
            return self.num_actors
        return max(1, (os.cpu_count() or 4) - 2)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d):
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c
