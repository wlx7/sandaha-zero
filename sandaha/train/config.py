from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class TrainConfig:
    name: str = "run"
    preset: str = "medium"              # tiny / small / medium / large / xlarge / custom
    arch_type: str = "dueling_film_v2"  # dueling_film_v2 / mlp_v1
    feat_ver: int = 2                   # 2 (tactical + pair-void + margins) / 1 (legacy)
    play_hidden: list = field(default_factory=lambda: [768, 512, 256])
    aux_hidden: list = field(default_factory=lambda: [384, 256])
    num_actors: int = 0                 # 0 = auto (cpu_count - 2)
    device: str = "auto"                # auto / cpu / cuda
    learner_threads: int = 0            # torch threads for a CPU learner (0 = auto)
    games_per_actor: int = 16           # hands each worker plays side by side (batched inference)
    send_games: int = 32                # finished hands bundled into one message to the learner
    lr: float = 3e-4
    lr_end: float = 1.5e-5              # final learning rate of the schedule
    lr_schedule: str = "cosine"         # constant / linear / cosine (over max_hours, else max_games)
    warmup_steps: int = 300             # linear LR warmup steps (per phase)
    weight_decay: float = 1e-5          # AdamW weight decay on 2D weights
    ema_decay: float = 0.9995           # Polyak EMA decay for checkpoint / final export
    batch_size: int = 2048
    buffer_size: int = 1_000_000        # play samples kept in replay (others scaled)
    replay_ratio: float = 1.5           # times each sample is used on average
    epsilon_start: float = 0.10
    epsilon_end: float = 0.003
    epsilon_decay_games: int = 350_000
    q_noise_min: float = 0.002          # Ape-X plausible Q-perturbation min std (in score/9)
    q_noise_max: float = 0.032          # Ape-X plausible Q-perturbation max std (in score/9)
    league_frac: float = 0.20           # fraction of games played against league pool (historical/baseline/rule)
    league_start_games: int = 200_000   # start league matches after initial warmup
    teacher_model: str = ""             # optional model (.npz)
    teacher_games: int = 0              # 0 = pure on-policy warmup (recommended for Q-learning)
    teacher_frac: float = 0.0           # initial fraction of games stepped by teacher_model
    max_games: int = 0                  # 0 = unlimited (stop by time or manually)
    max_hours: float = 3.0              # 0 = unlimited
    export_every: int = 50              # learner steps between weight pushes to actors
    eval_every_minutes: float = 10.0
    eval_deals: int = 300               # duplicate deals (x2 games) per evaluation
    baseline: str = ""                  # path to an older model (.npz) to compare against
    init_from: str = ""                 # warm start from a checkpoint (.pt or .npz)
    seed: int = 0
    # surrender curriculum: early on an untrained banker always loses, so the
    # agent would learn "always surrender" and never learn to play as banker.
    no_surrender_games: int = 300_000   # surrender disabled for the first N games
    no_surrender_frac: float = 0.12     # afterwards, fraction of games without surrender
    loss_win_weight: float = 0.3
    loss_pts_weight: float = 0.3
    loss_value_weight: float = 0.5      # weight of unbiased state-value V(s) loss in Dueling net
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
            d = json.load(f)
        # Preserve v1 defaults when loading legacy config.json without arch_type
        if "arch_type" not in d:
            d = dict(d, arch_type="mlp_v1", feat_ver=1)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d):
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c
