"""Deep Monte-Carlo (DMC) self-play training (v2: Dueling FiLM-ResNet, EMA,
Ape-X plausible Q-noise exploration, and League self-play).

Run directory layout::

    runs/<name>/config.json     training configuration
    runs/<name>/metrics.jsonl   learner metrics (one JSON object per line)
    runs/<name>/evals.jsonl     evaluation results
    runs/<name>/latest.npz      newest live actor weights (numpy, playable)
    runs/<name>/checkpoints/    periodic EMA snapshots (ckpt_<games>.npz)
    runs/<name>/final.npz       final EMA model
    runs/<name>/status.json     running / finished / error
"""
from __future__ import annotations

import copy
import glob
import json
import math
import multiprocessing as mp
import os
import queue as queue_mod
import random
import time
import traceback

import numpy as np

PHASE_SHARE = {"play": 1.0, "bury": 0.20, "bid": 0.20, "trump": 0.08}


def _write_json(path, obj):
    from ..ai.model import replace_retry
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    replace_retry(tmp, path)


def _append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ actor --

def _game_batch(g, record, mode="self", parity=0, no_sur=False):
    """Turn one finished game's decisions into uint8 training arrays per phase."""
    res = g.result
    pts = min(res["xian_points"], 400) / 400.0
    banker = res["banker"]
    learner_is_banker = (banker % 2 == parity) if mode == "league" else None
    by_phase = {}
    for seat, phase, st, a in record:
        if mode == "league":
            if ((seat == banker) != learner_is_banker):
                continue
        sc = res["scores"][seat]
        S, A, T = by_phase.setdefault(phase, ([], [], []))
        S.append(st)
        A.append(a)
        T.append((sc / 9.0, 1.0 if sc > 0 else 0.0, pts))
    if not by_phase:
        return {}
    return {
        phase: (
            np.rint(np.clip(np.stack(S), 0.0, 1.0) * 255.0).astype(np.uint8),
            np.rint(np.clip(np.stack(A), 0.0, 1.0) * 255.0).astype(np.uint8),
            np.asarray(T, dtype=np.float32),
        )
        for phase, (S, A, T) in by_phase.items()
    }


def _merge_batches(batches):
    out = {}
    for b in batches:
        for phase, arrs in b.items():
            out.setdefault(phase, []).append(arrs)
    return {
        phase: tuple(np.concatenate([a[k] for a in lst], axis=0) for k in range(3))
        for phase, lst in out.items()
    }


def actor_main(idx, run_dir, q, cfg_dict, stop_event, games_counter):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    from ..bots import RuleBot
    from ..rules import Rules
    from .config import TrainConfig

    cfg = TrainConfig.from_dict(cfg_dict)
    rules = Rules.from_dict(cfg.rules)
    rules_ns = Rules.from_dict(dict(cfg.rules, allow_surrender=False))
    rng = random.Random(cfg.seed * 10007 + idx * 131 + 17)
    np_rng = np.random.default_rng(cfg.seed * 10007 + idx * 131 + 17)
    latest = os.path.join(run_dir, "latest.npz")
    ver_path = os.path.join(run_dir, "latest.ver")
    return _batched_actor(
        idx, run_dir, cfg, rules, rules_ns, rng, np_rng, latest, ver_path, q, stop_event, games_counter, RuleBot
    )


def _batched_actor(
    idx, run_dir, cfg, rules, rules_ns, rng, np_rng, latest, ver_path, q, stop_event, games_counter, RuleBot
):
    from ..ai import features as F
    from ..ai.model import Model
    from ..engine import BID, Game

    model, ver = None, None
    K = max(1, cfg.games_per_actor)
    n_actors = cfg.resolved_actors()
    greedy_cut = max(1, n_actors // 10)
    if idx < greedy_cut or cfg.q_noise_max <= 0:
        base_sigma = 0.0
    else:
        denom = max(1, n_actors - 1 - greedy_cut)
        frac_i = (idx - greedy_cut) / denom
        lo = max(1e-5, cfg.q_noise_min)
        hi = max(lo, cfg.q_noise_max)
        base_sigma = lo * ((hi / lo) ** frac_i)

    # Optional teacher & baseline models
    teacher = None
    if cfg.teacher_model and os.path.exists(cfg.teacher_model):
        try:
            teacher = Model.load(cfg.teacher_model)
        except Exception:
            teacher = None

    baseline_opp = None
    b_path = cfg.baseline or cfg.teacher_model
    if b_path and os.path.exists(b_path):
        try:
            baseline_opp = Model.load(b_path)
        except Exception:
            baseline_opp = None

    rule_bot = RuleBot(seed=cfg.seed + idx + 1)
    ckpt_pool: list[tuple[str, Model]] = []
    last_pool_scan = 0.0

    def refresh_pool():
        nonlocal last_pool_scan, ckpt_pool
        now = time.time()
        if now - last_pool_scan < 45.0:
            return
        last_pool_scan = now
        ck_dir = os.path.join(run_dir, "checkpoints")
        paths = sorted(glob.glob(os.path.join(ck_dir, "*.npz")))
        if not paths:
            return
        # Keep up to 6 spread-out checkpoints (biasing toward more recent ones)
        selected = paths[-4:]
        if len(paths) > 6:
            selected = [paths[len(paths) // 3], paths[(2 * len(paths)) // 3]] + selected
        known = dict(ckpt_pool)
        new_pool = []
        for p in selected:
            if p in known:
                new_pool.append((p, known[p]))
            else:
                try:
                    new_pool.append((p, Model.load(p)))
                except Exception:
                    pass
        ckpt_pool = new_pool

    def pick_league_opp():
        refresh_pool()
        r = rng.random()
        if baseline_opp is not None and r < 0.40:
            return ("model", baseline_opp)
        if ckpt_pool and r < 0.85:
            # Prefer newer checkpoints slightly
            weights = [i + 1 for i in range(len(ckpt_pool))]
            chosen = rng.choices(ckpt_pool, weights=weights, k=1)[0][1]
            return ("model", chosen)
        if baseline_opp is not None and r < 0.88:
            return ("model", baseline_opp)
        return ("rule", rule_bot)

    def new_slot():
        n = games_counter.value
        local_n = max(0, n - init_games_offset)
        no_sur = local_n < cfg.no_surrender_games or rng.random() < cfg.no_surrender_frac
        g = Game(rules_ns if no_sur else rules, seed=rng.randrange(1 << 30), first_bidder=rng.randrange(4))
        # Decide game mode: teacher warmup / league / pure self-play
        if teacher is not None and cfg.teacher_games > 0 and local_n < cfg.teacher_games:
            t_prob = cfg.teacher_frac * (1.0 - local_n / cfg.teacher_games)
            if rng.random() < t_prob:
                return {"g": g, "rec": [], "mode": "teacher", "opp": ("model", teacher), "parity": 0, "no_sur": False}
        if cfg.league_frac > 0 and local_n >= cfg.league_start_games and rng.random() < cfg.league_frac:
            return {"g": g, "rec": [], "mode": "league", "opp": pick_league_opp(), "parity": rng.randrange(2), "no_sur": (no_sur and local_n >= cfg.no_surrender_games)}
        return {"g": g, "rec": [], "mode": "self", "opp": None, "parity": 0, "no_sur": (no_sur and local_n >= cfg.no_surrender_games)}

    def epsilon_and_sigma():
        local_n = max(0, games_counter.value - init_games_offset)
        frac = min(1.0, local_n / max(1, cfg.epsilon_decay_games))
        eps = cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * frac
        if base_sigma <= 0.0:
            return (0.0 if frac >= 1.0 else eps * 0.25), 0.0
        return eps, base_sigma

    slots = []
    pending = []
    init_games_offset = games_counter.value
    target_fv = cfg.feat_ver
    try:
        while not stop_event.is_set():
            try:
                with open(ver_path) as f:
                    v = f.read().strip()
            except OSError:
                v = None
            if v is not None and v != ver:
                try:
                    model = Model.load(latest)
                    ver = v
                except Exception:
                    time.sleep(0.15)
                    continue
            if model is None:
                time.sleep(0.3)
                continue

            while len(slots) < K:
                slots.append(new_slot())

            eps, sigma = epsilon_and_sigma()
            latest_groups = {}
            opp_model_groups = {}

            for i, sl in enumerate(slots):
                g = sl["g"]
                s = g.to_act
                mode = sl["mode"]
                if mode == "self":
                    acts = g.legal_actions()
                    ph, st, A = F.encode(g, s, acts, target_fv)
                    latest_groups.setdefault(ph, []).append((i, s, acts, st, A))
                elif mode == "teacher":
                    t_mod = sl["opp"][1]
                    acts = g.legal_actions()
                    ph, st_t, A_t = F.encode(g, s, acts, t_mod.feat_ver)
                    if t_mod.feat_ver == target_fv:
                        st_rec, A_rec = st_t, A_t
                    else:
                        _, st_rec, A_rec = F.encode(g, s, acts, target_fv)
                    opp_model_groups.setdefault((id(t_mod), t_mod, ph), []).append(
                        (i, s, acts, st_t, A_t, True, st_rec, A_rec)
                    )
                else:  # league
                    parity = sl["parity"]
                    if g.phase == BID:
                        use_latest = (s % 2 == parity)
                    else:
                        learner_is_bk = (g.banker % 2 == parity)
                        use_latest = ((s == g.banker) == learner_is_bk)
                    if use_latest:
                        acts = g.legal_actions()
                        ph, st, A = F.encode(g, s, acts, target_fv)
                        latest_groups.setdefault(ph, []).append((i, s, acts, st, A))
                    else:
                        kind, opp_obj = sl["opp"]
                        if kind == "rule":
                            g.step(opp_obj.act(g))
                        else:
                            acts = g.legal_actions()
                            if len(acts) == 1:
                                g.step(acts[0])
                            else:
                                ph, st_o, A_o = F.encode(g, s, acts, opp_obj.feat_ver)
                                opp_model_groups.setdefault((id(opp_obj), opp_obj, ph), []).append(
                                    (i, s, acts, st_o, A_o, False, None, None)
                                )

            # Batch-evaluate learner decisions
            for ph, items in latest_groups.items():
                outs = model.q_batch(ph, [it[3] for it in items], [it[4] for it in items])
                for (i, s, acts, st, A), out in zip(items, outs):
                    m_acts = len(acts)
                    if m_acts == 1:
                        j = 0
                    elif eps > 0.0 and rng.random() < eps:
                        j = rng.randrange(m_acts)
                    elif sigma > 0.0:
                        noisy_q = out[:, 0] + np_rng.normal(0.0, sigma, size=m_acts)
                        j = int(np.argmax(noisy_q))
                    else:
                        j = int(np.argmax(out[:, 0]))
                    slots[i]["rec"].append((s, ph, st, A[j]))
                    slots[i]["g"].step(acts[j])

            # Batch-evaluate opponent / teacher model decisions
            for (_, opp_mod, ph), items in opp_model_groups.items():
                outs = opp_mod.q_batch(ph, [it[3] for it in items], [it[4] for it in items])
                for (i, s, acts, _, _, rec_flag, st_rec, A_rec), out in zip(items, outs):
                    m_acts = len(acts)
                    if m_acts > 1 and rec_flag and rng.random() < 0.02:
                        j = rng.randrange(m_acts)
                    else:
                        j = int(np.argmax(out[:, 0]))
                    if rec_flag:
                        slots[i]["rec"].append((s, ph, st_rec, A_rec[j]))
                    slots[i]["g"].step(acts[j])

            # Harvest finished games
            done = 0
            for i in range(len(slots) - 1, -1, -1):
                if slots[i]["g"].over:
                    sl = slots[i]
                    gb = _game_batch(sl["g"], sl["rec"], mode=sl["mode"], parity=sl["parity"], no_sur=sl["no_sur"])
                    if gb:
                        pending.append(gb)
                    del slots[i]
                    done += 1
            if done:
                with games_counter.get_lock():
                    games_counter.value += done
            if len(pending) >= cfg.send_games:
                merged = _merge_batches(pending)
                while not stop_event.is_set():
                    try:
                        q.put(merged, timeout=0.5)
                        break
                    except queue_mod.Full:
                        continue
                pending = []
    except KeyboardInterrupt:
        pass


# ------------------------------------------------------------------- eval --

def eval_main(run_dir, ckpt, games, others, deals, rules_dict, seed):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    from ..ai.agent import AIAgent
    from ..ai.model import Model
    from ..arena import duel
    from ..bots import RuleBot
    from ..rules import Rules

    rules = Rules.from_dict(rules_dict)
    agent = AIAgent(Model.load(ckpt))
    out = {"type": "eval", "time": time.time(), "games": games, "ckpt": os.path.basename(ckpt)}
    out["vs_rule"] = duel(agent, RuleBot(seed=1), deals, rules, seed=seed)
    for key, path in others:
        if path and os.path.exists(path):
            out[key] = duel(agent, AIAgent(Model.load(path)), deals, rules, seed=seed)
            out[key]["opponent"] = os.path.basename(path)
    _append_jsonl(os.path.join(run_dir, "evals.jsonl"), out)


# ---------------------------------------------------------------- learner --

class Replay:
    def __init__(self, cap, s_dim, a_dim):
        self.S = np.zeros((cap, s_dim), np.uint8)
        self.A = np.zeros((cap, a_dim), np.uint8)
        self.T = np.zeros((cap, 3), np.float32)
        self.cap, self.ptr, self.size, self.added = cap, 0, 0, 0

    def add(self, S, A, T):
        n = len(S)
        if n == 0:
            return
        if n >= self.cap:
            S, A, T, n = S[-self.cap:], A[-self.cap:], T[-self.cap:], self.cap
        end = self.ptr + n
        if end <= self.cap:
            self.S[self.ptr:end], self.A[self.ptr:end], self.T[self.ptr:end] = S, A, T
        else:
            k = self.cap - self.ptr
            self.S[self.ptr:], self.A[self.ptr:], self.T[self.ptr:] = S[:k], A[:k], T[:k]
            self.S[:n - k], self.A[:n - k], self.T[:n - k] = S[k:], A[k:], T[k:]
        self.ptr = end % self.cap
        self.size = min(self.cap, self.size + n)
        self.added += n

    def sample(self, n, rng):
        idx = rng.integers(0, self.size, n)
        return self.S[idx], self.A[idx], self.T[idx]


def train(cfg, run_dir, log=print):
    import torch
    import torch.nn.functional as Fn

    from ..ai.features import PHASES, dims_for
    from ..ai.model import Model, replace_retry, save_params
    from .torch_model import SanDaHaNet

    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    status_path = os.path.join(run_dir, "status.json")
    stop_path = os.path.join(run_dir, "STOP")
    if os.path.exists(stop_path):
        os.remove(stop_path)
    t_start = time.time()
    status = {"state": "starting", "pid": os.getpid(), "start": t_start, "name": cfg.name}
    _write_json(status_path, status)

    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device
    if device == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.manual_seed(cfg.seed)
    if device == "cpu":
        torch.set_num_threads(cfg.learner_threads or max(2, min(8, (os.cpu_count() or 8) // 4)))

    arch = cfg.arch()
    init_model = None
    if cfg.init_from and cfg.init_from.endswith(".npz"):
        init_model = Model.load(cfg.init_from)
        if init_model.meta.get("arch"):
            arch = init_model.meta["arch"]
        cfg.arch_type = init_model.arch_type
        cfg.feat_ver = init_model.feat_ver
    elif cfg.init_from:
        sd_meta = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        arch = sd_meta.get("arch", arch)
        cfg.arch_type = sd_meta.get("arch_type", "mlp_v1")
        cfg.feat_ver = int(sd_meta.get("feat_ver", 2 if cfg.arch_type == "dueling_film_v2" else 1))

    cfg.save(os.path.join(run_dir, "config.json"))
    dm = dims_for(cfg.feat_ver)
    net = SanDaHaNet(arch, arch_type=cfg.arch_type, feat_ver=cfg.feat_ver).to(device)
    games_offset = 0
    saved_opts = None
    saved_ema = None
    if cfg.init_from:
        if init_model is not None:
            net.load_numpy(init_model.params)
            games_offset = int(init_model.meta.get("games", 0))
        else:
            sd = torch.load(cfg.init_from, map_location=device, weights_only=False)
            net.load_state_dict(sd["net"])
            saved_ema = sd.get("ema")
            saved_opts = sd.get("opts")
            games_offset = int(sd.get("games", 0))
        log(f"warm start from {cfg.init_from} ({games_offset} games, arch_type={cfg.arch_type}, feat_ver={cfg.feat_ver})")

    ema_net = copy.deepcopy(net).eval()
    if saved_ema is not None:
        ema_net.load_state_dict(saved_ema)
    for p_param in ema_net.parameters():
        p_param.requires_grad_(False)

    phase_params = {p: list(net.nets[p].parameters()) for p in PHASES}
    ema_phase_params = {p: list(ema_net.nets[p].parameters()) for p in PHASES}
    opts = {}
    for p in PHASES:
        dec = [w for w in phase_params[p] if w.ndim >= 2]
        nodec = [w for w in phase_params[p] if w.ndim < 2]
        opts[p] = torch.optim.AdamW(
            [
                {"params": dec, "weight_decay": cfg.weight_decay},
                {"params": nodec, "weight_decay": 0.0},
            ],
            lr=cfg.lr,
        )
        if saved_opts and p in saved_opts:
            try:
                opts[p].load_state_dict(saved_opts[p])
            except Exception:
                pass

    use_bf16 = (device == "cuda") and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    phase_batch_warm = {
        "play": cfg.batch_size,
        "bury": min(cfg.batch_size, 512),
        "bid": min(cfg.batch_size, 512),
        "trump": min(cfg.batch_size, 256),
    }
    phase_batch_steady = {
        "play": cfg.batch_size,
        "bury": max(1024, cfg.batch_size // 2),
        "bid": max(1024, cfg.batch_size // 2),
        "trump": max(512, cfg.batch_size // 4),
    }
    phase_replay_warm = {
        "play": cfg.replay_ratio,
        "bury": cfg.replay_ratio * 2.0,
        "bid": cfg.replay_ratio * 2.0,
        "trump": cfg.replay_ratio * 3.0,
    }
    phase_replay_steady = {
        "play": cfg.replay_ratio,
        "bury": cfg.replay_ratio * 2.2,
        "bid": cfg.replay_ratio * 2.2,
        "trump": cfg.replay_ratio * 2.0,
    }

    def base_lr_at(now, games):
        if cfg.lr_schedule == "constant":
            return cfg.lr
        if cfg.max_hours:
            frac = (now - t_start) / (cfg.max_hours * 3600.0)
        elif cfg.max_games:
            frac = (games - games_offset) / cfg.max_games
        else:
            return cfg.lr
        frac = min(1.0, max(0.0, frac))
        if cfg.lr_schedule == "linear":
            w = 1.0 - frac
        else:
            w = 0.5 * (1.0 + math.cos(math.pi * frac))
        return cfg.lr_end + (cfg.lr - cfg.lr_end) * w

    meta = {
        "name": cfg.name,
        "arch": arch,
        "arch_type": cfg.arch_type,
        "feat_ver": cfg.feat_ver,
        "rules": cfg.rules,
        "version": 2 if cfg.arch_type == "dueling_film_v2" else 1,
    }

    def export_net(which_net, path, games):
        m = dict(meta, games=games, time=time.time())
        save_params(path, which_net.to_numpy(), m)

    latest = os.path.join(run_dir, "latest.npz")
    ver_path = os.path.join(run_dir, "latest.ver")
    version = 0
    last_push_time = 0.0

    def push(games, force=False):
        nonlocal version, last_push_time
        now_p = time.time()
        if not force and (now_p - last_push_time) < 4.0:
            return
        export_net(net, latest, games)
        version += 1
        last_push_time = now_p
        with open(ver_path + ".tmp", "w") as f:
            f.write(str(version))
        replace_retry(ver_path + ".tmp", ver_path)

    push(games_offset, force=True)

    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=384)
    stop_event = ctx.Event()
    games_counter = ctx.Value("q", games_offset)
    n_actors = cfg.resolved_actors()
    cfg_dict = json.loads(json.dumps(cfg.__dict__))
    actors = [
        ctx.Process(target=actor_main, args=(i, run_dir, q, cfg_dict, stop_event, games_counter), daemon=True)
        for i in range(n_actors)
    ]
    for p in actors:
        p.start()
    log(f"device={device} actors={n_actors} arch={arch} arch_type={cfg.arch_type} feat_ver={cfg.feat_ver}")

    bufs = {}
    for p in PHASES:
        cap = max(30_000, int(cfg.buffer_size * PHASE_SHARE[p]))
        bufs[p] = Replay(cap, *dm[p])
    # Track virtual sample consumption so changing batch_size after warmup has no step jump
    consumed = {p: 0.0 for p in PHASES}
    steps = {p: 0 for p in PHASES}
    loss_acc = {p: [] for p in PHASES}
    np_rng = np.random.default_rng(cfg.seed)
    total_steps = 0
    last_metric = time.time()
    last_games = games_offset
    last_eval = time.time()
    eval_proc = None
    prev_ckpt = None
    if games_offset > 0:
        min_fill = {
            "play": max(24_000, cfg.batch_size * 6),
            "bury": max(6_000, cfg.batch_size * 2),
            "bid": max(5_000, cfg.batch_size * 2),
            "trump": max(2_500, cfg.batch_size),
        }
    else:
        min_fill = {"play": 4_000, "bury": 1_200, "bid": 1_000, "trump": 400}
    status["state"] = "running"
    _write_json(status_path, status)

    def do_eval(games, wait=False):
        nonlocal eval_proc, prev_ckpt
        ck = os.path.join(run_dir, "checkpoints", f"ckpt_{games:09d}.npz")
        use_ema = ema_net if (cfg.ema_decay > 0 and min(steps.values()) >= 20) else net
        export_net(use_ema, ck, games)
        torch.save(
            {
                "net": net.state_dict(),
                "ema": ema_net.state_dict(),
                "opts": {k: o.state_dict() for k, o in opts.items()},
                "games": games,
                "arch": arch,
                "arch_type": cfg.arch_type,
                "feat_ver": cfg.feat_ver,
            },
            os.path.join(run_dir, "latest.pt"),
        )
        if eval_proc is not None and eval_proc.is_alive():
            if not wait:
                return
            eval_proc.join()
        others = [("vs_baseline", cfg.baseline), ("vs_prev", prev_ckpt)]
        eval_proc = ctx.Process(
            target=eval_main,
            args=(run_dir, ck, games, others, cfg.eval_deals, cfg.rules, 12345),
            daemon=False,
        )
        eval_proc.start()
        prev_ckpt = ck
        if wait:
            eval_proc.join()

    reason = "finished"
    t_ingest = t_learn = 0.0
    inv255 = 1.0 / 255.0
    ema_alpha = 1.0 - cfg.ema_decay
    try:
        while True:
            # ---- ingest
            t0 = time.time()
            got = 0
            while got < 96:
                try:
                    batch = q.get(timeout=0.04 if got == 0 else 0)
                except queue_mod.Empty:
                    break
                for p, (S, A, T) in batch.items():
                    bufs[p].add(S, A, T)
                got += 1
            t1 = time.time()
            t_ingest += t1 - t0

            # ---- learn
            cur_lr = base_lr_at(t1, games_counter.value)

            for p in PHASES:
                b = bufs[p]
                if b.size < min_fill[p]:
                    continue
                steady = (games_offset > 0) or (steps[p] >= 400)
                bsz = (phase_batch_steady if steady else phase_batch_warm)[p]
                rep = (phase_replay_steady if steady else phase_replay_warm)[p]
                avail = b.added * rep - consumed[p]
                budget = int(avail // bsz)
                p_opt = opts[p]
                eff_warmup = (
                    max(300, cfg.warmup_steps)
                    if saved_opts is None
                    else max(150, cfg.warmup_steps // 2)
                )
                for _ in range(min(budget, 48)):
                    w_mult = (
                        max(0.05, (steps[p] + 1) / eff_warmup)
                        if (cfg.warmup_steps > 0 and steps[p] < eff_warmup)
                        else 1.0
                    )
                    step_lr = cur_lr * w_mult
                    for grp in p_opt.param_groups:
                        grp["lr"] = step_lr

                    S, A, T = b.sample(bsz, np_rng)
                    s_t = torch.from_numpy(S).to(device, non_blocking=True).to(torch.float32).mul_(inv255)
                    a_t = torch.from_numpy(A).to(device, non_blocking=True).to(torch.float32).mul_(inv255)
                    t_t = torch.from_numpy(T).to(device, non_blocking=True)

                    if use_bf16:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            q_out, v_out = net.forward_train(p, s_t, a_t)
                        q_out = q_out.float()
                        v_out = v_out.float() if v_out is not None else None
                    else:
                        q_out, v_out = net.forward_train(p, s_t, a_t)
                    l_q = Fn.mse_loss(q_out[:, 0], t_t[:, 0])
                    l_w = Fn.binary_cross_entropy_with_logits(q_out[:, 1], t_t[:, 1])
                    l_p = Fn.mse_loss(q_out[:, 2], t_t[:, 2])
                    loss = l_q + cfg.loss_win_weight * l_w + cfg.loss_pts_weight * l_p

                    if v_out is not None and cfg.loss_value_weight > 0:
                        lv_q = Fn.mse_loss(v_out[:, 0], t_t[:, 0])
                        lv_w = Fn.binary_cross_entropy_with_logits(v_out[:, 1], t_t[:, 1])
                        lv_p = Fn.mse_loss(v_out[:, 2], t_t[:, 2])
                        loss = loss + cfg.loss_value_weight * (
                            lv_q + cfg.loss_win_weight * lv_w + cfg.loss_pts_weight * lv_p
                        )

                    p_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(phase_params[p], 10.0)
                    p_opt.step()
                    consumed[p] += bsz

                    if cfg.ema_decay > 0:
                        with torch.no_grad():
                            eff_alpha = (
                                ema_alpha
                                if games_offset > 0
                                else max(ema_alpha, 1.0 / (steps[p] + 10))
                            )
                            torch._foreach_lerp_(ema_phase_params[p], phase_params[p], eff_alpha)

                    steps[p] += 1
                    total_steps += 1
                    loss_acc[p].append(torch.stack([l_q.detach(), l_w.detach(), l_p.detach()]))
                    if total_steps % cfg.export_every == 0:
                        push(games_counter.value)

            # ---- bookkeeping
            now = time.time()
            t_learn += now - t1
            games = games_counter.value
            if now - last_metric >= 10:
                local_n = max(0, games - games_offset)
                rec = {
                    "type": "train",
                    "time": now,
                    "elapsed": now - t_start,
                    "games": games,
                    "games_per_s": (games - last_games) / (now - last_metric),
                    "steps": total_steps,
                    "version": version,
                    "lr": cur_lr,
                    "epsilon": cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) *
                    min(1.0, local_n / max(1, cfg.epsilon_decay_games)),
                }
                for p in PHASES:
                    if loss_acc[p]:
                        arr = torch.stack(loss_acc[p]).cpu().numpy()
                        rec[f"loss_{p}"] = float(arr[:, 0].mean())
                        rec[f"winloss_{p}"] = float(arr[:, 1].mean())
                    rec[f"samples_{p}"] = bufs[p].added
                    loss_acc[p] = []
                rec["t_ingest"], rec["t_learn"] = t_ingest, t_learn
                t_ingest = t_learn = 0.0
                _append_jsonl(os.path.join(run_dir, "metrics.jsonl"), rec)
                log(
                    f"games={games} g/s={rec['games_per_s']:.1f} steps={total_steps} lr={cur_lr:.2e} "
                    f"loss_play={rec.get('loss_play', float('nan')):.4f} "
                    f"loss_bury={rec.get('loss_bury', float('nan')):.4f} "
                    f"loss_bid={rec.get('loss_bid', float('nan')):.4f}"
                )
                last_metric, last_games = now, games
            if now - last_eval >= cfg.eval_every_minutes * 60:
                do_eval(games)
                last_eval = now
            if os.path.exists(stop_path):
                reason = "stopped"
                break
            if cfg.max_games and games - games_offset >= cfg.max_games:
                break
            if cfg.max_hours and now - t_start >= cfg.max_hours * 3600:
                break
            if not any(p.is_alive() for p in actors):
                reason = "actors died"
                break
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception:
        reason = "error"
        status["error"] = traceback.format_exc()
        log(status["error"])
    finally:
        stop_event.set()
        games = games_counter.value
        try:
            use_ema = ema_net if (cfg.ema_decay > 0 and min(steps.values()) >= 20) else net
            export_net(use_ema, os.path.join(run_dir, "final.npz"), games)
            status["state"] = "evaluating"
            _write_json(status_path, status)
            do_eval(games, wait=True)
        except Exception:
            status["error"] = traceback.format_exc()
        for p in actors:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        status.update({
            "state": "error" if reason == "error" else "finished",
            "reason": reason,
            "end": time.time(),
            "games": games,
        })
        _write_json(status_path, status)
        log(f"training {reason}: {games} games")
    return run_dir
