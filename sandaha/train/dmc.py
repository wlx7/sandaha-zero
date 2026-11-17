"""Deep Monte-Carlo (DMC) self-play training, in the spirit of DouZero.

* N actor processes play complete hands with an epsilon-greedy copy of the
  current network (pure numpy inference, one CPU thread each).
* Every decision (bid / trump / bury / play) becomes a sample whose target is
  the final result of the hand for the acting seat: [score / 9, won, 闲 points / 400].
* The learner (PyTorch, GPU if available) regresses Q(s, a) on those targets and
  periodically pushes fresh weights to the actors through ``latest.npz``.

Run directory layout::

    runs/<name>/config.json     training configuration
    runs/<name>/metrics.jsonl   learner metrics (one JSON object per line)
    runs/<name>/evals.jsonl     evaluation results
    runs/<name>/latest.npz      newest weights (numpy, playable)
    runs/<name>/checkpoints/    periodic snapshots  (ckpt_<games>.npz)
    runs/<name>/final.npz       final model
    runs/<name>/status.json     running / finished / error
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import queue as queue_mod
import random
import sys
import time
import traceback

import numpy as np

PHASE_SHARE = {"play": 1.0, "bury": 0.15, "bid": 0.15, "trump": 0.05}


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

def actor_main(idx, run_dir, q, cfg_dict, stop_event, games_counter):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    from ..ai.agent import AIAgent
    from ..ai.model import Model
    from ..engine import Game
    from ..rules import Rules
    from .config import TrainConfig

    cfg = TrainConfig.from_dict(cfg_dict)
    rules = Rules.from_dict(cfg.rules)
    rules_ns = Rules.from_dict(dict(cfg.rules, allow_surrender=False))
    rng = random.Random(cfg.seed * 1000 + idx)
    latest = os.path.join(run_dir, "latest.npz")
    ver_path = os.path.join(run_dir, "latest.ver")
    if cfg.games_per_actor > 1:
        return _batched_actor(cfg, rules, rules_ns, rng, latest, ver_path, q, stop_event, games_counter)
    model, ver = None, None
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
                    time.sleep(0.2)
                    continue
            if model is None:
                time.sleep(0.5)
                continue
            n = games_counter.value
            frac = min(1.0, n / max(1, cfg.epsilon_decay_games))
            eps = cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * frac
            agent = AIAgent(model, epsilon=eps, seed=rng.randrange(1 << 30))
            agent.record = []
            no_sur = n < cfg.no_surrender_games or rng.random() < cfg.no_surrender_frac
            g = Game(rules_ns if no_sur else rules, seed=rng.randrange(1 << 30),
                     first_bidder=rng.randrange(4))
            while not g.over:
                g.step(agent.act(g))
            res = g.result
            pts = min(res["xian_points"], 400) / 400
            by_phase = {}
            for seat, phase, st, a in agent.record:
                sc = res["scores"][seat]
                by_phase.setdefault(phase, ([], [], []))
                S, A, T = by_phase[phase]
                S.append(st)
                A.append(a)
                T.append((sc / 9.0, 1.0 if sc > 0 else 0.0, pts))
            batch = {}
            for phase, (S, A, T) in by_phase.items():
                batch[phase] = (np.rint(np.stack(S) * 255).astype(np.uint8),
                                np.rint(np.stack(A) * 255).astype(np.uint8),
                                np.asarray(T, np.float32))
            with games_counter.get_lock():
                games_counter.value += 1
            while not stop_event.is_set():
                try:
                    q.put(batch, timeout=0.5)
                    break
                except queue_mod.Full:
                    continue
    except KeyboardInterrupt:
        pass


def _game_batch(g, record):
    """Turn one finished game's decisions into training arrays per phase."""
    res = g.result
    pts = min(res["xian_points"], 400) / 400
    by_phase = {}
    for seat, phase, st, a in record:
        sc = res["scores"][seat]
        S, A, T = by_phase.setdefault(phase, ([], [], []))
        S.append(st)
        A.append(a)
        T.append((sc / 9.0, 1.0 if sc > 0 else 0.0, pts))
    return {phase: (np.rint(np.stack(S) * 255).astype(np.uint8),
                    np.rint(np.stack(A) * 255).astype(np.uint8),
                    np.asarray(T, np.float32))
            for phase, (S, A, T) in by_phase.items()}


def _merge_batches(batches):
    out = {}
    for b in batches:
        for phase, arrs in b.items():
            out.setdefault(phase, []).append(arrs)
    return {phase: tuple(np.concatenate([a[k] for a in lst]) for k in range(3))
            for phase, lst in out.items()}


def _batched_actor(cfg, rules, rules_ns, rng, latest, ver_path, q, stop_event, games_counter):
    """Play ``games_per_actor`` hands side by side and evaluate all pending
    decisions of the same phase in one network pass.  Every forward pass then
    streams the weights from memory once for many decisions instead of once
    per decision, which is what limits throughput on many-core machines."""
    from ..ai import features as F
    from ..ai.model import Model
    from ..engine import Game

    model, ver = None, None
    K = cfg.games_per_actor

    def new_game():
        n = games_counter.value
        no_sur = n < cfg.no_surrender_games or rng.random() < cfg.no_surrender_frac
        return Game(rules_ns if no_sur else rules, seed=rng.randrange(1 << 30),
                    first_bidder=rng.randrange(4))

    def epsilon():
        frac = min(1.0, games_counter.value / max(1, cfg.epsilon_decay_games))
        return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * frac

    games = []
    records = []
    pending = []
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
                    time.sleep(0.2)
                    continue
            if model is None:
                time.sleep(0.5)
                continue
            while len(games) < K:
                games.append(new_game())
                records.append([])
            eps = epsilon()
            # one decision for every live game, grouped by phase
            groups = {}
            for i, g in enumerate(games):
                s = g.to_act
                acts = g.legal_actions()
                phase, st, A = F.encode(g, s, acts)
                groups.setdefault(phase, []).append((i, s, acts, st, A))
            for phase, items in groups.items():
                outs = model.q_batch(phase, [it[3] for it in items], [it[4] for it in items])
                for (i, s, acts, st, A), out in zip(items, outs):
                    if len(acts) > 1 and rng.random() < eps:
                        j = rng.randrange(len(acts))
                    else:
                        j = int(np.argmax(out[:, 0]))
                    records[i].append((s, phase, st, A[j]))
                    games[i].step(acts[j])
            # harvest finished games
            done = 0
            for i in range(len(games) - 1, -1, -1):
                if games[i].over:
                    pending.append(_game_batch(games[i], records[i]))
                    del games[i], records[i]
                    done += 1
            if done:
                with games_counter.get_lock():
                    games_counter.value += done
            # ship many games per message: per-message overhead, not bytes,
            # is what limits the learner's intake on slow cores
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

    from ..ai.features import DIMS, PHASES
    from ..ai.model import Model, replace_retry, save_params
    from .torch_model import SanDaHaNet

    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    cfg.save(os.path.join(run_dir, "config.json"))
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
    torch.manual_seed(cfg.seed)
    if device == "cpu":
        torch.set_num_threads(cfg.learner_threads or max(2, min(8, (os.cpu_count() or 8) // 4)))
    arch = cfg.arch()
    init_model = None
    if cfg.init_from and cfg.init_from.endswith(".npz"):
        init_model = Model.load(cfg.init_from)
        if init_model.meta.get("arch"):
            arch = init_model.meta["arch"]  # warm start keeps the checkpoint's architecture
    elif cfg.init_from:
        arch = torch.load(cfg.init_from, map_location="cpu", weights_only=False).get("arch", arch)
    net = SanDaHaNet(arch).to(device)
    games_offset = 0
    if cfg.init_from:
        if init_model is not None:
            net.load_numpy(init_model.params)
            games_offset = int(init_model.meta.get("games", 0))
        else:
            sd = torch.load(cfg.init_from, map_location=device, weights_only=False)
            net.load_state_dict(sd["net"])
            games_offset = int(sd.get("games", 0))
        log(f"warm start from {cfg.init_from} ({games_offset} games)")
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    def lr_at(now, games):
        """Learning-rate schedule driven by elapsed time (or games if no time limit)."""
        if cfg.lr_schedule == "constant":
            return cfg.lr
        if cfg.max_hours:
            frac = (now - t_start) / (cfg.max_hours * 3600)
        elif cfg.max_games:
            frac = (games - games_offset) / cfg.max_games
        else:
            return cfg.lr
        frac = min(1.0, max(0.0, frac))
        if cfg.lr_schedule == "linear":
            w = 1 - frac
        else:
            w = 0.5 * (1 + math.cos(math.pi * frac))
        return cfg.lr_end + (cfg.lr - cfg.lr_end) * w
    meta = {"name": cfg.name, "arch": arch, "rules": cfg.rules, "version": 1}

    def export(path, games):
        m = dict(meta, games=games, time=time.time())
        save_params(path, net.to_numpy(), m)

    latest = os.path.join(run_dir, "latest.npz")
    ver_path = os.path.join(run_dir, "latest.ver")
    version = 0

    def push(games):
        nonlocal version
        export(latest, games)
        version += 1
        with open(ver_path + ".tmp", "w") as f:
            f.write(str(version))
        replace_retry(ver_path + ".tmp", ver_path)

    push(games_offset)

    # children inherit the environment: keep numpy single-threaded in actors
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=256)
    stop_event = ctx.Event()
    games_counter = ctx.Value("q", games_offset)
    n_actors = cfg.resolved_actors()
    cfg_dict = json.loads(json.dumps(cfg.__dict__))
    actors = [ctx.Process(target=actor_main, args=(i, run_dir, q, cfg_dict, stop_event, games_counter),
                          daemon=True) for i in range(n_actors)]
    for p in actors:
        p.start()
    log(f"device={device} actors={n_actors} arch={arch}")

    bufs = {}
    for p in PHASES:
        cap = max(20_000, int(cfg.buffer_size * PHASE_SHARE[p]))
        bufs[p] = Replay(cap, *DIMS[p])
    steps = {p: 0 for p in PHASES}
    loss_acc = {p: [] for p in PHASES}
    np_rng = np.random.default_rng(cfg.seed)
    total_steps = 0
    last_metric = time.time()
    last_games = games_offset
    last_eval = time.time()
    eval_proc = None
    prev_ckpt = None
    min_fill = {p: min(5_000, bufs[p].cap // 4) for p in PHASES}
    status["state"] = "running"
    _write_json(status_path, status)

    def do_eval(games, wait=False):
        nonlocal eval_proc, prev_ckpt
        ck = os.path.join(run_dir, "checkpoints", f"ckpt_{games:09d}.npz")
        export(ck, games)
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "games": games, "arch": arch},
                   os.path.join(run_dir, "latest.pt"))
        if eval_proc is not None and eval_proc.is_alive():
            if not wait:
                return
            eval_proc.join()
        others = [("vs_baseline", cfg.baseline), ("vs_prev", prev_ckpt)]
        eval_proc = ctx.Process(target=eval_main,
                                args=(run_dir, ck, games, others, cfg.eval_deals, cfg.rules, 12345),
                                daemon=False)
        eval_proc.start()
        prev_ckpt = ck
        if wait:
            eval_proc.join()

    reason = "finished"
    t_ingest = t_learn = 0.0
    try:
        while True:
            # ---- ingest
            t0 = time.time()
            got = 0
            while got < 64:
                try:
                    batch = q.get(timeout=0.05 if got == 0 else 0)
                except queue_mod.Empty:
                    break
                for p, (S, A, T) in batch.items():
                    bufs[p].add(S, A, T)
                got += 1
            t1 = time.time()
            t_ingest += t1 - t0
            # ---- learn
            cur_lr = lr_at(t1, games_counter.value)
            for grp in opt.param_groups:
                grp["lr"] = cur_lr
            for p in PHASES:
                b = bufs[p]
                if b.size < min_fill[p]:
                    continue
                budget = int(b.added * cfg.replay_ratio / cfg.batch_size) - steps[p]
                for _ in range(min(budget, 8)):
                    S, A, T = b.sample(cfg.batch_size, np_rng)
                    x = torch.from_numpy(np.concatenate([S, A], 1)).to(device, non_blocking=True).float() / 255.0
                    t = torch.from_numpy(T).to(device)
                    out = net(p, x)
                    l_q = Fn.mse_loss(out[:, 0], t[:, 0])
                    l_w = Fn.binary_cross_entropy_with_logits(out[:, 1], t[:, 1])
                    l_p = Fn.mse_loss(out[:, 2], t[:, 2])
                    loss = l_q + cfg.loss_win_weight * l_w + cfg.loss_pts_weight * l_p
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 40.0)
                    opt.step()
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
                rec = {"type": "train", "time": now, "elapsed": now - t_start, "games": games,
                       "games_per_s": (games - last_games) / (now - last_metric),
                       "steps": total_steps, "version": version, "lr": cur_lr,
                       "epsilon": cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) *
                       min(1.0, games / max(1, cfg.epsilon_decay_games))}
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
                log(f"games={games} g/s={rec['games_per_s']:.1f} steps={total_steps} "
                    f"loss_play={rec.get('loss_play', float('nan')):.4f}")
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
            export(os.path.join(run_dir, "final.npz"), games)
            status["state"] = "evaluating"
            _write_json(status_path, status)
            do_eval(games, wait=True)
        except Exception:
            status["error"] = traceback.format_exc()
        for p in actors:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        status.update({"state": "error" if reason == "error" else "finished", "reason": reason,
                       "end": time.time(), "games": games})
        _write_json(status_path, status)
        log(f"training {reason}: {games} games")
    return run_dir
