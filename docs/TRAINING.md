# Training guide

## 1. Local training (web dashboard)

1. Start with `start.bat` / `start.sh` and open **训练 (Training)**.
2. If PyTorch is missing, click **安装训练依赖** (install). The CUDA build is used when an NVIDIA GPU is detected.
3. Configure and start. Training runs in a background process. Closing the page does not stop it.
4. Publish a model with **发布为对弈AI**. A model named `sandaha-best` becomes the default AI.

## 2. Parameters

| Parameter | Default | Notes |
|---|---|---|
| preset | medium | tiny / small / medium / large / xlarge (see below) |
| num_actors | 0 (= CPU threads − 2) | **The main speed factor**: training is bound by CPU self-play |
| device | auto | Learner on GPU or CPU (`learner_threads` for CPU) |
| lr / batch_size | 3e-4 / 1024 | Use batch 512 for a CPU learner |
| replay_ratio | 2.0 | Average number of times each sample is used; 0.5 for a CPU learner |
| ε | 0.10 → 0.01 over 300k games | ε-greedy exploration |
| eval | every 10 min, 200 deals | Duplicate format: every deal is played twice with seats swapped |
| baseline | none | Older model to plot "win rate vs baseline" |
| init_from | none | Warm start from a checkpoint (same architecture) |
| no_surrender_games | 300k | Surrender curriculum (see README) |

| Preset | Play network | Params | Self-play / core* |
|---|---|---|---|
| tiny | 256-256 | 0.44M | fastest |
| small | 512-256-256 | 1.0M | ~11 games/s |
| medium (default) | 768-512-256 | 1.6M | ~6 games/s |
| large | 1024-1024-512-256 | 3.5M | ~3 games/s |
| xlarge | 2048-1024-1024-512 | 6.6M | slow |

\* measured per worker with 14 workers on a laptop; CPU dependent. Use small when compute is limited and medium when it is plentiful (see *Model size* in the README).

## 3. Output

`runs/<name>/` holds `config.json`, `metrics.jsonl`, `evals.jsonl`, `checkpoints/ckpt_<games>.npz` (playable numpy models), `latest.pt` (resumable PyTorch state) and `final.npz`.

## 4. RunPod

Self-play is CPU-bound, so **pick many vCPUs**. A GPU is optional. A 32-vCPU CPU pod (`cpu5c`, ≈ $1.12/h) reached ≈180 games/s, twice a 20-thread laptop.

1. Create a pod with image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` and expose HTTP port `8000`.
2. In the pod:

```bash
cd /workspace && git clone https://github.com/wlx7/sandaha-zero.git && cd sandaha-zero
# CPU-only pod
bash scripts/runpod_train.sh main medium 3 --actors 24 --learner-threads 6 --device cpu --batch 512 --replay 0.5
# GPU pod
bash scripts/runpod_train.sh main medium 3 --actors 30
```

3. Watch progress with `tail -f runs/main/metrics.jsonl`.
4. Download from `https://<podid>-8000.proxy.runpod.net/main/final.npz`, or copy the whole `runs/main` folder into your local `runs/` to see the curves in the dashboard.
5. **Stop or terminate the pod when you are done.**

### Vast.ai / other clouds

The script is plain Linux and has nothing RunPod-specific. Choose a PyTorch image with **as many CPU threads as possible**. A 128-thread EPYC + RTX 3070 on Vast.ai reached ≈1,450 games/s, more than 6× the 32-vCPU RunPod pod.

```bash
bash scripts/runpod_train.sh main medium 3 --actors 110 --batch 1024 --replay 1.0
```

Expose `http://localhost:8000` (tunnel or port mapping) to download results. Resume after an interruption with `--init runs/<name>/latest.pt`.

## 5. Evaluation

```bash
python -m sandaha.train.cli eval models/sandaha-best.npz rule --deals 500
```

- `win_rate`: share of games where team A (seats 0 and 2) has a positive combined score.
- `avg_score`: team A's net score per game, in units of the base stake.
- `banker_rate` / `banker_win_rate`: how often team A is the banker, and how often it wins as banker.
