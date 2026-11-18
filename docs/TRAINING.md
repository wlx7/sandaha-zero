# Training guide

## 1. Local training (web dashboard)

1. Start with `start.bat` / `start.sh` and open **训练 (Training)**.
2. If PyTorch is missing, click **安装训练依赖** (install). The CUDA build is used when an NVIDIA GPU is detected.
3. Configure and start. Training runs in a background process; closing the page does not stop it.
4. Publish a model with **发布为对弈AI**. A model named `sandaha-best` becomes the default AI.

## 2. Parameters (v2.0 Dueling FiLM-ResNet)

| Parameter | Default | Notes |
|---|---|---|
| `preset` | `medium` | `tiny` / `small` / `medium` / `large` / `xlarge` (see table below) |
| `arch_type` | `dueling_film_v2` | Pre-LN Residual State Tower + Dueling Value/Advantage + FiLM State-Action gating (backward-compatible with `mlp_v1`) |
| `feat_ver` | `2` | Tactical v2 features: pair-voids (`pair_void`), bidding memory, exact settlement step margins, boss cards & closing-seat flags |
| `num_actors` | `0` (= CPU threads − 2) | **Primary throughput factor**: self-play runs on CPU workers |
| `games_per_actor` / `send_games` | `16` / `32` | Batched lockstep games per actor; raise to `32 / 32` on many-core servers |
| `device` | `auto` | Learner runs on GPU (`bfloat16` TensorCore autocast) or CPU |
| `lr` / `lr_end` | `3e-4` → `1.5e-5` | Per-phase `AdamW` optimizers with linear warmup + cosine annealing |
| `ema_decay` | `0.9995` | Polyak EMA weights used for checkpoint and `final.npz` export |
| `batch_size` / `replay_ratio` | `2048` / `1.5` | Use `2048–4096` on GPU; `512` / `0.5` on CPU |
| `q_noise_min` / `q_noise_max` | `0.002` / `0.032` | Ape-X log-spaced plausible Q-value Gaussian exploration across actors |
| `league_frac` | `0.20` | Share of games played against historical checkpoints, baseline model, and `RuleBot` |
| `no_surrender_games` / `no_surrender_frac` | `300k` / `0.12` | Surrender curriculum; reduce `no_surrender_frac` to `0.02–0.03` during late fine-tuning to match the true game distribution |

Model presets (`dueling_film_v2` + `feat_ver=2`):

| Preset | Play hidden | Aux hidden | Params | Role |
|---|---|---|---|---|
| `tiny` | 256-192 | 160-128 | 0.83M | Quick smoke testing |
| `small` | 512-384-192 | 256-160 | 2.37M | Lightweight (`sandaha-lite.npz`) |
| `medium` (default) | 768-512-256 | 384-256 | 4.67M | Default champion (`sandaha-best.npz`) |
| `large` | 1024-768-384 | 512-256 | 7.79M | High-compute runs |
| `xlarge` | 1536-1024-512 | 768-384 | 16.7M | Experimental |

## 3. Output

`runs/<name>/` holds `config.json`, `metrics.jsonl`, `evals.jsonl`, `checkpoints/ckpt_<games>.npz` (playable NumPy EMA models), `latest.pt` (resumable PyTorch state with per-phase optimizers & EMA) and `final.npz`.

## 4. Cloud Training (Vast.ai / RunPod)

Pick a machine with **many CPU cores** plus a single NVIDIA GPU (RTX 3090 / 4090 / A6000):

```bash
git clone https://github.com/wlx7/sandaha-zero.git && cd sandaha-zero
pip install -e .

# Stage 1: Main cosine run
CUDA_VISIBLE_DEVICES=0 python -u -m sandaha.train.cli train \
  --name v2-medium --preset medium --actors 156 --games-per-actor 32 --send-games 32 \
  --batch 4096 --buffer 1500000 --replay 0.9 --lr 3e-4 --lr-end 1.5e-5 --lr-schedule cosine \
  --league-frac 0.22 --hours 3.0 --eval-minutes 15 --eval-deals 500 \
  --baseline models/sandaha-best.npz

# Stage 2: True-MDP low-LR fine-tuning (no_surrender_frac=0.02, cosine down to 3e-6)
CUDA_VISIBLE_DEVICES=0 python -u -m sandaha.train.cli train \
  --name v2-medium-true --preset medium --init runs/v2-medium/latest.pt \
  --actors 156 --games-per-actor 32 --send-games 32 --batch 4096 --replay 0.85 \
  --lr 2.5e-5 --lr-end 3e-6 --lr-schedule cosine --no-surrender-games 0 --no-surrender-frac 0.02 \
  --league-frac 0.25 --hours 3.0 --eval-minutes 15 --eval-deals 500 \
  --baseline models/sandaha-best.npz
```

**Always terminate cloud instances immediately after downloading artifacts.**

## 5. Evaluation & Standalone Packaging

```bash
# Duplicate-deal evaluation
python -m sandaha.train.cli eval models/sandaha-best.npz rule --deals 500
python scripts/benchmark.py models/sandaha-best.npz --deals 6000 --out docs/results.json

# Build standalone one-click Windows executable (dist/SanDaHa-Zero-Windows-x64.exe)
pip install pyinstaller
python scripts/build_windows.py

# Build signed standalone Android APK (requires JDK 17 + Android SDK + Python 3.11)
gradle -p android assembleRelease
```
