# SanDaHa-Zero · 三打哈 AI

**English** | [中文](README.zh-CN.md)

A self-play reinforcement-learning AI for **San Da Ha (三打哈)**, the Hunan trick-taking card game. It ships with a one-click local web client: play against the AI, play over your LAN, learn in teaching mode with move-by-move commentary, and train models from a visual dashboard.

- **Full rules**: bidding (including the forced-bid rule for 7+ permanent trumps), trump choice (including no-trump), surrender, burying the bottom, singles / pairs / tractors, trump throws (**side-suit throws are not allowed**), ruffing and discarding, bottom-capture (扣底) multipliers, and the complete 大光/小光/保庄/下庄/小倒/大倒 settlement. Every point where regional rule sets disagree is configurable. See [docs/RULES.zh-CN.md](docs/RULES.zh-CN.md).
- **One model makes every decision**: bidding, trump or surrender, burying, and card play (including going for 扣底). The training target is the **expected score including multipliers**, so win probability and multipliers are traded off automatically.
- **Light deployment**: playing needs only numpy (no PyTorch, no GPU). Training uses PyTorch, which the dashboard can install with one click.

![teaching mode](docs/img/table.jpg)

<p><img src="docs/img/mobile.jpg" width="260" alt="mobile"> <img src="docs/img/train.jpg" width="540" alt="training dashboard"></p>

## Quick start

**Windows**: double-click `start.bat`. **macOS / Linux**: run `./start.sh`.

The first run installs [uv](https://github.com/astral-sh/uv) and the dependencies (about a minute), then opens `http://127.0.0.1:8765`. Other phones or PCs on the same LAN can open the address shown on the home page, or scan its QR code.

```bash
uv run sandaha                      # or: pip install -e . && sandaha
sandaha --port 9000 --no-browser
```

> On Windows, allow Python through the firewall for *private networks* so other devices can connect.

## Features

### Play

| Mode | Description |
|---|---|
| **Battle** | Each of the 4 seats can be *me (this device)*, *wait for a player*, or *AI* (any trained model, or the rule-based AI). LAN players join the room and click "坐这里" (sit here). Every decision has a **30 s** limit. On timeout the server plays a random legal move (an expired bid counts as a pass). Set all seats to AI to watch AI games. |
| **Teaching** | You plus 3 AIs, **no time limit**. AI hands can be revealed. Every decision, including yours, gets commentary: the ranking of candidate moves, expected score (in units of the base stake), win probability, and predicted final 闲 points, plus a verdict (best / acceptable / mistake). The "提示" (hint) button pre-selects the AI's recommended cards. |

The table:

- Responsive layout for phones and desktops. The hand is grouped (trumps first, then each side suit) and wraps into rows, so every card's rank and suit stay readable. Trump cards are tinted gold and marked 主.
- Card fly-in animations, a highlight on the active player, and countdown bars.
- **Card counter** (toggle): every card still out, plus the remaining points.
- **History**: the bidding log, every trick (who played what, who won, how many points), and past hands.
- Known voids are shown automatically ("缺♠" means void in spades). The result dialog shows the 扣底 multiplier, the scores and the buried cards.

### Training dashboard

- Adjustable: model size (tiny / small / medium / large / xlarge or custom hidden layers), device, number of self-play workers, learning rate, batch size, replay ratio, ε schedule, game and time budget, evaluation interval and size, **baseline model** (to compare against older versions), warm start, and the surrender curriculum.
- Live charts: win rate vs the rule AI, vs the baseline, and vs the previous checkpoint; net score per game; banker win rate; losses of the four networks; self-play throughput.
- Checkpoints are saved at every evaluation. From the dashboard you can publish any of them as a playable AI, or run a head-to-head evaluation between any two models.

CLI (servers, RunPod):

```bash
python -m sandaha.train.cli train --name myrun --preset medium --hours 3
python -m sandaha.train.cli eval models/sandaha-best.npz rule --deals 500
```

See [docs/TRAINING.md](docs/TRAINING.md) for cloud training on RunPod.

## How the AI works

**Algorithm: Deep Monte-Carlo (DMC), as in DouZero**, adapted to San Da Ha.

Why DMC? The game has imperfect information (three hidden hands and a hidden bottom), a huge combinatorial and variable-size action space (follows, throws, discards), asymmetric 1-vs-3 teams, four different kinds of decisions, and sparse multiplied rewards.

The alternatives each break somewhere:
- CFR does not scale to information sets this large.
- ISMCTS / PIMC is slow at play time and suffers from strategy fusion.
- Policy-gradient methods struggle with variable combinatorial action heads.

DMC encodes the action as part of the network input and scores Q(s, a) for every legal move. That handles any action set, it is proven on DouDizhu, and it scales with cheap CPU self-play.

Design details:

- **Four Q-networks**: bidding, trump/surrender, burying, and play. Burying is a chain of 8 single-card decisions instead of C(29,8) ≈ 4.3M combinations.
- **Trump-canonical card encoding**: 46 card slots × 2 bits, laid out with the trump suit first so knowledge transfers between trump suits. The trump network re-encodes the hand under each candidate trump.
- **Play state (971 features) + action (104 features)**: own hand, cards played by each seat, the current trick, unseen cards, own buried cards, the banker's seat, known voids, the contract, 闲 points so far, trick points, and the current winner.
- **Reward = actual settlement score / 9**, with all multipliers, so the agent maximises expected score. Two auxiliary heads predict win probability and final 闲 points; teaching mode uses them for commentary.
- **Candidate pruning**: when a decision has more than 96 legal combinations, the agent keeps heuristic candidates plus random samples.
- **Surrender curriculum**: early on, the banker always loses, and DMC collapses into "always surrender". Surrender is therefore disabled for the first 300k games and for 30% of games afterwards.
- **numpy inference** with a split first layer (the state part is computed once per decision): about 1 ms per decision for the medium model.

### Model size

Each size trained for **45 minutes** on a laptop (20 threads, 14 self-play workers):

| Preset | Params | Self-play speed | Games in 45 min | Win rate vs rule AI |
|---|---|---|---|---|
| small | 1.0M | 151 games/s | 405k | 65–68% |
| medium | 1.6M | 87 games/s | 233k | 58–62% |
| large | 3.5M | 44 games/s | 119k | 51–59% |

Head-to-head (800 games each): small beats medium **69.2%**, small beats large **67.2%**, and medium beats large **56.8%**.

With limited compute, **data, not capacity, is the bottleneck**, so smaller networks win. With a 128-thread Vast.ai machine, medium was trained to 53.7M games and then beat small 62.0%. medium is therefore the default (`models/sandaha-best.npz`), and small ships as a lightweight option (`models/sandaha-lite.npz`).

### Results

**Training** (≈53.7M self-play games in total; the full curves are in the dashboard under `official`):

| Stage | Machine | Games | Win rate vs rule AI |
|---|---|---|---|
| 1 | RunPod 32-vCPU CPU pod ($1.12/h, ~40 min) | 0 → 0.46M | 72.2% |
| 2 | Vast.ai 128 threads + RTX 3070, constant LR | 0.35M → 3.68M | 74–75% (plateau) |
| 3 | same, cosine LR decay 3e-4 → 2e-5 for 2 h | 3.68M → 14.2M | 76.1%, +2.13 per game |
| 4 | Vast.ai 128-thread EPYC 7B12, batched actors (~2,900 games/s), cosine LR 1e-4 → 1e-5, ~4.5 h | 14.2M → 53.7M | **76–78%**, **+2.2** per game |

The learning-rate decay broke the plateau. With a constant LR, the 3.68M model was even (50.0%) against the 2.35M model. After the decay, the 14.2M model beats the 2.35M model 57.1%.

Stage 4 first fixed self-play throughput. On many-core machines every actor process streamed its own copy of the weights for each decision, so memory bandwidth, not compute, was the limit. Adding more processes made things slower. Each actor now plays 32 games at once and evaluates all their decisions in one batched network pass, and finished hands are sent to the learner in bundles. Together these raised throughput on the same machine from ≈490 to ≈2,900 games/s (`games_per_actor`, `send_games`). Gains have now flattened out: the final model is only even (49.8%) against its own 43.2M checkpoint.

**Final benchmark** (`scripts/benchmark.py`; each row is 6,000 deals × seat swap = 12,000 games; ± is a 95% CI):

| Opponent | Win rate | Net score / game | Our banker rate | Banker win rate |
|---|---|---|---|---|
| Rule-based AI | **75.4% ± 0.8** | **+2.09** | 18.7% | 72.1% |
| Random | 99.9% ± 0.1 | +5.62 | 0.6% | — |
| Previous release (14.2M games) | 52.4% ± 0.9 | +0.14 | 50.3% | 52.4% |
| Same run @ 16.0M | 52.8% ± 0.9 | +0.17 | 50.9% | 52.6% |
| Same run @ 29.2M | 50.9% ± 0.9 | +0.08 | 51.2% | 51.0% |
| Same run @ 43.2M | 49.8% ± 0.9 | −0.02 | 50.5% | 49.7% |
| small / lite (405k) | 62.0% ± 0.9 | +0.99 | 52.4% | 62.4% |

On the same 6,000 deals, the previous release (14.2M games) scored 74.8% / +2.06 against the rule AI.

<details><summary>Previous release benchmark (14.2M games, 972 games per row)</summary>

| Opponent | Win rate | Net score / game | Our banker rate | Banker win rate |
|---|---|---|---|---|
| Rule-based AI | **73.4% ± 2.8** | **+1.99** | 18.2% | 75.1% |
| Random | 99.8% ± 0.3 | +5.62 | 0.6% | — |
| Old: local medium (233k games) | 74.7% ± 2.7 | +1.43 | 53.4% | 66.1% |
| Old: RunPod (462k) | 57.3% ± 3.1 | +0.56 | 53.6% | 55.5% |
| Old: Vast (2.35M) | 53.5% ± 3.1 | +0.30 | 53.0% | 55.5% |
| Old: Vast (5.0M) | 52.5% ± 3.1 | +0.18 | 52.8% | 52.4% |
| small / lite (405k) | 60.9% ± 3.1 | +0.85 | 52.9% | 61.5% |

</details>

Notes:
- **Win rate** is the share of games where our two seats' combined score is positive. 三打哈 is 1-vs-3 and individual hands are very luck-dependent. The duplicate format (the same deal replayed with seats swapped) cancels most of the luck.
- Against the rule AI, our AI is rarely the banker (18%). The rule AI bids aggressively, and our AI learned to let it take the contract and profit as a defender. When our AI is the banker, it wins 72–75% of the time.
- The AI learned to **surrender** bad banker hands. With a contract ≥ 55, surrender settles as an ordinary loss (−3) and avoids 小倒/大倒 (−6/−9).

**Search (optional)**: for card play, the agent takes the network's top-3 moves, rolls each out to the end of the hand under 8 sampled deals consistent with what it has seen, and picks the best average. Against the same model without search, over 1,200 games: +0.15 ± 0.18 per deal. That is a small gain, not yet statistically significant, so search is off by default. Choose "+ 搜索增强" when picking an AI to enable it.

## Layout

```
sandaha/   cards, rules, moves, engine (pure Python rules engine)
           bots/ (rule-based baseline), ai/ (features, numpy inference, agent)
           train/ (DMC actors + learner, CLI), server/ (FastAPI + WebSocket), web/ (vanilla JS client)
models/    pretrained models: sandaha-best.npz (default, medium), sandaha-lite.npz (lightweight, small)
runs/      training output
docs/      rules and training docs
tests/     pytest unit tests
```

## License

**[PolyForm Noncommercial 1.0.0](LICENSE)**:
- Free for personal, research, hobby, educational and non-profit use.
- **Commercial use requires a separate license.** Contact the author via GitHub Issues.

PolyForm Noncommercial is a *source-available* license, not an OSI-approved open-source license. It was chosen deliberately so that commercial use requires permission.
