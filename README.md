# SanDaHa-Zero · 三打哈 AI

**English** | [中文](README.zh-CN.md)

A self-play reinforcement-learning AI for **San Da Ha (三打哈)**, the Hunan trick-taking card game. It ships with a one-click local web client: play against the AI, play over your LAN, learn in teaching mode with move-by-move commentary, and train models from a visual dashboard.

- **Full rules**: bidding (including the forced-bid rule for 7+ permanent trumps), trump choice (including no-trump), surrender, burying the bottom, singles / pairs / tractors, trump throws (**side-suit throws are not allowed**), ruffing and discarding, bottom-capture (扣底) multipliers, and the complete 大光/小光/保庄/下庄/小倒/大倒 settlement. Every point where regional rule sets disagree is configurable. See [docs/RULES.zh-CN.md](docs/RULES.zh-CN.md).
- **One model makes every decision**: bidding, trump or surrender, burying, and card play (including going for 扣底). The training target is the **expected score including multipliers**, so win probability and multipliers are traded off automatically.
- **Light deployment**: playing needs only numpy (no PyTorch, no GPU). Training uses PyTorch, which the dashboard can install with one click.

![table](docs/img/table.png)

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

<!--RESULTS_SIZE-->

### Results

<!--RESULTS_MAIN-->

## Layout

```
sandaha/   cards, rules, moves, engine (pure Python rules engine)
           bots/ (rule-based baseline), ai/ (features, numpy inference, agent)
           train/ (DMC actors + learner, CLI), server/ (FastAPI + WebSocket), web/ (vanilla JS client)
models/    pretrained models (sandaha-best.npz is the default AI)
runs/      training output
docs/      rules and training docs
tests/     pytest unit tests
```

## License

**[PolyForm Noncommercial 1.0.0](LICENSE)**:
- Free for personal, research, hobby, educational and non-profit use.
- **Commercial use requires a separate license.** Contact the author via GitHub Issues.

PolyForm Noncommercial is a *source-available* license, not an OSI-approved open-source license. It was chosen deliberately so that commercial use requires permission.
