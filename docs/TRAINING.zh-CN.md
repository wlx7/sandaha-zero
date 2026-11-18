# 训练指南

## 1. 本地训练（网页）

1. 运行 `start.bat` / `start.sh`（或直接运行一键版 `SanDaHa-Zero-Windows-x64.exe`），进入首页的“训练”。
2. 如果显示“未安装 PyTorch”，点“安装训练依赖”（有 NVIDIA 显卡时自动安装 CUDA 版）。
3. 设置参数后点“开始训练”。训练在后台子进程中运行，可以关闭网页，下次打开会继续显示进度。
4. 满意后在训练详情中点“发布为对弈 AI”。命名为 `sandaha-best` 的模型会成为默认 AI。

## 2. 参数说明（v2.0 Dueling FiLM-ResNet）

| 参数 | 默认 | 说明 |
|---|---|---|
| 模型大小 (`preset`) | medium | tiny / small / medium / large / xlarge，见下表。越大容量越高，但自对弈越慢 |
| 网络架构 (`arch_type`) | `dueling_film_v2` | Pre-LN 残差状态塔 + Dueling 价值/优势分解 + FiLM 状态-动作门控（兼容旧版 `mlp_v1`） |
| 特征版本 (`feat_ver`) | `2` | 含对子断门 (`pair_void`)、叫分记忆、结算档位裕度、顶牌与关门位检测（兼容旧版 `1`） |
| 自对弈进程数 (`num_actors`) | 0（自动 = CPU 线程数 − 2） | **最重要的速度因素**。训练瓶颈在 CPU 自对弈 |
| `games_per_actor` / `send_games` | 16 / 32 | 每个进程同时进行多局，决策合并成一次批量网络计算；多核服务器建议调到 `32 / 32` |
| 设备 (`device`) | 自动 | 学习器使用 GPU（自动开启 `bfloat16` TensorCore 加速）或 CPU |
| 学习率 (`lr` → `lr_end`) | `3e-4` → `1.5e-5` | 分阶段独立 `AdamW` 优化器 + 余弦衰减 + 线性预热 (`warmup_steps`) |
| EMA 衰减 (`ema_decay`) | `0.9995` | 导出检查点与最终模型时使用 Polyak 指数滑动平均权重，显著降低后期震荡 |
| 批大小 / 样本复用率 | `2048` / `1.5` | GPU 学习器建议 `2048–4096`；CPU 学习器建议 `512` / `0.5` |
| Ape-X Q 噪声 (`q_noise_min/max`) | `0.002` / `0.032` | 不同自对弈进程采用对数分布的 Q 值微扰探索，避免无意义随机出牌污染回报 |
| 联赛自对弈 (`league_frac`) | `0.20` | 一定比例对局与历史检查点池、基准模型和规则 AI 交手，防止策略遗忘与循环克制 |
| 评测间隔 / 牌局数 | 10 分钟 / 300 副 | 每副牌换位各打一次（复式赛），抵消发牌运气 |
| 投降课程 | 30 万局 (`0.12`) | 前 30 万局禁止投降防止初期坍缩；后期续训收敛阶段建议调低 `no_surrender_frac` 至 `0.02–0.03` 以贴合真实博弈分布 |

模型预设（`dueling_film_v2` + `feat_ver=2`）：

| 预设 | 出牌网络隐层 | 叫分/定主/埋底隐层 | 参数量 | 适用场景 |
|---|---|---|---|---|
| tiny | 256-192 | 160-128 | 0.83M | 快速冒烟验证 |
| small | 512-384-192 | 256-160 | 2.37M | 轻量版（`sandaha-lite.npz`） |
| medium（默认） | 768-512-256 | 384-256 | 4.67M | 主力版（`sandaha-best.npz`） |
| large | 1024-768-384 | 512-256 | 7.79M | 超算大规模训练 |
| xlarge | 1536-1024-512 | 768-384 | 16.7M | 极大规模实验 |

## 3. 输出文件

```
runs/<名称>/config.json      配置
runs/<名称>/metrics.jsonl    训练指标（每 10 秒一行）
runs/<名称>/evals.jsonl      评测结果
runs/<名称>/checkpoints/     每次评测的 EMA 检查点 ckpt_<局数>.npz（numpy，可直接对弈）
runs/<名称>/latest.pt        PyTorch 检查点（含四个阶段的优化器与 EMA 状态，可无缝续训）
runs/<名称>/final.npz        最终导出的 EMA 模型
```

## 4. 在云端（Vast.ai / RunPod）训练

自对弈主要消耗 CPU，学习器建议使用任意单卡 NVIDIA GPU（RTX 3090 / 4090 / A6000 等）配合高核数 EPYC CPU：

```bash
git clone https://github.com/wlx7/sandaha-zero.git && cd sandaha-zero
pip install -e .

# 第一阶段：从零训练（156 自对弈进程 × 每进程并行 32 局）
CUDA_VISIBLE_DEVICES=0 python -u -m sandaha.train.cli train \
  --name v2-medium --preset medium --actors 156 --games-per-actor 32 --send-games 32 \
  --batch 4096 --buffer 1500000 --replay 0.9 --lr 3e-4 --lr-end 1.5e-5 --lr-schedule cosine \
  --league-frac 0.22 --hours 3.0 --eval-minutes 15 --eval-deals 500 \
  --baseline models/sandaha-best.npz

# 第二阶段：真实分布精调收官（调低 no_surrender_frac 至 0.02，余弦退火至 3e-6）
CUDA_VISIBLE_DEVICES=0 python -u -m sandaha.train.cli train \
  --name v2-medium-true --preset medium --init runs/v2-medium/latest.pt \
  --actors 156 --games-per-actor 32 --send-games 32 --batch 4096 --replay 0.85 \
  --lr 2.5e-5 --lr-end 3e-6 --lr-schedule cosine --no-surrender-games 0 --no-surrender-frac 0.02 \
  --league-frac 0.25 --hours 3.0 --eval-minutes 15 --eval-deals 500 \
  --baseline models/sandaha-best.npz
```

> **重要提醒**：云端训练结束后请第一时间下载 `final.npz` / `checkpoints/` 并**立即销毁实例**，以免持续计费。

## 5. 命令行评测与打包构建

```bash
# 复式赛评测（A 坐 0/2 号位，B 坐 1/3 号位，每副牌交换座位各打一次）
python -m sandaha.train.cli eval models/sandaha-best.npz rule --deals 500
python scripts/benchmark.py models/sandaha-best.npz --deals 6000 --out docs/results.json

# 构建 Windows 免安装单文件版 EXE（输出至 dist/SanDaHa-Zero-Windows-x64.exe）
pip install pyinstaller
python scripts/build_windows.py

# 构建 Android 独立安装包 APK（需 JDK 17 + Android SDK + Python 3.11，自动签名）
gradle -p android assembleRelease
```
