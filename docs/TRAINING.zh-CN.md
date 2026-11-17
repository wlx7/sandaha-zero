# 训练指南

## 1. 本地训练（网页）

1. 运行 `start.bat` / `start.sh`，进入首页的“训练”。
2. 如果显示“未安装 PyTorch”，点“安装训练依赖”（有 NVIDIA 显卡时自动安装 CUDA 版）。
3. 设置参数后点“开始训练”。训练在后台子进程中运行，可以关闭网页，下次打开会继续显示进度。
4. 满意后在训练详情中点“发布为对弈 AI”。命名为 `sandaha-best` 的模型会成为默认 AI。

## 2. 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| 模型大小 | medium | tiny/small/medium/large/xlarge，见下表。越大越强，但自对弈越慢 |
| 自对弈进程数 | 0（自动 = CPU 线程数 − 2） | **最重要的速度因素**。训练瓶颈在 CPU 自对弈 |
| games_per_actor / send_games | 8 / 16 | 每个进程同时进行多局，决策合并成一次批量网络计算；打完的牌局打包发给学习器。多核机器的瓶颈在内存带宽，可调到 32 |
| 设备 | 自动 | 学习器使用 GPU 或 CPU。没有 GPU 时 CPU 也能训练，学习器线程见 `learner_threads` |
| 学习率 | 3e-4 | Adam |
| 批大小 | 1024 | CPU 学习器建议 512 |
| 样本复用率 | 2.0 | 每个样本平均被训练几次。CPU 学习器建议 0.5 |
| 探索率 ε | 0.10 → 0.01 | 线性衰减，衰减局数默认 30 万 |
| 评测间隔 / 牌局数 | 10 分钟 / 200 副 | 每副牌换位各打一次（复式赛），可消除牌运影响 |
| 基准模型 | 无 | 选一个旧模型，曲线中会多一条“对基准模型胜率” |
| 继续训练自 | 无 | 从已有模型热启动（结构需一致） |
| 投降课程 | 30 万局 | 前 N 局禁止投降，防止学成“一律投降” |

模型预设（出牌网络 / 其它网络隐层）：

| 预设 | 出牌网络 | 参数量 | 每进程自对弈速度* |
|---|---|---|---|
| tiny | 256-256 | 0.44M | 最快 |
| small | 512-256-256 | 1.0M | ~11 局/秒 |
| medium（默认） | 768-512-256 | 1.6M | ~6 局/秒 |
| large | 1024-1024-512-256 | 3.5M | ~3 局/秒 |
| xlarge | 2048-1024-1024-512 | 6.6M | 很慢 |

\* 实测：笔记本 14 个进程并行时的单进程速度（含投降等短局），不同 CPU 差异较大。算力有限时推荐 small，算力充足时推荐 medium，详见 README 中的“模型大小选择”。

## 3. 输出文件

```
runs/<名称>/config.json      配置
runs/<名称>/metrics.jsonl    训练指标（每 10 秒一行）
runs/<名称>/evals.jsonl      评测结果
runs/<名称>/checkpoints/     每次评测的检查点 ckpt_<局数>.npz（numpy，可直接对弈）
runs/<名称>/latest.pt        PyTorch 检查点（含优化器，可续训）
runs/<名称>/final.npz        最终模型
```

## 4. 在 RunPod 上训练

自对弈主要消耗 CPU，所以**优先选 vCPU 多的机器**，GPU 不是必须的。实测 32 vCPU 的 CPU Pod（cpu5c，约 $1.12/小时）约 180 局/秒，是 20 线程笔记本的 2 倍。

1. 新建 Pod：镜像选 `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`，开放 HTTP 端口 `8000`。
2. 在 Pod 终端执行：

```bash
cd /workspace && git clone https://github.com/wlx7/sandaha-zero.git && cd sandaha-zero
# 纯 CPU 机器：
bash scripts/runpod_train.sh main medium 3 --actors 24 --learner-threads 6 --device cpu --batch 512 --replay 0.5
# 有 GPU 的机器：
bash scripts/runpod_train.sh main medium 3 --actors 30
```

3. 查看进度：`tail -f runs/main/metrics.jsonl`。
4. 下载模型：脚本会在 8000 端口提供 `runs/` 目录，浏览器访问 `https://<podid>-8000.proxy.runpod.net/main/final.npz` 即可下载。也可以把整个 `runs/main` 目录复制到本地的 `runs/` 下，在训练中心查看完整曲线。
5. **用完记得停止或删除 Pod**，否则会持续计费。

### Vast.ai / AutoDL 等其它云

脚本是通用 Linux 脚本，不依赖 RunPod。选 PyTorch 镜像，**优先选 CPU 线程多的机器**（实测 Vast.ai 上 128 线程 EPYC + RTX 3070 开启批量自对弈后约 2,900 局/秒，是 32 vCPU RunPod CPU Pod 的 12 倍以上；不开批量时最高只有约 490–1,450 局/秒，而且受内存带宽限制，进程越多反而越慢）：

```bash
git clone https://github.com/wlx7/sandaha-zero.git && cd sandaha-zero
bash scripts/runpod_train.sh main medium 3 --actors 116 --games-per-actor 32 --batch 2048 --replay 0.75
```

下载结果：为 `http://localhost:8000` 开一个 HTTP 隧道 / 端口映射即可。中断后可用 `--init runs/<名称>/latest.pt` 接着训练。

## 5. 命令行评测

```bash
# 复式赛：A 坐 0/2 号位，B 坐 1/3 号位，每副牌交换座位再打一次
python -m sandaha.train.cli eval models/sandaha-best.npz rule --deals 500
python -m sandaha.train.cli eval runs/main/final.npz runs/main/checkpoints/ckpt_000500000.npz
```

输出说明：`win_rate` 是 A 方两人合计得分为正的局数比例；`avg_score` 是 A 方每局的平均净得分（底注）；`banker_rate` 和 `banker_win_rate` 分别是 A 方做庄的比例和做庄胜率。
