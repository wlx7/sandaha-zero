#!/usr/bin/env bash
# Train on a RunPod (or any Linux GPU) machine.
#   bash scripts/runpod_train.sh <run-name> [preset] [hours] [extra cli args...]
# Uses the PyTorch already present in the RunPod pytorch image.
# Results land in /workspace/sandaha-zero/runs/<run-name>; a tiny HTTP server on
# port 8000 serves that folder so you can download the models.
set -e
NAME=${1:-runpod}
PRESET=${2:-medium}
HOURS=${3:-3}
shift 3 || true
cd /workspace
if [ ! -d sandaha-zero ]; then
  git clone --depth 1 https://github.com/wlx7/sandaha-zero.git
fi
cd sandaha-zero
git pull --ff-only || true
python -m pip install -q --upgrade numpy fastapi "uvicorn[standard]" segno
python -m pip install -q -e . --no-deps
mkdir -p runs
nohup python -m sandaha.train.cli train --name "$NAME" --preset "$PRESET" --hours "$HOURS" "$@" \
  > "runs/$NAME.log" 2>&1 &
echo "training started: runs/$NAME (log: runs/$NAME.log)"
if ! pgrep -f "http.server 8000" >/dev/null; then
  nohup python -m http.server 8000 --directory runs > /dev/null 2>&1 &
  echo "serving runs/ on port 8000"
fi
