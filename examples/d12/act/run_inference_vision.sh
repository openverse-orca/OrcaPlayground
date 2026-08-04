#!/bin/bash
# run_inference_vision.sh - ACT 视觉推理入口
#
# 用法:
#   bash run_inference_vision.sh                        # 使用默认 checkpoint
#   bash run_inference_vision.sh /path/to/best_model.pt # 指定 checkpoint

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

if [ -x "$HOME/anaconda3/envs/orcalab/bin/python" ]; then
    PYTHON="$HOME/anaconda3/envs/orcalab/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

CHECKPOINT="${1:-$PROJECT_DIR/examples/d12/checkpoints/act_vision_demo/best_model.pt}"
NORM_STATS="${2:-}"
MAX_STEPS="${3:-6300}"
EPISODES="${4:-1}"
CAPTURE_EVERY_N="${5:-50}"
REF_TRAJ="${6:-$PROJECT_DIR/examples/d12/act/ref_trajectory/ref_demo.hdf5}"

if [ -z "$NORM_STATS" ]; then
    NORM_STATS="$(dirname "$CHECKPOINT")/norm_stats.pt"
fi

echo "=========================================="
echo "  D12 ACT 视觉推理"
echo "=========================================="
echo "  Checkpoint: $CHECKPOINT"
echo "  Norm stats: $NORM_STATS"
echo "  Max steps:  $MAX_STEPS"
echo "  Episodes:   $EPISODES"
echo "  Capture:    every $CAPTURE_EVERY_N steps"
echo "  Ref traj:   $REF_TRAJ"
echo "=========================================="

$PYTHON examples/d12/act/run_d12_act.py \
    --checkpoint "$CHECKPOINT" \
    --norm_stats "$NORM_STATS" \
    --max_steps "$MAX_STEPS" \
    --num_episodes "$EPISODES" \
    --frame_skip 5 \
    --exec_mode chunk \
    --ema_alpha 0.9 \
    --capture_images \
    --capture_every_n "$CAPTURE_EVERY_N" \
    --ref_trajectory "$REF_TRAJ" \
    --no_sleep
