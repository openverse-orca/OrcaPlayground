#!/bin/bash
# XPBD/PBD 布料粒子显示：启动 OrcaEditor + cloth_demo + Play，等待 gRPC 50251。
# 与 examples/fluid 隔离；勿在 fluid 目录添加 PBD 脚本。
set -euo pipefail

CLOTH_FOLD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCA_ROOT="${ORCA_ROOT:-$HOME/OrcaApr24}"

ORCA_EDITOR_DIR="${ORCA_EDITOR_DIR:-$ORCA_ROOT/OrcaStudio_2409/build/bin/profile}"
ORCA_EDITOR="${ORCA_EDITOR:-$ORCA_EDITOR_DIR/OrcaEditor}"
PROJECT_PATH="${PROJECT_PATH:-$ORCA_ROOT/OrcaStudio_2409}"
AUTO_SCRIPT="${AUTO_SCRIPT:-$CLOTH_FOLD_DIR/auto_start_cloth_demo.py}"
REPLAY_BIN="${REPLAY_BIN:-$ORCA_ROOT/XPBD/Orca/build/ParticleRender/pbd_particle_replay}"
REPLAY_CONFIG="${REPLAY_CONFIG:-$ORCA_ROOT/XPBD/Orca/ParticleRender/config/pbd_particle_render.json}"
GRPC_PORT="${GRPC_PORT:-50251}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-180}"

usage() {
    cat <<EOF
Usage: $0 [--stop] [--no-wait] [--editor-only]

  PBD 布料联调（与 fluid/SPH 无关）：
  启动 OrcaEditor → cloth_demo → Play → 等待 localhost:$GRPC_PORT

  --stop         结束 OrcaEditor / pbd_particle_replay
  --no-wait      不等待 50251
  --editor-only  同 --no-wait

目录: $CLOTH_FOLD_DIR
EOF
}

wait_for_port() {
    local port=$1 max=$2 waited=0
    echo "[CLOTH_FOLD] Waiting for localhost:$port (max ${max}s)..."
    while [ "$waited" -lt "$max" ]; do
        if ss -lntp 2>/dev/null | grep -q ":$port "; then
            echo "[CLOTH_FOLD] Port $port is listening."
            ss -lntp 2>/dev/null | grep ":$port " || true
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        if [ $((waited % 20)) -eq 0 ]; then
            echo "[CLOTH_FOLD]   ... still waiting (${waited}/${max}s)"
        fi
    done
    echo "[CLOTH_FOLD] ERROR: port $port not open. cloth_demo in Play?"
    return 1
}

stop_all() {
    echo "[CLOTH_FOLD] Stopping..."
    pkill -f pbd_particle_replay 2>/dev/null || true
    pkill -f OrcaEditor 2>/dev/null || true
    sleep 2
    pgrep -a OrcaEditor 2>/dev/null || echo "[CLOTH_FOLD] OrcaEditor stopped."
}

NO_WAIT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stop) stop_all; exit 0 ;;
        --no-wait|--editor-only) NO_WAIT=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown: $1"; usage; exit 1 ;;
    esac
done

if [ ! -x "$ORCA_EDITOR" ]; then
    echo "[CLOTH_FOLD] ERROR: OrcaEditor not found: $ORCA_EDITOR"
    exit 1
fi

if [ -z "${DISPLAY:-}" ]; then
    echo "[CLOTH_FOLD] WARNING: DISPLAY empty — need GUI session."
fi

if pgrep -x OrcaEditor >/dev/null 2>&1; then
    echo "[CLOTH_FOLD] OrcaEditor already running (PID $(pgrep -x OrcaEditor))."
else
    export ORCA_LEVEL_NAME=cloth_demo
    echo "[CLOTH_FOLD] Starting OrcaEditor..."
    echo "[CLOTH_FOLD]   Project: $PROJECT_PATH"
    echo "[CLOTH_FOLD]   Script:  $AUTO_SCRIPT"
    cd "$ORCA_EDITOR_DIR"
    "$ORCA_EDITOR" \
        --project-path="$PROJECT_PATH" \
        --skipWelcomeScreenDialog \
        --runpython \
        "$AUTO_SCRIPT" \
        &
    echo "[CLOTH_FOLD] OrcaEditor PID: $!"
fi

if [ "$NO_WAIT" = true ]; then
    echo "[CLOTH_FOLD] Editor started. Play manually, then:"
    echo "  $CLOTH_FOLD_DIR/run_pbd_particle_replay.sh"
    exit 0
fi

wait_for_port "$GRPC_PORT" "$MAX_WAIT_SEC"

echo ""
echo "[CLOTH_FOLD] Ready. Run replay:"
echo "  $CLOTH_FOLD_DIR/run_pbd_particle_replay.sh"
