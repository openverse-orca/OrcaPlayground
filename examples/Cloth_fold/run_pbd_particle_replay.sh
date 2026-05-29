#!/bin/bash
# 离线 TXT → OrcaStudio（需 Editor 已 Play 且 50251 监听）
set -euo pipefail

ORCA_ROOT="${ORCA_ROOT:-$HOME/OrcaApr24}"
REPLAY_BIN="${REPLAY_BIN:-$ORCA_ROOT/XPBD/Orca/build/ParticleRender/pbd_particle_replay}"
REPLAY_CONFIG="${REPLAY_CONFIG:-$ORCA_ROOT/XPBD/Orca/ParticleRender/config/pbd_particle_render.json}"

if [ ! -x "$REPLAY_BIN" ]; then
    echo "[CLOTH_FOLD] ERROR: build replay first:"
    echo "  cd $ORCA_ROOT/XPBD/Orca && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
    exit 1
fi

if ! ss -lntp 2>/dev/null | grep -q ':50251 '; then
    echo "[CLOTH_FOLD] WARNING: 50251 not listening. Start studio first:"
    echo "  $(dirname "$0")/start_cloth_demo.sh"
fi

exec "$REPLAY_BIN" "$REPLAY_CONFIG" "$@"
