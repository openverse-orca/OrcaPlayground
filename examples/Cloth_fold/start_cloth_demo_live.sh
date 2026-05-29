#!/bin/bash
# PBD 实时联调：启动 cloth_demo + 50251，并打印 XPBD gRPC demo 命令。
set -euo pipefail

CLOTH_FOLD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCA_ROOT="${ORCA_ROOT:-$HOME/OrcaApr24}"

"$CLOTH_FOLD_DIR/start_cloth_demo.sh" "$@"

echo ""
echo "[CLOTH_FOLD LIVE] Studio ready. In another terminal:"
echo "  export ORCA_ROOT=$ORCA_ROOT"
echo "  export PBD_GRPC=1"
echo "  export PBD_GRPC_CONFIG=\$ORCA_ROOT/XPBD/Orca/ParticleRender/config/pbd_particle_render.json"
echo "  export PARTICLE_INTERVAL=0.0333333"
echo "  cd \$ORCA_ROOT/XPBD/xpbd"
echo "  ./build.sh dual_gripper_cross_v4_grpc"
echo "  ./build/dual_gripper_cross_v4_grpc"
