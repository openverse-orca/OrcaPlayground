#!/usr/bin/env bash
# 一键清理旧进程并启动 run_fluid_sim_auto.py（水壶自动轨迹配置）
#
# GUI 参数（传给 run_fluid_sim.py）：
#   --gui          启用 SPlisHSPlasH / OrcaSPH 可视化窗口
#   --mujoco-gui   启用 MuJoCo 原生查看器（launch_passive）
#   --all-gui      同时启用上述两者
#
# 示例：
#   ./run_fluid_sim_auto.sh --gui
#   ./run_fluid_sim_auto.sh --mujoco-gui
#   ./run_fluid_sim_auto.sh --gui --mujoco-gui
#   ./run_fluid_sim_auto.sh --all-gui --max-sim-time 20
#   ORCA_LEVEL_NAME=FluidTest_Hotel_Bar_Fangfang_AutoMove ./run_fluid_sim_auto.sh --gui
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_PY="${SCRIPT_DIR}/run_fluid_sim_auto.py"
CONDA_ENV="${ORCA_CONDA_ENV:-orca-apr24}"

SPH_GUI=false
MUJOCO_GUI=false
EXTRA_ARGS=()

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    echo ""
    echo "额外参数会原样传给 run_fluid_sim_auto.py（如 --max-sim-time 20 --max-steps 0）。"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gui|--sph-gui)
            SPH_GUI=true
            shift
            ;;
        --mujoco-gui)
            MUJOCO_GUI=true
            shift
            ;;
        --all-gui)
            SPH_GUI=true
            MUJOCO_GUI=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "[fluid_auto] 停止旧进程..."
pkill -f "run_fluid_sim_auto.py|run_fluid_sim.py|orcalink --port 50351|SPHSimulator" 2>/dev/null || true
sleep 1

REMAINING=$(ps -ef | grep -E "run_fluid_sim_auto.py|run_fluid_sim.py|orcalink --port 50351|SPHSimulator" | grep -v grep || true)
if [[ -n "$REMAINING" ]]; then
    echo "[fluid_auto] 警告：仍有相关进程："
    echo "$REMAINING"
else
    echo "[fluid_auto] 无残留进程。"
fi

GUI_ARGS=()
[[ "$SPH_GUI" == true ]] && GUI_ARGS+=(--gui)
[[ "$MUJOCO_GUI" == true ]] && GUI_ARGS+=(--mujoco-gui)

echo "[fluid_auto] SPH GUI:    $SPH_GUI"
echo "[fluid_auto] MuJoCo GUI: $MUJOCO_GUI"
echo "[fluid_auto] 启动: python ${AUTO_PY} ${GUI_ARGS[*]} ${EXTRA_ARGS[*]}"

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

exec python "$AUTO_PY" "${GUI_ARGS[@]}" "${EXTRA_ARGS[@]}"
