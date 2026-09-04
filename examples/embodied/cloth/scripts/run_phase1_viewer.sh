#!/usr/bin/env bash
# 使用 orcapc（或当前 CONDA_PREFIX）内的 Python，避免误用 /usr/bin/python3。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/OrcaSys/miniconda3/envs/orcapc/bin/python" ]]; then
  PY="${HOME}/OrcaSys/miniconda3/envs/orcapc/bin/python"
else
  echo "错误: 未找到带 mujoco 的 Python。请先: conda activate orcapc" >&2
  exit 1
fi
exec "$PY" "$ROOT/scripts/run_phase1_viewer.py" "$@"
