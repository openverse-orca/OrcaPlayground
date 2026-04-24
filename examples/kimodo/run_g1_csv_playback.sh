#!/usr/bin/env bash
set -euo pipefail

# Environment that has this project, orca_gym, gymnasium, numpy, and yaml.
PLAYBACK_ENV="${PLAYBACK_ENV:-orcalab}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
ORCAGYM_ADDR="${ORCAGYM_ADDR:-127.0.0.1:50051}"


OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/generated}"
OUTPUT_NAME="${OUTPUT_NAME:-g1_motion}"

CSV_PATH="${CSV_PATH:-$OUTPUT_DIR/$OUTPUT_NAME.csv}"
MODE="${MODE:-direct}"
FPS="${FPS:-30}"
SPEED="${SPEED:-1.0}"
LOOP="${LOOP:-1}"
SPAWN="${SPAWN:-1}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PLAYBACK_ENV"

cd "$PROJECT_ROOT"

host="${ORCAGYM_ADDR%:*}"
port="${ORCAGYM_ADDR##*:}"
for _ in $(seq 1 60); do
  if timeout 1 bash -c ":</dev/tcp/${host}/${port}" 2>/dev/null; then
    break
  fi
  echo "Waiting for OrcaGym server at ${ORCAGYM_ADDR} ..."
  sleep 1
done

args=(
  "$SCRIPT_DIR/playback_g1_csv_orca.py"
  "$CSV_PATH"
  "--orcagym-addr" "$ORCAGYM_ADDR"
  "--mode" "$MODE"
  "--fps" "$FPS"
  "--speed" "$SPEED"
)

if [[ "$LOOP" == "1" ]]; then
  args+=("--loop")
fi

if [[ "$SPAWN" == "0" ]]; then
  args+=("--no-spawn")
fi

exec python "${args[@]}"
