#!/usr/bin/env bash
set -euo pipefail

# One-command workflow:
#   1. Generate a G1 qpos CSV from a text prompt with Kimodo.
#   2. Start OrcaLab if the OrcaGym port is not already listening.
#   3. Play the generated CSV in the OrcaLab/OrcaGym G1 scene.

CALLING_DIR="$(pwd -P)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
ORCALAB_ENV="${ORCALAB_ENV:-orcalab}"
PLAYBACK_ENV="${PLAYBACK_ENV:-orcalab}"
ORCAGYM_ADDR="${ORCAGYM_ADDR:-127.0.0.1:50051}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/generated}"
OUTPUT_NAME="${OUTPUT_NAME:-g1_motion}"
START_ORCALAB="${START_ORCALAB:-1}"

case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$CALLING_DIR/$OUTPUT_DIR" ;;
esac

host="${ORCAGYM_ADDR%:*}"
port="${ORCAGYM_ADDR##*:}"

is_port_open() {
  timeout 1 bash -c ":</dev/tcp/${host}/${port}" 2>/dev/null
}

wait_for_orcagym() {
  for _ in $(seq 1 120); do
    if is_port_open; then
      return 0
    fi
    echo "Waiting for OrcaGym server at ${ORCAGYM_ADDR} ..."
    sleep 1
  done
  echo "Timed out waiting for OrcaGym server at ${ORCAGYM_ADDR}" >&2
  return 1
}

cleanup() {
  if [[ "${ORCALAB_PID:-}" != "" ]]; then
    if kill -0 "$ORCALAB_PID" 2>/dev/null; then
      echo "OrcaLab is still running as PID $ORCALAB_PID."
      echo "Close it from the UI, or run: kill $ORCALAB_PID"
    fi
  fi
}
trap cleanup EXIT

cd "$PROJECT_ROOT"

echo "==> Generating CSV with Kimodo"
OUTPUT_DIR="$OUTPUT_DIR" OUTPUT_NAME="$OUTPUT_NAME" \
  "$SCRIPT_DIR/generate_g1_csv_with_kimodo.sh"

if ! is_port_open; then
  if [[ "$START_ORCALAB" != "1" ]]; then
    echo "OrcaGym server is not running and START_ORCALAB=0." >&2
    exit 1
  fi

  echo "==> Starting OrcaLab"
  ORCALAB_ENV="$ORCALAB_ENV" PROJECT_ROOT="$PROJECT_ROOT" \
    "$SCRIPT_DIR/start_orcalab_server.sh" &
  ORCALAB_PID=$!
else
  ORCALAB_PID=""
  echo "==> OrcaGym server already running at ${ORCAGYM_ADDR}"
fi

wait_for_orcagym

echo "==> Playing generated CSV"
CSV_PATH="$OUTPUT_DIR/$OUTPUT_NAME.csv" \
OUTPUT_DIR="$OUTPUT_DIR" \
OUTPUT_NAME="$OUTPUT_NAME" \
PROJECT_ROOT="$PROJECT_ROOT" \
PLAYBACK_ENV="$PLAYBACK_ENV" \
ORCAGYM_ADDR="$ORCAGYM_ADDR" \
"$SCRIPT_DIR/run_g1_csv_playback.sh"
