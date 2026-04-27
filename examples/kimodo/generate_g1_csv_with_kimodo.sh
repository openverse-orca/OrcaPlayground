#!/usr/bin/env bash
set -euo pipefail

# Generate a G1 MuJoCo qpos CSV from a text prompt using the Kimodo project.
# This script lives in OrcaPlayground so the whole G1 playback workflow is
# managed from one place.

KIMODO_ENV="${KIMODO_ENV:-kimodo}"
KIMODO_ROOT="${KIMODO_ROOT:-/home/user/下载/kimodo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/generated}"
PROMPT="${PROMPT:-A humanoid robot walks forward and pick up something.}"
OUTPUT_NAME="${OUTPUT_NAME:-g1_motion}"
MODEL="${MODEL:-g1}"
DURATION="${DURATION:-5.0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-}"
LOCAL_CACHE="${LOCAL_CACHE:-True}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$KIMODO_ENV"

mkdir -p "$OUTPUT_DIR"
cd "$KIMODO_ROOT"

# httpx/huggingface_hub rejects socks:// in ALL_PROXY. Keep HTTP(S)_PROXY if
# configured, but remove the ambiguous all-proxy variables for this process.
case "${ALL_PROXY:-}" in
  socks://*) unset ALL_PROXY ;;
esac
case "${all_proxy:-}" in
  socks://*) unset all_proxy ;;
esac

export LOCAL_CACHE

output_stem="$OUTPUT_DIR/$OUTPUT_NAME"
cmd=(
  kimodo_gen
  "$PROMPT"
  --model "$MODEL"
  --duration "$DURATION"
  --diffusion_steps "$DIFFUSION_STEPS"
  --num_samples "$NUM_SAMPLES"
  --output "$output_stem"
)

if [[ -n "$SEED" ]]; then
  cmd+=(--seed "$SEED")
fi

echo "Kimodo root: $KIMODO_ROOT"
echo "Prompt: $PROMPT"
echo "Output stem: $output_stem"
"${cmd[@]}"

csv_path="$output_stem.csv"
if [[ ! -f "$csv_path" ]]; then
  echo "Expected CSV not found: $csv_path" >&2
  exit 1
fi

echo "Generated CSV: $csv_path"
 
