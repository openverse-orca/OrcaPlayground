#!/usr/bin/env bash
set -euo pipefail

# Environment that has the `orcalab` command installed.
ORCALAB_ENV="${ORCALAB_ENV:-orcalab}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ORCALAB_ENV"

cd "$PROJECT_ROOT"
exec orcalab .
