#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python - <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch", "safetensors") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing required dependencies: "
        + ", ".join(missing)
        + ". Install them before running this script."
    )
PY

BASE="${BASE:-dapo32b/base}"
FULL="${FULL:-dapo32b/full}"
OUT="${OUT:-dapo32b/approx_top1}"
KEEP_RATIO="${KEEP_RATIO:-0.01}"
DEVICE="${DEVICE:-cuda:0}"
SVD_OVERSAMPLE="${SVD_OVERSAMPLE:-16}"
SVD_NITER="${SVD_NITER:-2}"
SEED="${SEED:-0}"

python alpharl/checkpoint_reconstruction.py \
  --base-model-path "$BASE" \
  --trained-model-path "$FULL" \
  --output-path "$OUT" \
  --keep-ratio "$KEEP_RATIO" \
  --device "$DEVICE" \
  --svd-oversample "$SVD_OVERSAMPLE" \
  --svd-niter "$SVD_NITER" \
  --seed "$SEED"
