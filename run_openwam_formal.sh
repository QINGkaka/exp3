#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-$ROOT_DIR/config.openwam.formal.json}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/openwam_formal_100}"
MODEL_GPUS="${MODEL_GPUS:-20-39}"
SIM_GPUS="${SIM_GPUS:-0-19}"
PORT_BASE="${PORT_BASE:-8848}"
MAX_USED_MEMORY_MIB="${MAX_USED_MEMORY_MIB:-2048}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing $CONFIG" >&2
  echo "Create it first: cp config.openwam.formal.template.json config.openwam.formal.json" >&2
  exit 2
fi

exec python "$ROOT_DIR/run_openwam_formal.py" \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --model-gpus "$MODEL_GPUS" \
  --sim-gpus "$SIM_GPUS" \
  --port-base "$PORT_BASE" \
  --max-used-memory-mib "$MAX_USED_MEMORY_MIB" \
  "$@"
