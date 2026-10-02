#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/openwam_formal_100}"
mkdir -p "$RUN_DIR"

export CONFIG="${CONFIG:-$ROOT_DIR/config.openwam.formal.json}"
export RUN_DIR
export MODEL_GPUS="${MODEL_GPUS:-3,4,6}"
export SIM_GPUS="${SIM_GPUS:-1,2,5}"
export PORT_BASE="${PORT_BASE:-8848}"

cd "$ROOT_DIR"
./run_openwam_formal.sh "$@" 2>&1 | tee -a "$RUN_DIR/launcher.log"
