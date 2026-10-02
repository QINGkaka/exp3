#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

EPISODES="${EPISODES:-4}"
TASKS=(adjust_bottle grab_roller place_container_plate)
BASE_CONFIG="${BASE_CONFIG:-config.clean40.smoke.json}"
RUN_TAG="medium_$(date +%Y%m%d_%H%M%S)"
CONFIG_PATH="runs/${RUN_TAG}.config.json"
START_EPOCH="$(date +%s)"

cleanup() {
  rm -f "$CONFIG_PATH"
}
trap cleanup EXIT

python - "$BASE_CONFIG" "$CONFIG_PATH" "$EPISODES" <<'PY'
import json
import sys
from pathlib import Path

source, target, episodes = sys.argv[1], sys.argv[2], int(sys.argv[3])
config = json.loads(Path(source).read_text())
config["label"] = f"openwam-medium-{episodes}episodes"
config["protocol"]["episodes"] = episodes
config["paths"]["manifest_root"] = str(Path("manifests") / f"{episodes}_episodes")
Path(target).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
PY

echo "[medium] start=$(date -Is) episodes_per_task=$EPISODES tasks=${TASKS[*]}"
echo "[medium] total planned episodes=$(( ${#TASKS[@]} * EPISODES * 2 * 2 ))"

for task in "${TASKS[@]}"; do
  for method in no_wm wm; do
    before="$(date +%s)"
    echo "[medium] START task=$task method=$method at=$(date -Is)"
    python run_experiment3.py run \
      --config "$CONFIG_PATH" \
      --split unseen \
      --task "$task" \
      --method "$method" \
      --run-dir "runs/${RUN_TAG}/${task}"
    after="$(date +%s)"
    echo "[medium] DONE task=$task method=$method elapsed=$((after - before))s at=$(date -Is)"
  done
done

END_EPOCH="$(date +%s)"
echo "[medium] finished=$(date -Is) total_elapsed=$((END_EPOCH - START_EPOCH))s"
echo "[medium] results=runs/${RUN_TAG}"
