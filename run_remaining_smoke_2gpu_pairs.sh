#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

EPISODES="${EPISODES:-1}"
BASE_CONFIG="${BASE_CONFIG:-config.clean40.smoke.json}"
RUN_TAG="${RUN_TAG:-remaining_smoke_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$ROOT_DIR/runs/$RUN_TAG"
DRY_RUN="${DRY_RUN:-0}"

# Keep each task and all of its missing comparisons on one worker. The longer
# tasks are distributed so the two workers should finish at roughly similar times.
WORKER_A_JOBS=(
  "seen:put_bottles_dustbin,place_fan,place_a2b_right:randomized"
  "unseen:stack_bowls_three,open_microwave,move_can_pot:all"
)
WORKER_B_JOBS=(
  "seen:place_dual_shoes,place_mouse_pad:randomized"
  "unseen:move_pillbottle_pad,press_stapler,handover_block,turn_switch:all"
)

mkdir -p "$RUN_ROOT/logs"
START_EPOCH="$(date +%s)"

format_seconds() {
  local total="$1"
  printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

make_config() {
  local output="$1" model_gpu="$2" sim_gpu="$3" port="$4" worker="$5"
  python - "$BASE_CONFIG" "$output" "$EPISODES" "$model_gpu" "$sim_gpu" "$port" "$worker" <<'PY'
import json
import sys
from pathlib import Path

source, target = Path(sys.argv[1]), Path(sys.argv[2])
episodes, model_gpu, sim_gpu, port = map(int, sys.argv[3:7])
worker = sys.argv[7]
config = json.loads(source.read_text(encoding="utf-8"))
config["label"] = f"openwam-remaining-smoke-{worker}-{episodes}episode"
config["smoke_only"] = True
config["protocol"]["episodes"] = episodes
config["paths"]["manifest_root"] = str(
    Path(__file__).resolve().parent / "manifests" / f"{episodes}_episodes"
)
config["hardware"]["openwam_model_gpu"] = model_gpu
config["hardware"]["openwam_sim_gpu"] = sim_gpu
config["hardware"]["openwam_port"] = port
config["hardware"]["openwam_startup_timeout_seconds"] = 1200
target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

check_preflight() {
  [[ -f "$BASE_CONFIG" ]] || { echo "[smoke] missing config: $BASE_CONFIG" >&2; return 1; }
  command -v nvidia-smi >/dev/null || { echo "[smoke] nvidia-smi not found" >&2; return 1; }

  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  (( gpu_count >= 7 )) || {
    echo "[smoke] expected GPU indexes 0,1,4,6; only $gpu_count GPUs detected" >&2
    return 1
  }

  if command -v ss >/dev/null; then
    if ss -ltn | awk '{print $4}' | grep -Eq '(^|:)(8848|8849)$'; then
      echo "[smoke] port 8848 or 8849 is already in use; stop the old OpenWAM server first" >&2
      return 1
    fi
  fi
}

run_worker() {
  local worker="$1" config="$2"; shift 2
  local jobs=("$@")
  local total="${#jobs[@]}" index=0 worker_start now job split tasks condition task_start task_end run_name
  worker_start="$(date +%s)"

  for job in "${jobs[@]}"; do
    IFS=: read -r split tasks condition <<<"$job"
    index=$((index + 1))
    task_start="$(date +%s)"
    run_name="${split}_${condition}"
    echo "[$worker] START batch=$index/$total split=$split tasks=$tasks condition=$condition at=$(date -Is)"

    command=(python run_experiment3.py run
      --config "$config"
      --family openwam
      --split "$split"
      --tasks "$tasks"
      --condition "$condition"
      --method all
      --run-dir "$RUN_ROOT/$worker/$run_name")
    [[ "$DRY_RUN" == "1" ]] && command+=(--dry-run)
    "${command[@]}"

    task_end="$(date +%s)"
    echo "[$worker] DONE  batch=$index/$total tasks=$tasks batch_elapsed=$(format_seconds "$((task_end - task_start))") worker_elapsed=$(format_seconds "$((task_end - worker_start))") at=$(date -Is)"
  done

  now="$(date +%s)"
  echo "[$worker] COMPLETE jobs=$total elapsed=$(format_seconds "$((now - worker_start))") at=$(date -Is)"
}

heartbeat() {
  while kill -0 "$PID_A" 2>/dev/null || kill -0 "$PID_B" 2>/dev/null; do
    sleep 60
    local now gpu_line
    now="$(date +%s)"
    gpu_line="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i 0,1,4,6 | tr '\n' ';')"
    echo "[smoke] HEARTBEAT elapsed=$(format_seconds "$((now - START_EPOCH))") GPUs(index,MiB,util%)=$gpu_line"
  done
}

terminate_tree() {
  local parent="$1" child
  while read -r child; do
    [[ -n "$child" ]] || continue
    terminate_tree "$child"
  done < <(pgrep -P "$parent" 2>/dev/null || true)
  kill -TERM "$parent" 2>/dev/null || true
}

interrupt_workers() {
  echo "[smoke] interrupt received; stopping workers and their OpenWAM servers" >&2
  terminate_tree "$PID_A"
  terminate_tree "$PID_B"
  wait "$PID_A" "$PID_B" 2>/dev/null || true
  exit 130
}

check_preflight
make_config "$RUN_ROOT/worker_a.config.json" 4 0 8848 worker_a
make_config "$RUN_ROOT/worker_b.config.json" 6 1 8849 worker_b

echo "[smoke] start=$(date -Is) episodes_per_combination=$EPISODES"
echo "[smoke] missing combinations: seen-randomized=10 unseen-clean/randomized=28 total_rollouts=$((38 * EPISODES))"
echo "[smoke] worker_a model_gpu=4 sim_gpu=0 port=8848 jobs=${#WORKER_A_JOBS[@]}"
echo "[smoke] worker_b model_gpu=6 sim_gpu=1 port=8849 jobs=${#WORKER_B_JOBS[@]}"
echo "[smoke] results=$RUN_ROOT"

run_worker worker_a "$RUN_ROOT/worker_a.config.json" "${WORKER_A_JOBS[@]}" \
  > >(awk '{ print "[A] " $0; fflush() }' | tee "$RUN_ROOT/logs/worker_a.log") 2>&1 &
PID_A=$!
run_worker worker_b "$RUN_ROOT/worker_b.config.json" "${WORKER_B_JOBS[@]}" \
  > >(awk '{ print "[B] " $0; fflush() }' | tee "$RUN_ROOT/logs/worker_b.log") 2>&1 &
PID_B=$!

trap interrupt_workers INT TERM
heartbeat &
HEARTBEAT_PID=$!

set +e
wait "$PID_A"; STATUS_A=$?
wait "$PID_B"; STATUS_B=$?
kill "$HEARTBEAT_PID" 2>/dev/null
wait "$HEARTBEAT_PID" 2>/dev/null
set -e

END_EPOCH="$(date +%s)"
echo "[smoke] finished=$(date -Is) elapsed=$(format_seconds "$((END_EPOCH - START_EPOCH))") worker_a_status=$STATUS_A worker_b_status=$STATUS_B"
echo "[smoke] logs=$RUN_ROOT/logs"
echo "[smoke] results=$RUN_ROOT"

if (( STATUS_A != 0 || STATUS_B != 0 )); then
  echo "[smoke] FAILED: inspect worker logs above; completed task directories are retained" >&2
  exit 1
fi
