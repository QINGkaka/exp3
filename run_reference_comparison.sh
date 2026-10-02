#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BASE_CONFIG="${BASE_CONFIG:-config.clean40.smoke.json}"
RUN_TAG="${RUN_TAG:-reference_comparison_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$ROOT_DIR/runs/$RUN_TAG"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-2048}"

# Each task stays on one worker for both checkpoints and all requested conditions.
# Format: split:tasks:condition:episodes
WORKER_A_JOBS=(
  "seen:place_fan,place_a2b_right,put_bottles_dustbin,place_mouse_pad,place_dual_shoes:clean:8"
)
WORKER_B_JOBS=(
  "unseen:adjust_bottle,grab_roller,place_container_plate,move_pillbottle_pad,move_can_pot:all:4"
)
WORKER_C_JOBS=(
  "unseen:open_microwave,press_stapler,stack_bowls_three,handover_block,turn_switch:all:4"
)

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/configs"
START_EPOCH="$(date +%s)"

format_seconds() {
  local total="$1"
  printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

make_config() {
  local output="$1" episodes="$2" model_gpu="$3" sim_gpu="$4" port="$5" worker="$6"
  python - "$BASE_CONFIG" "$output" "$episodes" "$model_gpu" "$sim_gpu" "$port" "$worker" "$ROOT_DIR" <<'PY'
import json
import sys
from pathlib import Path

source, target = Path(sys.argv[1]), Path(sys.argv[2])
episodes, model_gpu, sim_gpu, port = map(int, sys.argv[3:7])
worker, root = sys.argv[7], Path(sys.argv[8])
config = json.loads(source.read_text(encoding="utf-8"))
config["label"] = f"openwam-reference-comparison-{worker}-{episodes}episodes"
config["smoke_only"] = True
config["protocol"]["episodes"] = episodes
config["paths"]["manifest_root"] = str(root / "manifests" / f"{episodes}_episodes")
config["hardware"]["openwam_model_gpu"] = model_gpu
config["hardware"]["openwam_sim_gpu"] = sim_gpu
config["hardware"]["openwam_port"] = port
config["hardware"]["openwam_startup_timeout_seconds"] = 1200
target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

check_preflight() {
  [[ -f "$BASE_CONFIG" ]] || { echo "[reference] missing config: $BASE_CONFIG" >&2; return 1; }
  command -v nvidia-smi >/dev/null || { echo "[reference] nvidia-smi not found" >&2; return 1; }

  local gpu_count busy
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  (( gpu_count >= 7 )) || {
    echo "[reference] GPU indexes 1-6 are required; detected only $gpu_count GPUs" >&2
    return 1
  }

  busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v limit="$MAX_GPU_USED_MIB" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if (($1 + 0) >= 1 && ($1 + 0) <= 6 && ($2 + 0) > (limit + 0)) print "GPU " $1 ": " $2 " MiB"}')"
  if [[ -n "$busy" && "$ALLOW_BUSY_GPUS" != "1" ]]; then
    echo "[reference] refusing to start because GPUs are already occupied:" >&2
    echo "$busy" >&2
    echo "[reference] stop those jobs, or set ALLOW_BUSY_GPUS=1 only if sharing is intentional" >&2
    return 1
  fi

  if ss -ltn | awk '{print $4}' | grep -Eq '(^|:)(8848|8849|8850)$'; then
    echo "[reference] one of ports 8848-8850 is already in use" >&2
    return 1
  fi

  python run_experiment3.py validate --config "$BASE_CONFIG"
}

run_worker() {
  local worker="$1" model_gpu="$2" sim_gpu="$3" port="$4"
  shift 4
  local jobs=("$@")
  local total="${#jobs[@]}" index=0 worker_start job split tasks condition episodes
  local task_start task_end config run_name
  worker_start="$(date +%s)"

  for job in "${jobs[@]}"; do
    IFS=: read -r split tasks condition episodes <<<"$job"
    index=$((index + 1))
    task_start="$(date +%s)"
    config="$RUN_ROOT/configs/${worker}_${episodes}episodes.json"
    make_config "$config" "$episodes" "$model_gpu" "$sim_gpu" "$port" "$worker"
    run_name="${split}_${condition}_${episodes}episodes"
    echo "[$worker] START batch=$index/$total split=$split condition=$condition episodes=$episodes tasks=$tasks at=$(date -Is)"

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
    echo "[$worker] DONE batch=$index/$total elapsed=$(format_seconds "$((task_end - task_start))") total=$(format_seconds "$((task_end - worker_start))") at=$(date -Is)"
  done
  echo "[$worker] COMPLETE elapsed=$(format_seconds "$(($(date +%s) - worker_start))") at=$(date -Is)"
}

heartbeat() {
  while kill -0 "$PID_A" 2>/dev/null || kill -0 "$PID_B" 2>/dev/null \
    || kill -0 "$PID_C" 2>/dev/null; do
    sleep 60
    local now gpu_line
    now="$(date +%s)"
    gpu_line="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i 1,2,3,4,5,6 | tr '\n' ';')"
    echo "[reference] HEARTBEAT elapsed=$(format_seconds "$((now - START_EPOCH))") GPUs(index,MiB,util%)=$gpu_line"
  done
}

terminate_tree() {
  local parent="$1" child
  while read -r child; do
    [[ -n "$child" ]] && terminate_tree "$child"
  done < <(pgrep -P "$parent" 2>/dev/null || true)
  kill -TERM "$parent" 2>/dev/null || true
}

interrupt_workers() {
  echo "[reference] interrupt received; stopping all workers and OpenWAM servers" >&2
  for pid in "$PID_A" "$PID_B" "$PID_C"; do terminate_tree "$pid"; done
  wait "$PID_A" "$PID_B" "$PID_C" 2>/dev/null || true
  exit 130
}

write_summary() {
  python - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seen = ["place_fan", "place_a2b_right", "put_bottles_dustbin", "place_mouse_pad", "place_dual_shoes"]
unseen = ["adjust_bottle", "grab_roller", "place_container_plate", "move_pillbottle_pad", "move_can_pot", "open_microwave", "press_stapler", "stack_bowls_three", "handover_block", "turn_switch"]
labels = {task: task.replace("_", " ").title() for task in seen + unseen}
values = {}
for path in root.glob("worker_*/*/openwam/*/*/raw/*/*/result.json"):
    data = json.loads(path.read_text(encoding="utf-8"))
    parts = path.parts
    openwam = parts.index("openwam")
    method, split = parts[openwam + 1], parts[openwam + 2]
    condition = "clean" if data["mode"] == "demo_clean" else "randomized"
    values[(split, data["task"], condition, method)] = f'{data["successes"]}/{data["total_episodes"]}'

lines = [
    "# OpenWAM Reference Comparison", "",
    "| Split | Task | Clean No-WM | Clean WM | Randomized No-WM | Randomized WM |",
    "|---|---|---:|---:|---:|---:|",
]
for split, tasks in (("seen", seen), ("unseen", unseen)):
    for task in tasks:
        cells = [values.get((split, task, cond, method), "-") for cond, method in (
            ("clean", "no_wm"), ("clean", "wm"), ("randomized", "no_wm"), ("randomized", "wm"))]
        lines.append(f'| {split.title()} | {labels[task]} | ' + " | ".join(cells) + " |")
(root / "reference_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[reference] comparison table: {root / 'reference_comparison.md'}")
PY
}

check_preflight

echo "[reference] start=$(date -Is) total_rollouts=240"
echo "[reference] seen: 5 tasks x 2 methods x 8 episodes = 80"
echo "[reference] unseen: 10 tasks x 2 conditions x 2 methods x 4 episodes = 160"
echo "[reference] GPU pairs: A=4+1 B=5+2 C=6+3; GPU 0 excluded; ports=8848-8850"
echo "[reference] results=$RUN_ROOT"

run_worker worker_a 4 1 8848 "${WORKER_A_JOBS[@]}" > >(awk '!/OIDN Error:/ {print "[A] " $0; fflush()}' | tee "$RUN_ROOT/logs/worker_a.log") 2>&1 & PID_A=$!
run_worker worker_b 5 2 8849 "${WORKER_B_JOBS[@]}" > >(awk '!/OIDN Error:/ {print "[B] " $0; fflush()}' | tee "$RUN_ROOT/logs/worker_b.log") 2>&1 & PID_B=$!
run_worker worker_c 6 3 8850 "${WORKER_C_JOBS[@]}" > >(awk '!/OIDN Error:/ {print "[C] " $0; fflush()}' | tee "$RUN_ROOT/logs/worker_c.log") 2>&1 & PID_C=$!

trap interrupt_workers INT TERM
heartbeat & HEARTBEAT_PID=$!

set +e
wait "$PID_A"; STATUS_A=$?
wait "$PID_B"; STATUS_B=$?
wait "$PID_C"; STATUS_C=$?
kill "$HEARTBEAT_PID" 2>/dev/null
wait "$HEARTBEAT_PID" 2>/dev/null
set -e

END_EPOCH="$(date +%s)"
echo "[reference] finished=$(date -Is) elapsed=$(format_seconds "$((END_EPOCH - START_EPOCH))") status=$STATUS_A,$STATUS_B,$STATUS_C"
echo "[reference] logs=$RUN_ROOT/logs"

if (( STATUS_A != 0 || STATUS_B != 0 || STATUS_C != 0 )); then
  echo "[reference] FAILED: completed outputs are retained; inspect worker logs" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" ]]; then
  write_summary
fi
