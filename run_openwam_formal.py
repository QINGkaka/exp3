#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from run_experiment3 import CONDITIONS, ROOT, SPLITS, load_json, load_tasks, resolve_config_paths, validate


def parse_gpus(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(item))
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise argparse.ArgumentTypeError("GPU list must contain unique non-negative indexes")
    return result


def load_step_limits(robotwin_repo: Path) -> dict[str, int]:
    path = robotwin_repo / "task_config/_eval_step_limit.yml"
    limits: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        task, value = line.split(":", 1)
        limits[task.strip()] = int(value.strip())
    return limits


def assign_tasks(tasks: list[str], limits: dict[str, int], workers: int) -> list[list[str]]:
    assignments: list[list[str]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for task in sorted(tasks, key=lambda name: (-limits.get(name, 1000), name)):
        worker = min(range(workers), key=lambda index: (loads[index], index))
        assignments[worker].append(task)
        loads[worker] += limits.get(task, 1000)
    return assignments


def port_available(host: str, port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def check_gpus(requested: list[int], max_used_memory_mib: int, allow_busy: bool) -> None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot query GPUs with nvidia-smi: {exc}") from exc
    usage = {}
    for line in output.splitlines():
        index, used = (int(value.strip()) for value in line.split(",", 1))
        usage[index] = used
    missing = sorted(set(requested) - set(usage))
    if missing:
        raise RuntimeError(f"requested GPU indexes do not exist: {missing}")
    busy = {index: usage[index] for index in requested if usage[index] > max_used_memory_mib}
    if busy and not allow_busy:
        detail = ", ".join(f"GPU {index}: {used} MiB" for index, used in busy.items())
        raise RuntimeError(
            f"requested GPUs are already busy ({detail}); use --allow-busy-gpus only if sharing is intentional"
        )


def write_worker_config(
    base: dict[str, Any], path: Path, model_gpu: int, sim_gpu: int, port: int, worker: int
) -> None:
    config = json.loads(json.dumps(base))
    config["label"] = f"{base['label']}-worker-{worker:02d}"
    config["hardware"].update(
        openwam_model_gpu=model_gpu,
        openwam_sim_gpu=sim_gpu,
        openwam_port=port,
    )
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pump_output(process: subprocess.Popen[str], prefix: str, log_path: Path) -> None:
    assert process.stdout is not None
    with log_path.open("a", encoding="utf-8") as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            if "OIDN Error:" not in line:
                sys.stdout.write(f"[{prefix}] {line}")
                sys.stdout.flush()


def worker_main(
    worker: int,
    config_path: Path,
    tasks: list[str],
    run_root: Path,
    log_path: Path,
    dry_run: bool,
) -> int:
    by_split = {
        split: [task for task in tasks if task in set(load_tasks(path))]
        for split, path in SPLITS.items()
    }
    for split, selected in by_split.items():
        if not selected:
            continue
        command = [
            sys.executable,
            str(ROOT / "run_experiment3.py"),
            "run",
            "--config",
            str(config_path),
            "--family",
            "openwam",
            "--split",
            split,
            "--tasks",
            ",".join(selected),
            "--condition",
            "all",
            "--method",
            "all",
            "--run-dir",
            str(run_root / f"worker_{worker:02d}" / split),
        ]
        if dry_run:
            command.append("--dry-run")
        print(f"[W{worker:02d}] START split={split} tasks={','.join(selected)}", flush=True)
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with ACTIVE_LOCK:
            ACTIVE[worker] = process
        pump_output(process, f"W{worker:02d}", log_path)
        status = process.wait()
        with ACTIVE_LOCK:
            ACTIVE.pop(worker, None)
        if status:
            print(f"[W{worker:02d}] FAILED split={split} status={status}", flush=True)
            return status
        print(f"[W{worker:02d}] DONE split={split}", flush=True)
    return 0


def summarize(run_root: Path, expected_tasks: list[str], episodes: int) -> None:
    values: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for path in run_root.glob("worker_*/*/openwam/*/*/raw/*/*/result.json"):
        result = load_json(path)
        parts = path.parts
        index = parts.index("openwam")
        method, split = parts[index + 1], parts[index + 2]
        condition = "clean" if result["mode"] == CONDITIONS["clean"] else "randomized"
        key = (split, result["task"], condition, method)
        values[key] = result

    expected = {
        (split, task, condition, method)
        for split, path in SPLITS.items()
        for task in load_tasks(path)
        for condition in CONDITIONS
        for method in ("no_wm", "wm")
    }
    complete = {
        key for key, value in values.items()
        if value.get("num_episodes") == episodes and len(value.get("episodes", [])) == episodes
    }
    missing = sorted(expected - complete)
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, task, condition, method in sorted(complete):
        result = values[(split, task, condition, method)]
        rows.append({
            "split": split,
            "task": task,
            "condition": condition,
            "method": method,
            "successes": result["successes"],
            "episodes": episodes,
            "success_rate": result["successes"] / episodes,
        })
    with (summary_dir / "all_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["split", "task"])
        writer.writeheader()
        writer.writerows(rows)
    status = {
        "expected_cells": len(expected),
        "complete_cells": len(complete),
        "missing_cells": ["/".join(item) for item in missing],
        "expected_rollouts": len(expected_tasks) * 2 * 2 * episodes,
    }
    (summary_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# OpenWAM Experiment 3 Formal Results",
        "",
        f"Protocol: {len(expected_tasks)} tasks, 2 conditions, 2 models, {episodes} initial states per cell.",
        "",
        "| Split | Condition | Method | Successes | Rollouts | Success rate |",
        "|---|---|---|---:|---:|---:|",
    ]
    for split in SPLITS:
        split_tasks = set(load_tasks(SPLITS[split]))
        for condition in CONDITIONS:
            for method in ("no_wm", "wm"):
                selected = [
                    row for row in rows
                    if row["split"] == split
                    and row["task"] in split_tasks
                    and row["condition"] == condition
                    and row["method"] == method
                ]
                successes = sum(int(row["successes"]) for row in selected)
                rollouts = sum(int(row["episodes"]) for row in selected)
                rate = "-" if not rollouts else f"{100 * successes / rollouts:.2f}%"
                lines.append(
                    f"| {split.title()} | {condition.title()} | {method} | "
                    f"{successes} | {rollouts} | {rate} |"
                )
    lines.extend([
        "",
        f"Complete cells: {len(complete)}/{len(expected)}.",
        "",
    ])
    (summary_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[formal] summary={summary_dir} complete={len(complete)}/{len(expected)} "
        f"rollouts_expected={status['expected_rollouts']}",
        flush=True,
    )
    if missing:
        raise RuntimeError(f"Formal run is incomplete: {len(missing)} task-condition-method cells missing")


ACTIVE: dict[int, subprocess.Popen[str]] = {}
ACTIVE_LOCK = threading.Lock()
STOP = threading.Event()


def stop_all(signum: int, _frame: Any) -> None:
    print(f"[formal] signal={signum}; stopping worker process groups", file=sys.stderr, flush=True)
    STOP.set()
    with ACTIVE_LOCK:
        processes = list(ACTIVE.values())
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.openwam.formal.json")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/openwam_formal_100")
    parser.add_argument("--model-gpus", type=parse_gpus, default=parse_gpus("20-39"))
    parser.add_argument("--sim-gpus", type=parse_gpus, default=parse_gpus("0-19"))
    parser.add_argument("--port-base", type=int, default=8848)
    parser.add_argument("--max-used-memory-mib", type=int, default=2048)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--allow-busy-gpus", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.model_gpus) != len(args.sim_gpus):
        parser.error("--model-gpus and --sim-gpus must have the same length")
    if set(args.model_gpus) & set(args.sim_gpus):
        parser.error("model and simulator GPU lists must be disjoint")

    config_path = args.config.resolve()
    base = resolve_config_paths(load_json(config_path), config_path)
    if set(base.get("models", {})) != {"openwam"}:
        parser.error("formal launcher requires an OpenWAM-only config")
    episodes = int(base["protocol"]["episodes"])
    if episodes != args.expected_episodes:
        parser.error(
            f"config has protocol.episodes={episodes}, expected {args.expected_episodes}"
        )
    validate(base)

    if not args.dry_run:
        try:
            check_gpus(
                args.model_gpus + args.sim_gpus,
                args.max_used_memory_mib,
                args.allow_busy_gpus,
            )
        except RuntimeError as exc:
            parser.error(str(exc))

    host = str(base["hardware"].get("openwam_host", "127.0.0.1"))
    ports = [args.port_base + index for index in range(len(args.model_gpus))]
    if not args.dry_run:
        occupied = [port for port in ports if not port_available(host, port)]
        if occupied:
            parser.error(f"ports already in use: {occupied}")

    all_tasks = load_tasks(ROOT / "tasks/all_50.txt")
    limits = load_step_limits(Path(base["paths"]["robotwin_repo"]))
    assignments = assign_tasks(all_tasks, limits, len(args.model_gpus))
    run_root = args.run_dir.resolve()
    (run_root / "configs").mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    plan = []
    for index, tasks in enumerate(assignments):
        config = run_root / "configs" / f"worker_{index:02d}.json"
        write_worker_config(
            base, config, args.model_gpus[index], args.sim_gpus[index], ports[index], index
        )
        plan.append({
            "worker": index,
            "model_gpu": args.model_gpus[index],
            "sim_gpu": args.sim_gpus[index],
            "port": ports[index],
            "step_weight": sum(limits.get(task, 1000) for task in tasks),
            "tasks": tasks,
        })
    (run_root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        f"[formal] start={datetime.now().astimezone().isoformat()} workers={len(plan)} "
        f"tasks=50 states={episodes} conditions=2 models=2 "
        f"rollouts={len(all_tasks) * len(CONDITIONS) * 2 * episodes} run={run_root}",
        flush=True,
    )
    for item in plan:
        print(
            f"[formal] W{item['worker']:02d} model={item['model_gpu']} sim={item['sim_gpu']} "
            f"port={item['port']} weight={item['step_weight']} tasks={','.join(item['tasks'])}",
            flush=True,
        )

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)
    statuses: dict[int, int] = {}
    threads = []
    start_monotonic = time.monotonic()
    batch_size = int(base["hardware"].get("openwam_episode_batch_size", 100))
    expected_batches = (
        len(all_tasks) * len(CONDITIONS) * 2 * math.ceil(episodes / batch_size)
    )

    def launch(index: int, tasks: list[str]) -> None:
        statuses[index] = worker_main(
            index,
            run_root / "configs" / f"worker_{index:02d}.json",
            tasks,
            run_root,
            run_root / "logs" / f"worker_{index:02d}.log",
            args.dry_run,
        )

    for index, tasks in enumerate(assignments):
        thread = threading.Thread(target=launch, args=(index, tasks), daemon=False)
        thread.start()
        threads.append(thread)
    last_heartbeat = 0.0
    while any(thread.is_alive() for thread in threads):
        for thread in threads:
            thread.join(timeout=1)
        if not any(thread.is_alive() for thread in threads):
            break
        now = time.monotonic()
        if now - last_heartbeat >= 60:
            last_heartbeat = now
            with ACTIVE_LOCK:
                active_workers = sorted(ACTIVE)
            complete_parts = sum(1 for _ in run_root.glob("worker_*/*/openwam/*/*/raw/*/*/parts/*.json"))
            print(
                f"[formal] HEARTBEAT active={active_workers} "
                f"completed_batches={complete_parts}/{expected_batches} "
                f"elapsed_seconds={int(now - start_monotonic)}",
                flush=True,
            )
        if STOP.is_set():
            break
    for thread in threads:
        thread.join()
    failed = {worker: status for worker, status in statuses.items() if status}
    if failed:
        print(f"[formal] failed workers={failed}; rerun the same command to resume", file=sys.stderr)
        return 1
    if not args.dry_run:
        summarize(run_root, all_tasks, episodes)
    print(f"[formal] finished={datetime.now().astimezone().isoformat()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
