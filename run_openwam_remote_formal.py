#!/usr/bin/env python3
"""Run RoboTwin locally against persistent OpenWAM servers on a remote GPU host."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from run_experiment3 import CONDITIONS, ROOT, load_json, load_tasks, resolve_config_paths, validate
from run_openwam_formal import assign_tasks, load_step_limits, parse_gpus, summarize


def wait_for_endpoint(
    host: str,
    port: int,
    processes: list[subprocess.Popen],
    timeout: int,
    *,
    robotwin_python: str,
    openwam_repo: str,
    readiness_token: str,
) -> None:
    deadline = time.monotonic() + timeout
    probe = """
import sys
from benchmarks.utils.transport import WSPolicyClient
with WSPolicyClient(sys.argv[1], timeout=5, open_timeout=2) as client:
    pong = client.ping()
if pong.get('type') != 'pong' or pong.get('readiness_token') != sys.argv[2]:
    raise SystemExit(2)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [openwam_repo, str(Path(openwam_repo) / "benchmarks/robotwin"), env.get("PYTHONPATH", "")]
    )
    while time.monotonic() < deadline:
        failed = [process for process in processes if process.poll() is not None]
        if failed:
            raise RuntimeError(f"remote service or tunnel exited early with code {failed[0].returncode}")
        result = subprocess.run(
            [robotwin_python, "-c", probe, f"ws://{host}:{port}", readiness_token],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    raise TimeoutError(f"OpenWAM endpoint {host}:{port} was not ready within {timeout}s")


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def ssh_control_args(control_path: str | None) -> list[str]:
    return [] if control_path is None else ["-o", f"ControlPath={control_path}"]


SSH_KEEPALIVE_ARGS = [
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
]


def stop_remote_services(
    remote_host: str, readiness_token: str, *, control_path: str | None = None
) -> None:
    """Terminate deploy processes that outlive their controlling SSH connection."""
    pattern = f"[{readiness_token[0]}]{readiness_token[1:]}"
    subprocess.run(
        [
            "ssh", *ssh_control_args(control_path), remote_host,
            f"pkill -TERM -f -- {shlex.quote(pattern)} || true",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def remote_command(
    remote_host: str, command: list[str], *, control_path: str | None = None
) -> list[str]:
    return ["ssh", *ssh_control_args(control_path), remote_host, shlex.join(command)]


def copy_existing_parts(
    run_root: Path, worker: int, split: str, tasks: list[str], method: str,
    *, state: int | None = None, resume_roots: list[Path] | None = None,
    rollouts_per_state: int | None = None,
) -> int:
    """Reuse immutable part files across worker plans and prior run directories."""
    copied = 0
    source_roots = [run_root, *(resume_roots or [])]
    for task in tasks:
        for mode in CONDITIONS.values():
            destination = (
                run_root / f"worker_{worker:02d}" / split / "openwam" / method / split
                / "raw" / task / mode / "parts"
            )
            pattern = f"s{state:03d}_r*.json" if state is not None else "*.json"
            for source_root in source_roots:
                sources = source_root.glob(
                    f"worker_*/{split}/openwam/{method}/{split}/raw/"
                    f"{task}/{mode}/parts/{pattern}"
                )
                for source in sources:
                    target = destination / source.name
                    if target.exists() or source == target:
                        continue
                    try:
                        payload = json.loads(source.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    rollout_index = payload.get("rollout_index")
                    if (
                        rollouts_per_state is not None
                        and isinstance(rollout_index, int)
                        and rollout_index >= rollouts_per_state
                    ):
                        continue
                    if not isinstance(payload.get("episodes"), list):
                        continue
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    copied += 1
    return copied


def write_worker_config(
    base: dict[str, Any], path: Path, *, sim_gpu: int, port: int, worker: int,
    episodes: int, rollouts_per_state: int
) -> None:
    config = deepcopy(base)
    config["label"] = f"{base['label']}-remote-worker-{worker:02d}"
    config["protocol"]["episodes"] = episodes
    config["protocol"]["rollouts_per_state"] = rollouts_per_state
    config["hardware"].update(
        openwam_host="127.0.0.1",
        openwam_port=port,
        openwam_sim_gpu=sim_gpu,
        openwam_manage_server=False,
        openwam_session_isolation=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_for_tasks(tasks: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split, task_file in (("seen", ROOT / "tasks/seen_40.txt"), ("unseen", ROOT / "tasks/unseen_10.txt")):
        members = set(load_tasks(task_file))
        selected = [task for task in tasks if task in members]
        if selected:
            result[split] = selected
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.openwam.formal.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-from", type=Path, action="append", default=[],
        help="Prior run root to search for reusable per-rollout results (repeatable)",
    )
    parser.add_argument("--remote-host", required=True, help="SSH host or config alias")
    parser.add_argument("--remote-openwam-repo", required=True, help="OpenWAM checkout on the remote host")
    parser.add_argument("--remote-python", required=True, help="Remote Python with the OpenWAM environment")
    parser.add_argument("--remote-no-wm-checkpoint", required=True)
    parser.add_argument("--remote-wm-checkpoint", required=True)
    parser.add_argument(
        "--remote-text-embedding-cache",
        default=(
            "/home/data/tmp/hanwen.cui/openwam/robotwin_text_embeddings/"
            "wan22_ti2v_5b_umt5_xxl_bf16_clean50"
        ),
        help="Remote precomputed UMT5 cache directory",
    )
    parser.add_argument(
        "--no-text-embedding-cache",
        action="store_true",
        help="Load Wan T5 instead of using the configured precomputed embedding cache",
    )
    parser.add_argument("--remote-gpus", type=parse_gpus, default=parse_gpus("0-7"))
    parser.add_argument("--sim-gpus", type=parse_gpus, default=parse_gpus("0-7"))
    parser.add_argument("--servers-per-gpu", type=int, default=1)
    parser.add_argument("--clients-per-server", type=int, default=4)
    parser.add_argument("--remote-port-base", type=int, default=8848)
    parser.add_argument("--local-port-base", type=int, default=9848)
    parser.add_argument("--methods", choices=("no_wm", "wm", "all"), default="all")
    parser.add_argument(
        "--condition", choices=("clean", "randomized", "all"), default="all"
    )
    parser.add_argument("--tasks", help="Optional comma-separated task subset for smoke tests")
    parser.add_argument("--episodes", type=int)
    parser.add_argument(
        "--rollouts-per-state", type=int,
        help="Repeat each fixed manifest state this many times with distinct policy seeds",
    )
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--no-sync-server", action="store_true")
    parser.add_argument(
        "--dynamic-pool", action="store_true",
        help="Dynamically assign task/condition jobs to simulation workers",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(args.remote_gpus) != len(args.sim_gpus):
        parser.error("--remote-gpus and --sim-gpus must have equal lengths")
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.rollouts_per_state is not None and args.rollouts_per_state <= 0:
        parser.error("--rollouts-per-state must be positive")
    if args.clients_per_server <= 0:
        parser.error("--clients-per-server must be positive")
    if args.servers_per_gpu <= 0:
        parser.error("--servers-per-gpu must be positive")

    config_path = args.config.resolve()
    base = resolve_config_paths(load_json(config_path), config_path)
    validate(base)
    episodes = args.episodes or int(base["protocol"]["episodes"])
    rollouts_per_state = args.rollouts_per_state or int(
        base["protocol"].get("rollouts_per_state", 1)
    )
    all_tasks = load_tasks(ROOT / "tasks/all_50.txt")
    tasks = all_tasks
    if args.tasks:
        tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
        invalid = sorted(set(tasks) - set(all_tasks))
        if invalid:
            parser.error(f"unknown task(s): {invalid}")
    methods = ["no_wm", "wm"] if args.methods == "all" else [args.methods]

    limits = load_step_limits(Path(base["paths"]["robotwin_repo"]))
    server_count = len(args.sim_gpus) * args.servers_per_gpu
    worker_count = server_count * args.clients_per_server
    assignments = assign_tasks(tasks, limits, worker_count)
    run_root = args.run_dir.resolve()
    resume_roots = [path.resolve() for path in args.resume_from]
    missing_resume_roots = [path for path in resume_roots if not path.is_dir()]
    if missing_resume_roots:
        parser.error(f"resume root(s) do not exist: {missing_resume_roots}")
    config_root = run_root / "remote_configs"
    log_root = run_root / "remote_logs"
    config_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    servers = []
    for gpu_slot in range(args.servers_per_gpu):
        for gpu_index, (remote_gpu, sim_gpu) in enumerate(zip(args.remote_gpus, args.sim_gpus)):
            index = gpu_slot * len(args.remote_gpus) + gpu_index
            servers.append({
                "server": index,
                "gpu_slot": gpu_slot,
                "remote_gpu": remote_gpu,
                "sim_gpu": sim_gpu,
                "remote_port": args.remote_port_base + index,
                "local_port": args.local_port_base + index,
            })

    plan = []
    for index, assigned in enumerate(assignments):
        # Stripe client layers across all servers so the weighted assignment's
        # longest jobs do not cluster on the same model/simulation GPU.
        server_index = index % len(servers)
        client_index = index // len(servers)
        server = servers[server_index]
        port = server["local_port"]
        config_path_out = config_root / f"worker_{index:02d}.json"
        write_worker_config(
            base, config_path_out, sim_gpu=server["sim_gpu"], port=port,
            worker=index, episodes=episodes, rollouts_per_state=rollouts_per_state,
        )
        plan.append({
            "worker": index,
            "server": server_index,
            "client": client_index,
            "remote_gpu": server["remote_gpu"],
            "sim_gpu": server["sim_gpu"],
            "remote_port": server["remote_port"],
            "local_port": port,
            "step_weight": sum(limits.get(task, 1000) for task in assigned),
            "tasks": assigned,
        })
    (run_root / "remote_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[remote-formal] servers={len(servers)} servers_per_gpu={args.servers_per_gpu} "
        f"clients_per_server={args.clients_per_server} "
        f"workers={len(plan)} tasks={len(tasks)} states={episodes} "
        f"rollouts_per_state={rollouts_per_state} "
        f"methods={','.join(methods)} run={run_root}", flush=True,
    )
    if args.dry_run:
        return 0

    if not args.no_sync_server:
        local_openwam = Path(base["paths"]["openwam_repo"])
        runtime_files = (
            "openwam/deploy/server.py",
            "openwam/deploy/model_loader.py",
            "openwam/model/video_backbone/wan_backbone.py",
            "openwam/model/video_backbone/wan/text_embedding_cache.py",
        )
        for relative_path in runtime_files:
            subprocess.run(
                [
                    "rsync", "-a", "-e", "ssh",
                    str(local_openwam / relative_path),
                    f"{args.remote_host}:{args.remote_openwam_repo}/{relative_path}",
                ],
                check=True,
            )

    control_stem = uuid.uuid4().hex[:12]
    # OpenSSH servers commonly cap multiplexed sessions at 10. Keep each
    # master below that limit when multiple model replicas share one GPU host.
    control_groups = (len(servers) + 7) // 8
    service_control_paths = [
        f"/tmp/exp3-svc-{control_stem}-{index}" for index in range(control_groups)
    ]
    tunnel_control_paths = [
        f"/tmp/exp3-tun-{control_stem}-{index}" for index in range(control_groups)
    ]
    all_control_paths = service_control_paths + tunnel_control_paths
    for control_path in all_control_paths:
        subprocess.run(
            [
                "ssh", "-fN",
                "-o", "ControlMaster=yes",
                "-o", "ControlPersist=60",
                *SSH_KEEPALIVE_ARGS,
                "-o", f"ControlPath={control_path}",
                args.remote_host,
            ],
            check=True,
        )

    def close_control_master() -> None:
        for control_path in all_control_paths:
            subprocess.run(
                ["ssh", "-o", f"ControlPath={control_path}", "-O", "exit", args.remote_host],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    atexit.register(close_control_master)

    checkpoints = {
        "no_wm": args.remote_no_wm_checkpoint,
        "wm": args.remote_wm_checkpoint,
    }
    token = f"exp3-remote-{uuid.uuid4().hex}"
    interrupted = False

    def handle_signal(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    for method in methods:
        services: list[subprocess.Popen] = []
        tunnels: list[subprocess.Popen] = []
        logs = []
        workers: list[tuple[int, subprocess.Popen, Any]] = []
        try:
            for item in servers:
                control_group = item["server"] // 8
                server_log = (log_root / f"{method}_server_{item['server']:02d}.log").open("a")
                logs.append(server_log)
                command = [
                    "env", f"CUDA_VISIBLE_DEVICES={item['remote_gpu']}",
                    f"OPENWAM_TEXT_EMBEDDING_CACHE_DIR={args.remote_text_embedding_cache}",
                    f"OPENWAM_DISABLE_TEXT_EMBEDDING_CACHE={int(args.no_text_embedding_cache)}",
                    args.remote_python,
                    f"{args.remote_openwam_repo}/scripts/deploy.py",
                    "--ckpt-dir", checkpoints[method],
                    "--device", "cuda:0",
                    "--host", "127.0.0.1",
                    "--port", str(item["remote_port"]),
                    "--compile-enabled", "false",
                    "--session-isolation",
                    "--readiness-token", token,
                ]
                service = subprocess.Popen(
                    remote_command(
                        args.remote_host, command,
                        control_path=service_control_paths[control_group],
                    ),
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                services.append(service)
                tunnel_log = (log_root / f"{method}_tunnel_{item['server']:02d}.log").open("a")
                logs.append(tunnel_log)
                tunnel = subprocess.Popen(
                    [
                        # Keep the forwarding client alive so its process state is a
                        # reliable health signal. A multiplexed `ssh -N` can install
                        # the forwarding on the master and then exit successfully.
                        "ssh", *SSH_KEEPALIVE_ARGS, "-N",
                        "-o", "ExitOnForwardFailure=yes",
                        "-L", f"127.0.0.1:{item['local_port']}:127.0.0.1:{item['remote_port']}",
                        args.remote_host,
                    ],
                    stdout=tunnel_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                tunnels.append(tunnel)

            for item in servers:
                wait_for_endpoint(
                    "127.0.0.1",
                    item["local_port"],
                    services + tunnels,
                    args.startup_timeout,
                    robotwin_python=base["paths"]["robotwin_python"],
                    openwam_repo=base["paths"]["openwam_repo"],
                    readiness_token=token,
                )
            print(f"[remote-formal] {method}: all endpoints ready", flush=True)

            for split in ("seen", "unseen"):
                workers = []
                if args.dynamic_pool:
                    split_tasks = split_for_tasks(tasks).get(split, [])
                    conditions = list(CONDITIONS) if args.condition == "all" else [args.condition]
                    state_sharded = rollouts_per_state > 1
                    pending = sorted(
                        [
                            (task, condition, state)
                            for task in split_tasks
                            for condition in conditions
                            for state in (range(episodes) if state_sharded else [None])
                        ],
                        key=lambda job: (-limits.get(job[0], 1000), job[0], job[1]),
                    )
                    available = list(plan)
                    active: dict[int, tuple[subprocess.Popen, str, str, int | None]] = {}
                    worker_logs: dict[int, Any] = {}

                    def launch_job(
                        item: dict[str, Any], task: str, condition: str, state: int | None
                    ) -> None:
                        worker = item["worker"]
                        copied = copy_existing_parts(
                            run_root, worker, split, [task], method, state=state,
                            resume_roots=resume_roots,
                            rollouts_per_state=rollouts_per_state,
                        )
                        if copied:
                            print(
                                f"[remote-formal] W{worker:02d} reused {copied} existing part(s)",
                                flush=True,
                            )
                        if worker not in worker_logs:
                            worker_log = (
                                log_root / f"{method}_worker_{worker:02d}_{split}.log"
                            ).open("a")
                            worker_logs[worker] = worker_log
                            logs.append(worker_log)
                        command = [
                            sys.executable, str(ROOT / "run_experiment3.py"), "run",
                            "--config", str(config_root / f"worker_{worker:02d}.json"),
                            "--family", "openwam",
                            "--split", split,
                            "--tasks", task,
                            "--condition", condition,
                            "--method", method,
                            "--run-dir", str(run_root / f"worker_{worker:02d}" / split),
                        ]
                        process = subprocess.Popen(
                            command,
                            cwd=str(ROOT),
                            stdout=worker_logs[worker],
                            stderr=subprocess.STDOUT,
                            text=True,
                            start_new_session=True,
                            env=(
                                os.environ.copy()
                                if state is None
                                else {**os.environ, "EXP3_STATE_INDEX": str(state)}
                            ),
                        )
                        workers.append((worker, process, worker_logs[worker]))
                        active[worker] = (process, task, condition, state)
                        state_label = "" if state is None else f"/state={state}"
                        print(
                            f"[remote-pool] {method}/{split} W{worker:02d} START "
                            f"{task}/{condition}{state_label} pending={len(pending)}",
                            flush=True,
                        )

                    last_heartbeat = 0.0
                    while pending or active:
                        if interrupted:
                            raise KeyboardInterrupt
                        failed_service = next(
                            (p for p in services + tunnels if p.poll() is not None), None
                        )
                        if failed_service is not None:
                            raise RuntimeError(
                                f"{method} remote service/tunnel exited with "
                                f"{failed_service.returncode}"
                            )

                        finished = []
                        for worker, (process, task, condition, state) in active.items():
                            returncode = process.poll()
                            if returncode is None:
                                continue
                            state_label = "" if state is None else f"/state={state}"
                            if returncode:
                                raise RuntimeError(
                                    f"{method}/{split} W{worker:02d} failed "
                                    f"{task}/{condition}{state_label} with {returncode}"
                                )
                            print(
                                f"[remote-pool] {method}/{split} W{worker:02d} DONE "
                                f"{task}/{condition}{state_label}",
                                flush=True,
                            )
                            finished.append(worker)
                        for worker in finished:
                            del active[worker]
                            available.append(plan[worker])

                        available.sort(key=lambda item: item["worker"])
                        while pending and available:
                            item = available.pop(0)
                            task, condition, state = pending.pop(0)
                            launch_job(item, task, condition, state)

                        now = time.monotonic()
                        if now - last_heartbeat >= 60:
                            last_heartbeat = now
                            running = {
                                worker: (
                                    f"{task}/{condition}"
                                    + ("" if state is None else f"/state={state}")
                                )
                                for worker, (_, task, condition, state) in active.items()
                            }
                            print(
                                f"[remote-pool] {method}/{split} active={running} "
                                f"pending={len(pending)}",
                                flush=True,
                            )
                        time.sleep(1)
                    continue

                for item in plan:
                    selected = split_for_tasks(item["tasks"]).get(split, [])
                    if not selected:
                        continue
                    copied = copy_existing_parts(
                        run_root, item["worker"], split, selected, method,
                        resume_roots=resume_roots,
                        rollouts_per_state=rollouts_per_state,
                    )
                    if copied:
                        print(
                            f"[remote-formal] W{item['worker']:02d} reused {copied} existing part(s)",
                            flush=True,
                        )
                    worker_log = (
                        log_root / f"{method}_worker_{item['worker']:02d}_{split}.log"
                    ).open("a")
                    logs.append(worker_log)
                    command = [
                        sys.executable, str(ROOT / "run_experiment3.py"), "run",
                        "--config", str(config_root / f"worker_{item['worker']:02d}.json"),
                        "--family", "openwam",
                        "--split", split,
                        "--tasks", ",".join(selected),
                        "--condition", args.condition,
                        "--method", method,
                        "--run-dir", str(run_root / f"worker_{item['worker']:02d}" / split),
                    ]
                    process = subprocess.Popen(
                        command,
                        cwd=str(ROOT),
                        stdout=worker_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                    )
                    workers.append((item["worker"], process, worker_log))

                last_heartbeat = 0.0
                while any(process.poll() is None for _, process, _ in workers):
                    if interrupted:
                        raise KeyboardInterrupt
                    failed_service = next((p for p in services + tunnels if p.poll() is not None), None)
                    if failed_service is not None:
                        raise RuntimeError(
                            f"{method} remote service/tunnel exited with {failed_service.returncode}"
                        )
                    now = time.monotonic()
                    if now - last_heartbeat >= 60:
                        last_heartbeat = now
                        active = [worker for worker, process, _ in workers if process.poll() is None]
                        print(
                            f"[remote-formal] {method}/{split}: active workers={active}", flush=True
                        )
                    time.sleep(2)
                failed = {
                    worker: process.returncode
                    for worker, process, _ in workers
                    if process.returncode
                }
                if failed:
                    raise RuntimeError(f"{method}/{split} worker failures: {failed}")
            print(f"[remote-formal] {method}: complete", flush=True)
        finally:
            for _, process, _ in workers:
                stop_process(process)
            for process in tunnels:
                stop_process(process)
            for process in services:
                stop_process(process)
            stop_remote_services(
                args.remote_host, token, control_path=service_control_paths[0]
            )
            for handle in logs:
                handle.close()

    if (
        tasks == all_tasks
        and episodes == int(base["protocol"]["episodes"])
        and rollouts_per_state == 1
        and args.condition == "all"
        and methods == ["no_wm", "wm"]
    ):
        summarize(run_root, all_tasks, episodes)
    print("[remote-formal] finished", flush=True)
    close_control_master()
    atexit.unregister(close_control_master)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
