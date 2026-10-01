from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from wm_eval.runtime import EvalContext, load_json, python_command, run_logged


def _wait_for_port(host: str, port: int, process: subprocess.Popen[str], timeout: int = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"OpenWAM server exited early with code {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"OpenWAM server did not listen on {host}:{port} within {timeout}s")


def _valid_part(
    path: Path,
    *,
    task: str,
    mode: str,
    manifest_hash: str,
    episode_start: int,
    num_episodes: int,
    total_episodes: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = load_json(path)
    except (OSError, ValueError):
        return None
    expected = {
        "operation": "run_eval",
        "task": task,
        "mode": mode,
        "manifest_hash": manifest_hash,
        "episode_start": episode_start,
        "num_episodes": num_episodes,
        "total_episodes": total_episodes,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        return None
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != num_episodes:
        return None
    if [row.get("episode") for row in episodes] != list(
        range(episode_start, episode_start + num_episodes)
    ):
        return None
    return result


class OpenWAMAdapter:
    """Run OpenWAM with its official WebSocket server/client deployment."""

    family = "openwam"

    def __init__(self, context: EvalContext) -> None:
        self.context = context

    def run(
        self,
        *,
        method: str,
        model: dict[str, Any],
        splits: list[str],
        run_root: Path,
        dry_run: bool,
    ) -> None:
        config = self.context.config
        repo = Path(config["paths"]["openwam_repo"])
        env = self.context.environment()
        host = str(config["hardware"]["openwam_host"])
        port = int(config["hardware"]["openwam_port"])
        manage_server = bool(config["hardware"].get("openwam_manage_server", True))
        server_dir = run_root / self.family / method
        server_log_path = server_dir / "server.log"
        server_command = [
            *python_command(config, self.family),
            "scripts/deploy.py",
            "--ckpt-dir", model["checkpoint_dir"],
            "--device", f"cuda:{config['hardware']['openwam_model_gpu']}",
            "--host", host,
            "--port", str(port),
            *model.get("deploy_args", []),
        ]
        if config["hardware"].get("openwam_session_isolation", False):
            server_command.append("--session-isolation")
        server_dir.mkdir(parents=True, exist_ok=True)
        if not manage_server:
            server_log_path.write_text(
                f"external OpenWAM server: ws://{host}:{port}\n", encoding="utf-8"
            )
            server = None
            server_log = None
        elif dry_run:
            server_log_path.write_text(" ".join(server_command) + "\n", encoding="utf-8")
            server = None
            server_log = None
        else:
            server_log = server_log_path.open("w", encoding="utf-8")
            server = subprocess.Popen(
                server_command,
                cwd=str(repo),
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                timeout = int(config["hardware"].get("openwam_startup_timeout_seconds", 300))
                _wait_for_port(host, port, server, timeout=timeout)
            except Exception:
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                server_log.close()
                raise

        try:
            for split in splits:
                output_dir = server_dir / split
                rows = {task: {"task": task} for task in self.context.tasks(split)}
                for condition, mode in self.context.conditions.items():
                    for task in self.context.tasks(split):
                        raw_dir = output_dir / "raw" / task / mode
                        manifest_path = Path(config["paths"]["manifest_root"]) / task / mode / "manifest.json"
                        manifest_hash = "DRY_RUN"
                        if not dry_run:
                            manifest_hash = str(load_json(manifest_path)["manifest_hash"])
                        runtime_root = raw_dir / "runtime"
                        runtime_root.mkdir(parents=True, exist_ok=True)
                        result_path = raw_dir / "result.json"
                        client_env = env.copy()
                        client_env.update({
                            "CUDA_VISIBLE_DEVICES": str(config["hardware"]["openwam_sim_gpu"]),
                            "ROBOTWIN_PATH": config["paths"]["robotwin_repo"],
                            "ROBOTWIN_RUNTIME_ROOT": str(runtime_root),
                            "PYTHONPATH": os.pathsep.join([
                                str(repo), str(repo / "benchmarks/robotwin"),
                                env.get("PYTHONPATH", ""),
                            ]),
                        })
                        robotwin_bin = str(Path(config["paths"]["robotwin_python"]).parent)
                        client_env["PATH"] = robotwin_bin + os.pathsep + client_env.get("PATH", "")
                        total_episodes = int(config["protocol"]["episodes"])
                        batch_size = int(config["hardware"].get("openwam_episode_batch_size", total_episodes))
                        if batch_size <= 0:
                            raise ValueError("hardware.openwam_episode_batch_size must be positive")
                        parts_dir = raw_dir / "parts"
                        parts_dir.mkdir(parents=True, exist_ok=True)
                        part_results: list[dict[str, Any]] = []
                        for episode_start in range(0, total_episodes, batch_size):
                            num_episodes = min(batch_size, total_episodes - episode_start)
                            part_path = parts_dir / f"{episode_start:05d}_{episode_start + num_episodes:05d}.json"
                            part = None if dry_run else _valid_part(
                                part_path,
                                task=task,
                                mode=mode,
                                manifest_hash=manifest_hash,
                                episode_start=episode_start,
                                num_episodes=num_episodes,
                                total_episodes=total_episodes,
                            )
                            if part is not None:
                                print(
                                    f"[openwam] resume: {method}/{task}/{mode} "
                                    f"episodes {episode_start}:{episode_start + num_episodes}",
                                    flush=True,
                                )
                                part_results.append(part)
                                continue
                            command = [
                                config["paths"]["robotwin_python"],
                                "benchmarks/robotwin/eval_policy_wrapper.py", "labtasker",
                                "--operation", "run_eval",
                                "--task", task,
                                "--mode", mode,
                                "--policy-config", str(repo / "benchmarks/robotwin/policy_config.yml"),
                                "--host", host,
                                "--port", str(port),
                                "--result-file", str(part_path),
                                "--progress-file", str(raw_dir / "progress.json"),
                                "--total-episodes", str(total_episodes),
                                "--episode-start", str(episode_start),
                                "--num-episodes", str(num_episodes),
                                "--manifest-file", str(manifest_path),
                                "--manifest-hash", manifest_hash,
                                "--instruction-type", str(config["protocol"]["instruction_type"]),
                            ]
                            run_logged(
                                command,
                                cwd=repo,
                                env=client_env,
                                log_path=raw_dir / f"driver_{episode_start:05d}.log",
                                dry_run=dry_run,
                            )
                            if not dry_run:
                                part = _valid_part(
                                    part_path,
                                    task=task,
                                    mode=mode,
                                    manifest_hash=manifest_hash,
                                    episode_start=episode_start,
                                    num_episodes=num_episodes,
                                    total_episodes=total_episodes,
                                )
                                if part is None:
                                    raise RuntimeError(f"Invalid OpenWAM batch result: {part_path}")
                                part_results.append(part)
                        if not dry_run:
                            episodes = [row for part in part_results for row in part["episodes"]]
                            result = {
                                "operation": "run_eval",
                                "instruction_type": str(config["protocol"]["instruction_type"]),
                                "task": task,
                                "mode": mode,
                                "episode_start": 0,
                                "num_episodes": total_episodes,
                                "total_episodes": total_episodes,
                                "manifest_hash": manifest_hash,
                                "successes": sum(bool(row["success"]) for row in episodes),
                                "episodes": episodes,
                            }
                            result_path.write_text(
                                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                            )
                            rows[task][condition] = int(result["successes"]) / total_episodes
                if not dry_run:
                    self.context.write_result(
                        output_dir,
                        family=self.family,
                        method=method,
                        split=split,
                        checkpoint=model["checkpoint_dir"],
                        task_rows=list(rows.values()),
                    )
        finally:
            if manage_server and server is not None:
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(server.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    server.wait()
            if server_log is not None:
                server_log.close()
