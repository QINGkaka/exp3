from __future__ import annotations

import os
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
        server_dir.mkdir(parents=True, exist_ok=True)
        if dry_run:
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
            )
            try:
                _wait_for_port(host, port, server)
            except Exception:
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
                        command = [
                            config["paths"]["robotwin_python"],
                            "benchmarks/robotwin/eval_policy_wrapper.py", "labtasker",
                            "--operation", "run_eval",
                            "--task", task,
                            "--mode", mode,
                            "--policy-config", str(repo / "benchmarks/robotwin/policy_config.yml"),
                            "--host", host,
                            "--port", str(port),
                            "--result-file", str(result_path),
                            "--progress-file", str(raw_dir / "progress.json"),
                            "--total-episodes", str(config["protocol"]["episodes"]),
                            "--episode-start", "0",
                            "--num-episodes", str(config["protocol"]["episodes"]),
                            "--manifest-file", str(manifest_path),
                            "--manifest-hash", manifest_hash,
                            "--instruction-type", str(config["protocol"]["instruction_type"]),
                        ]
                        run_logged(
                            command,
                            cwd=repo,
                            env=client_env,
                            log_path=raw_dir / "driver.log",
                            dry_run=dry_run,
                        )
                        if not dry_run:
                            result = load_json(result_path)
                            rows[task][condition] = int(result["successes"]) / int(result["num_episodes"])
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
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            if server_log is not None:
                server_log.close()
