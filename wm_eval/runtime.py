from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalContext:
    config: dict[str, Any]
    splits: dict[str, Path]
    conditions: dict[str, str]
    task_filter: tuple[str, ...] | None = None

    def tasks(self, split: str) -> list[str]:
        tasks = load_tasks(self.splits[split])
        if self.task_filter is not None:
            invalid = [task for task in self.task_filter if task not in tasks]
            if invalid:
                raise ValueError(f"Tasks {invalid!r} are not in {split} split")
            return list(self.task_filter)
        return tasks

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        robotwin_python = self.config.get("paths", {}).get("robotwin_python")
        if robotwin_python:
            robotwin_prefix = Path(robotwin_python).resolve().parent.parent
            robotwin_bin = robotwin_prefix / "bin"
            current_path = env.get("PATH", "")
            env["PATH"] = str(robotwin_bin) + (os.pathsep + current_path if current_path else "")
            cuda_target = robotwin_prefix / "targets" / "x86_64-linux"
            if cuda_target.is_dir():
                for key, path in (
                    ("CPATH", cuda_target / "include"),
                    ("LIBRARY_PATH", cuda_target / "lib"),
                ):
                    current = env.get(key, "")
                    env[key] = str(path) + (os.pathsep + current if current else "")
            env.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        matched_libs = self.config["paths"].get("matched_nvidia_libs")
        if matched_libs:
            current = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = str(matched_libs) + (f":{current}" if current else "")
        vulkan_icd = self.config["paths"].get("vulkan_icd")
        if vulkan_icd:
            env["VK_ICD_FILENAMES"] = str(vulkan_icd)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def write_result(
        self,
        output_dir: Path,
        *,
        family: str,
        method: str,
        split: str,
        checkpoint: str,
        task_rows: list[dict[str, Any]],
    ) -> None:
        expected = self.tasks(split)
        by_task: dict[str, dict[str, Any]] = {}
        for row in task_rows:
            by_task.setdefault(row["task"], {}).update(row)
        missing = [task for task in expected if task not in by_task]
        if missing:
            raise RuntimeError(f"Missing parsed results for {family}/{method}/{split}: {missing}")

        rows = [by_task[task] for task in expected]
        means = {
            condition: sum(float(row[condition]) for row in rows) / len(rows)
            for condition in self.conditions
        }
        payload = {
            "family": family,
            "method": method,
            "split": split,
            "checkpoint": checkpoint,
            "task_file": str(self.splits[split]),
            "num_tasks": len(rows),
            "means": means,
            "per_task": rows,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (output_dir / "result.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["task", *self.conditions])
            writer.writeheader()
            writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tasks(path: Path) -> list[str]:
    tasks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        task = line.split("#", 1)[0].strip()
        if task:
            tasks.append(task)
    return tasks


def run_logged(
    command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, dry_run: bool
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(command)
    print(f"[wm-eval] cwd={cwd}\n[wm-eval] command={rendered}", flush=True)
    if dry_run:
        log_path.write_text(rendered + "\n", encoding="utf-8")
        return
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}; see {log_path}")


def python_command(config: dict[str, Any], family: str) -> list[str]:
    runtime = config.get("runtimes", {}).get(family, {})
    command = runtime.get("command")
    if command:
        if not isinstance(command, list) or not all(isinstance(part, str) and part for part in command):
            raise ValueError(f"runtimes.{family}.command must be a non-empty string list")
        return list(command)
    env_name = runtime.get("conda_env", family)
    conda = runtime.get("conda_executable", "conda")
    return [str(conda), "run", "--no-capture-output", "-n", str(env_name), "python"]
