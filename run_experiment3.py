#!/usr/bin/env python3
"""Run and summarize Experiment 3 without changing model implementations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from wm_eval.adapters import ADAPTERS
from wm_eval.runtime import EvalContext, load_json, load_tasks, run_logged


ROOT = Path(__file__).resolve().parent
TASK_DIR = ROOT / "tasks"
RUNS_DIR = ROOT / "runs"
SPLITS = {"seen": TASK_DIR / "seen_40.txt", "unseen": TASK_DIR / "unseen_10.txt"}
CONDITIONS = {"clean": "demo_clean", "randomized": "demo_randomized"}


def resolve_config_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    base = config_path.parent

    def resolve(value: str) -> str:
        if value == "REPLACE_ME":
            return value
        path = Path(os.path.expandvars(os.path.expanduser(value)))
        return str(path if path.is_absolute() else (base / path).resolve())

    for key, value in config.get("paths", {}).items():
        if value:
            config["paths"][key] = resolve(value)
    for family, methods in config.get("models", {}).items():
        path_keys = ("checkpoint", "dataset_stats") if family == "fastwam" else ("checkpoint_dir",)
        for model in methods.values():
            for key in path_keys:
                if model.get(key):
                    model[key] = resolve(model[key])
    return config


def git_revision(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    return {
        "path": str(repo),
        "commit": git("rev-parse", "HEAD") or None,
        "modified_tracked_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
    }


def validate(config: dict[str, Any]) -> None:
    all_tasks = load_tasks(TASK_DIR / "all_50.txt")
    seen = load_tasks(SPLITS["seen"])
    unseen = load_tasks(SPLITS["unseen"])
    errors: list[str] = []
    if len(all_tasks) != 50 or len(set(all_tasks)) != 50:
        errors.append(f"all_50.txt must contain 50 unique tasks, got {len(set(all_tasks))}")
    if len(seen) != 40 or len(set(seen)) != 40:
        errors.append(f"seen_40.txt must contain 40 unique tasks, got {len(set(seen))}")
    if len(unseen) != 10 or len(set(unseen)) != 10:
        errors.append(f"unseen_10.txt must contain 10 unique tasks, got {len(set(unseen))}")
    overlap = sorted(set(seen) & set(unseen))
    if overlap:
        errors.append(f"seen/unseen overlap: {overlap}")
    if set(seen) | set(unseen) != set(all_tasks):
        errors.append("seen_40 + unseen_10 does not exactly equal all_50")

    required_paths = ["robotwin_repo", "robotwin_python"]
    if "fastwam" in config.get("models", {}):
        required_paths.append("fastwam_repo")
    if "openwam" in config.get("models", {}):
        required_paths.append("openwam_repo")
    for key in required_paths:
        value = Path(config["paths"][key])
        if not value.exists():
            errors.append(f"paths.{key} does not exist: {value}")

    for family, methods in config.get("models", {}).items():
        if family not in ADAPTERS:
            errors.append(f"unsupported model family: {family}")
        for method, model in methods.items():
            keys = ("checkpoint", "dataset_stats") if family == "fastwam" else ("checkpoint_dir",)
            for key in keys:
                value = model.get(key)
                if not value or value == "REPLACE_ME" or not Path(value).exists():
                    errors.append(f"models.{family}.{method}.{key} is not ready: {value}")
    if errors:
        raise ValueError("Invalid Experiment 3 configuration:\n- " + "\n- ".join(errors))


def ensure_manifests(
    config: dict[str, Any], splits: list[str], run_root: Path, context: EvalContext
) -> Path:
    manifest_root = Path(config["paths"]["manifest_root"]).resolve()
    episodes = int(config["protocol"]["episodes"])
    openwam_repo = Path(config["paths"]["openwam_repo"])
    robotwin_repo = Path(config["paths"]["robotwin_repo"])
    robotwin_python = Path(config["paths"]["robotwin_python"])
    wrapper = openwam_repo / "benchmarks/robotwin/eval_policy_wrapper.py"
    policy_config = openwam_repo / "benchmarks/robotwin/policy_config.yml"
    env = context.environment()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(config["hardware"]["openwam_sim_gpu"]),
        "ROBOTWIN_PATH": str(robotwin_repo),
        "PYTHONPATH": os.pathsep.join([
            str(openwam_repo), str(openwam_repo / "benchmarks/robotwin"),
            env.get("PYTHONPATH", ""),
        ]),
    })

    for split in splits:
        for task in context.tasks(split):
            for mode in context.conditions.values():
                task_dir = manifest_root / task / mode
                manifest_path = task_dir / "manifest.json"
                if manifest_path.is_file():
                    manifest = load_json(manifest_path)
                    if (
                        manifest.get("task") == task
                        and manifest.get("mode") == mode
                        and len(manifest.get("entries", [])) >= episodes
                    ):
                        print(f"[experiment3] reuse manifest: {task}/{mode}", flush=True)
                        continue

                runtime_root = task_dir / "runtime"
                runtime_root.mkdir(parents=True, exist_ok=True)
                build_env = env.copy()
                build_env["ROBOTWIN_RUNTIME_ROOT"] = str(runtime_root)
                command = [
                    str(robotwin_python), str(wrapper), "labtasker",
                    "--operation", "build_manifest",
                    "--task", task,
                    "--mode", mode,
                    "--policy-config", str(policy_config),
                    "--result-file", str(manifest_path),
                    "--progress-file", str(task_dir / "progress.json"),
                    "--total-episodes", str(episodes),
                ]
                run_logged(
                    command,
                    cwd=openwam_repo,
                    env=build_env,
                    log_path=run_root / "manifests" / f"{task}_{mode}.log",
                    dry_run=False,
                )
    return manifest_root


def summarize(run_root: Path) -> None:
    results = [load_json(path) for path in sorted(run_root.glob("*/*/*/result.json"))]
    if not results:
        raise FileNotFoundError(f"No result.json files found under {run_root}")
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str, str], float] = {}
    for result in results:
        for condition in result.get("means", {}):
            key = (result["family"], result["method"], result["split"], condition)
            lookup[key] = float(result["means"][condition])
            rows.append({
                "family": result["family"], "method": result["method"],
                "split": result["split"], "condition": condition,
                "success_rate": lookup[key],
            })
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "success_rates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    deltas: list[dict[str, Any]] = []
    families = sorted({row["family"] for row in rows})
    for family in families:
        for split in SPLITS:
            for condition in CONDITIONS:
                no_wm = lookup.get((family, "no_wm", split, condition))
                wm = lookup.get((family, "wm", split, condition))
                if no_wm is None or wm is None:
                    continue
                deltas.append({
                    "family": family, "split": split, "condition": condition,
                    "no_wm": no_wm, "wm": wm, "wm_minus_no_wm": wm - no_wm,
                    "relative_improvement": None if no_wm == 0 else (wm - no_wm) / no_wm,
                })
    if deltas:
        with (summary_dir / "wm_improvements.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=deltas[0].keys())
            writer.writeheader()
            writer.writerows(deltas)

    lines = ["# Experiment 3 Summary", "", "Success rates are reported as percentages.", ""]
    for family in families:
        methods = sorted({row["method"] for row in rows if row["family"] == family})
        lines.extend([f"## {family}", "", "| Method | Setting | Success rate |", "|---|---|---:|"])
        for method in methods:
            for split, condition in (("seen", "clean"), ("seen", "randomized"), ("unseen", "clean"), ("unseen", "randomized")):
                value = lookup.get((family, method, split, condition))
                if value is not None:
                    lines.append(f"| {method} | {split.title()} + {condition.title()} | {100 * value:.2f}% |")
        lines.append("")
    if not deltas:
        lines.append("WM improvement is unavailable because this run does not contain both `no_wm` and `wm` checkpoints.")
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[experiment3] summary: {summary_dir / 'summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "run", "summarize"))
    parser.add_argument("--config", type=Path, default=ROOT / "config.smoke.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--family", choices=("all", "fastwam", "openwam"), default="all")
    parser.add_argument("--method", default="all")
    parser.add_argument("--split", choices=("all", "seen", "unseen"), default="all")
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--task", help="Run one task only (for a paired smoke rollout)")
    task_group.add_argument("--tasks", help="Run a comma-separated subset of tasks")
    parser.add_argument("--condition", choices=("all", "clean", "randomized"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.command == "summarize":
        if args.run_dir is None:
            parser.error("summarize requires --run-dir")
        summarize(args.run_dir.resolve())
        return

    config_path = args.config.resolve()
    config = resolve_config_paths(load_json(config_path), config_path)
    validate(config)
    print("[experiment3] configuration and 40/10 task split are valid")
    if args.command == "validate":
        return

    run_root = args.run_dir.resolve() if args.run_dir else (
        RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    families = list(config["models"]) if args.family == "all" else [args.family]
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    repositories = [git_revision(Path(config["paths"]["robotwin_repo"]))]
    for family in families:
        repositories.append(git_revision(Path(config["paths"][f"{family}_repo"])))
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "label": config["label"], "smoke_only": config["smoke_only"],
        "protocol": config["protocol"], "hardware": config["hardware"],
        "task_files": {key: str(value) for key, value in SPLITS.items()},
        "repositories": repositories,
    }
    (run_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    splits = list(SPLITS) if args.split == "all" else [args.split]
    conditions = CONDITIONS if args.condition == "all" else {args.condition: CONDITIONS[args.condition]}
    task_filter = None
    if args.task:
        task_filter = (args.task,)
    elif args.tasks:
        task_filter = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
        if not task_filter:
            parser.error("--tasks must contain at least one task")
        if len(set(task_filter)) != len(task_filter):
            parser.error("--tasks must not contain duplicates")
    context = EvalContext(config=config, splits=SPLITS, conditions=conditions, task_filter=task_filter)
    if not args.dry_run:
        ensure_manifests(config, splits, run_root, context)
    for family in families:
        adapter = ADAPTERS[family](context)
        methods = config["models"][family]
        selected_methods = list(methods) if args.method == "all" else [args.method]
        for method in selected_methods:
            if method not in methods:
                raise KeyError(f"Method {method!r} is not configured for {family}")
            adapter.run(
                method=method,
                model=methods[method],
                splits=splits,
                run_root=run_root,
                dry_run=args.dry_run,
            )
    if not args.dry_run:
        summarize(run_root)
    print(f"[experiment3] run directory: {run_root}")


if __name__ == "__main__":
    main()
