from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from wm_eval.runtime import EvalContext, load_json, python_command, run_logged


def _checkpoint_tag(path: Path) -> str:
    parts = path.resolve().parts
    if "runs" in parts:
        index = parts.index("runs")
        if index + 2 < len(parts):
            return f"{parts[index + 1]}_{parts[index + 2]}"
    return path.stem


class FastWAMAdapter:
    """Run FastWAM through its in-process RoboTwin evaluator."""

    family = "fastwam"

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
        for split in splits:
            self._run_split(method, model, split, run_root, dry_run)

    def _run_split(
        self,
        method: str,
        model: dict[str, Any],
        split: str,
        run_root: Path,
        dry_run: bool,
    ) -> None:
        config = self.context.config
        repo = Path(config["paths"]["fastwam_repo"])
        output_dir = run_root / self.family / method / split
        run_token = f"exp3_{run_root.name}_{method}_{split}"
        env = self.context.environment()
        env["FASTWAM_ROBOTWIN_MANIFEST_ROOT"] = str(Path(config["paths"]["manifest_root"]).resolve())
        env["DIFFSYNTH_MODEL_BASE_PATH"] = str(repo / "checkpoints")
        configured_gpus = config["hardware"].get(
            "fastwam_gpus", [config["hardware"].get("fastwam_gpu", 0)]
        )
        gpu_ids = [int(gpu_id) for gpu_id in configured_gpus]
        gpu_override = "[" + ",".join(str(gpu_id) for gpu_id in gpu_ids) + "]"
        command = [
            *python_command(config, self.family),
            "experiments/robotwin/run_robotwin_manager.py",
            f"task={model['task_config']}",
            f"ckpt={model['checkpoint']}",
            f"EVALUATION.dataset_stats_path={model['dataset_stats']}",
            f"EVALUATION.task_file={self.context.splits[split]}",
            f"EVALUATION.eval_num_episodes={config['protocol']['episodes']}",
            f"EVALUATION.instruction_type={config['protocol']['instruction_type']}",
            f"seed={config['protocol']['seed']}",
            f"EVALUATION.output_dir={run_token}",
            f"MULTIRUN.gpu_ids={gpu_override}",
            "MULTIRUN.max_tasks_per_gpu=1",
        ]
        run_logged(command, cwd=repo, env=env, log_path=output_dir / "driver.log", dry_run=dry_run)
        if dry_run:
            return

        source = (
            repo / "evaluate_results" / "robotwin" / _checkpoint_tag(Path(model["checkpoint"]))
            / run_token / "summary.json"
        )
        raw = load_json(source)
        rows = [
            {
                "task": row["task_name"],
                "clean": row["clean_success_rate"],
                "randomized": row["random_success_rate"],
            }
            for row in raw["per_task"]
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output_dir / "fastwam_summary.raw.json")
        self.context.write_result(
            output_dir,
            family=self.family,
            method=method,
            split=split,
            checkpoint=model["checkpoint"],
            task_rows=rows,
        )

