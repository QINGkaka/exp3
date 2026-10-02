#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from run_experiment3 import load_json, resolve_config_paths


EXPECTED = {
    "fastwam_repo": "7faa71108368fbb3b6885649f112af607427a2d4",
    "openwam_repo": "f6d9f1059eb63a8a76dc60e07da2dad21615a161",
    "robotwin_repo": "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4",
}
FASTWAM_PATCHED_FILES = {
    "configs/sim_robotwin.yaml",
    "experiments/robotwin/run_robotwin_manager.py",
    "third_party/RoboTwin/script/eval_policy.py",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.smoke.json"))
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = resolve_config_paths(load_json(config_path), config_path)
    snapshot_file = Path(__file__).with_name("upstream_revisions.json")
    snapshot_revisions = json.loads(snapshot_file.read_text()) if snapshot_file.is_file() else {}

    failed = False
    for key, expected in EXPECTED.items():
        repo = Path(config["paths"][key])
        actual = git(repo, "rev-parse", "HEAD")
        source = "git"
        if not actual:
            actual = snapshot_revisions.get(key, "")
            source = "snapshot manifest"
        dirty = git(repo, "status", "--porcelain", "--untracked-files=no").splitlines()
        state = "OK" if actual == expected else "DIFFERS"
        print(
            f"{key}: {state} expected={expected[:12]} "
            f"actual={actual[:12] or 'missing'} source={source}"
        )
        if key == "fastwam_repo":
            dirty_paths = {line.split(maxsplit=1)[-1] for line in dirty}
            unexpected = sorted(dirty_paths - FASTWAM_PATCHED_FILES)
            if unexpected:
                print(f"  unexpected tracked modifications: {', '.join(unexpected)}")
                failed = True
        elif dirty:
            print(f"  tracked modifications: {len(dirty)}")
            failed = True
        if actual != expected:
            failed = True

    fastwam = Path(config["paths"]["fastwam_repo"])
    patch = Path(__file__).with_name("patches") / "fastwam-eval.patch"
    reverse_check = subprocess.run(
        ["git", "-C", str(fastwam), "apply", "--reverse", "--check", str(patch)],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"fastwam_eval_patch: {'APPLIED' if reverse_check.returncode == 0 else 'MISSING_OR_DIFFERENT'}")
    if reverse_check.returncode != 0:
        failed = True
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
