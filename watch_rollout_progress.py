#!/usr/bin/env python3
"""Display aggregate rollout progress for a sharded Experiment 3 run."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "--"
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


def result_key(path: Path) -> str:
    parts = path.parts
    try:
        family_index = parts.index("openwam")
        method = parts[family_index + 1]
        split = parts[family_index + 2]
        condition = parts[-2].removeprefix("demo_")
        return f"{method}/{split}/{condition}"
    except (ValueError, IndexError):
        return "other"


def result_identity(path: Path) -> tuple[str, str, str, str]:
    parts = path.parts
    family_index = parts.index("openwam")
    return (
        parts[family_index + 1],
        parts[family_index + 2],
        parts[-3],
        parts[-2].removeprefix("demo_"),
    )


def read_progress(
    run_dir: Path, *, active_since: float | None = None
) -> tuple[int, dict[str, int], Path | None, float | None]:
    completed = 0
    breakdown: dict[str, int] = defaultdict(int)
    latest: Path | None = None
    latest_mtime = -1.0

    committed_ranges: dict[tuple[str, str, str, str], set[tuple[int, int]]] = defaultdict(set)
    committed_parts: set[tuple[str, str, str, str, int, int]] = set()
    for path in run_dir.glob("worker_*/**/parts/*.json"):
        try:
            data = json.loads(path.read_text())
            count = int(data.get("num_episodes", 0))
            start = int(data.get("episode_start", 0))
            mtime = path.stat().st_mtime
            identity = result_identity(path.parent.parent / "progress.json")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        part_identity = (*identity, start, start + count)
        if part_identity in committed_parts:
            continue
        committed_parts.add(part_identity)
        completed += count
        breakdown[result_key(path.parent.parent / "progress.json")] += count
        committed_ranges[identity].add((start, start + count))
        if mtime > latest_mtime:
            latest, latest_mtime = path, mtime

    live_progress: dict[tuple[str, str, str, str], tuple[Path, dict, float]] = {}
    for path in run_dir.glob("worker_*/**/progress.json"):
        try:
            data = json.loads(path.read_text())
            mtime = path.stat().st_mtime
            identity = result_identity(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if active_since is not None and mtime < active_since:
            continue
        previous = live_progress.get(identity)
        if previous is None or mtime > previous[2]:
            live_progress[identity] = (path, data, mtime)

    for identity, (path, data, mtime) in live_progress.items():
        count = int(data.get("completed", 0))
        episode = int(data.get("episode", -1))
        batch_start = episode - count + 1
        if (batch_start, batch_start + count) in committed_ranges[identity]:
            count = 0
        completed += count
        breakdown[result_key(path)] += count
        if mtime > latest_mtime:
            latest, latest_mtime = path, mtime
    return completed, dict(breakdown), latest, latest_mtime if latest is not None else None


def committed_before(run_dir: Path, cutoff: float) -> int:
    committed: dict[tuple[str, str, str, str, int, int], float] = {}
    for path in run_dir.glob("worker_*/**/parts/*.json"):
        try:
            data = json.loads(path.read_text())
            count = int(data.get("num_episodes", 0))
            start = int(data.get("episode_start", 0))
            identity = result_identity(path.parent.parent / "progress.json")
            mtime = path.stat().st_mtime
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        key = (*identity, start, start + count)
        committed[key] = min(mtime, committed.get(key, mtime))
    return sum(key[-1] - key[-2] for key, mtime in committed.items() if mtime < cutoff)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()

    started = (args.run_dir / "remote_plan.json").stat().st_mtime
    baseline = committed_before(args.run_dir, started)
    while True:
        now = time.time()
        completed, breakdown, latest, latest_mtime = read_progress(
            args.run_dir, active_since=started
        )
        elapsed = max(now - started, 1.0)
        rate = max(completed - baseline, 0) / elapsed * 3600
        remaining = args.total - completed
        eta = remaining / rate * 3600 if rate > 0 else float("inf")
        percent = completed / args.total * 100 if args.total else 100.0

        print("\033[2J\033[H", end="")
        print(time.strftime("Experiment 3 progress  %F %T"))
        print(f"Run:       {args.run_dir.resolve()}")
        print(f"Completed: {completed}/{args.total} ({percent:.2f}%)")
        print(f"Elapsed:   {format_duration(elapsed)}")
        print(f"Rate:      {rate:.2f} rollouts/hour")
        print(f"ETA:       {format_duration(eta)}")
        if latest is not None:
            print(f"Latest:    {latest.parent}")
        if latest_mtime is not None:
            print(f"Activity:  {format_duration(now - latest_mtime)} ago")
        print("\nBreakdown:")
        for key, value in sorted(breakdown.items()):
            print(f"  {key:<28} {value}")
        print(f"\nRefreshing every {args.interval:g}s; Ctrl-C stops only this monitor.")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
