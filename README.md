# World Model Action Evaluation

Independent evaluation harness for the RoboTwin experiments. Model repositories
remain external dependencies; this repository owns the protocol, task splits,
episode manifests, scheduling, provenance, and result aggregation.

## Layout

```text
wm_eval/runtime.py             shared execution and result contract
wm_eval/adapters/fastwam.py    FastWAM in-process adapter
wm_eval/adapters/openwam.py    OpenWAM WebSocket adapter
tasks/                         fixed 40/10 split
patches/fastwam-eval.patch     minimal FastWAM evaluation-interface patch
run_experiment3.py             Experiment 3 protocol and reporting
```

FastWAM and OpenWAM are not compared as if they were the same backbone. For
each family, the formal comparison is `no_wm` versus `wm` under an otherwise
identical evaluator and deployment.

## Upstream setup

The verified revisions are:

| Repository | Commit |
|---|---|
| FastWAM | `7faa71108368fbb3b6885649f112af607427a2d4` |
| OpenWAM | `f6d9f1059eb63a8a76dc60e07da2dad21615a161` |
| RoboTwin for OpenWAM | `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4` |

OpenWAM is used without source changes. FastWAM needs a small evaluator patch;
it does not alter model construction, forward passes, action sampling, or
checkpoint loading:

```bash
git -C /path/to/FastWAM apply /path/to/experiment3/patches/fastwam-eval.patch
python check_upstreams.py --config config.smoke.json
```

`check_upstreams.py` checks all three revisions, verifies that OpenWAM and the
official RoboTwin checkout have no tracked modifications, and confirms that
the FastWAM patch is applied.

## Protocol

The harness builds sealed episode manifests with the verified official
RoboTwin checkout. Every adapter receives the same initial-state seed and
instruction for a logical episode. OpenWAM consumes the manifest through its
official `labtasker run_eval` interface; FastWAM consumes it through the small
patch above.

Experiment 3 evaluates:

```text
Seen + Clean
Seen + Randomized
Unseen + Clean
Unseen + Randomized
```

Each cell is run for both `no_wm` and `wm`. The runner records repository
revisions, resolved configuration, checkpoint paths, hardware assignment, raw
results, per-task rates, aggregate rates, and WM absolute/relative gains.

## Configuration

Paths may be absolute or relative to the JSON configuration file. Conda names
are configuration values rather than hard-coded paths:

```json
"runtimes": {
  "fastwam": {"conda_env": "fastwam"},
  "openwam": {"conda_env": "openwam"}
}
```

An explicit command can be used instead:

```json
"runtimes": {
  "fastwam": {"command": ["/path/to/fastwam/bin/python"]}
}
```

Create `config.smoke.json` from `config.smoke.example.json`, or create
`config.formal.json` from `config.formal.template.json`. Fill in the RoboTwin
Python path and checkpoint entries. Both machine-local files are ignored by
Git.

## Commands

```bash
python run_experiment3.py validate --config config.smoke.json
python run_experiment3.py run --config config.smoke.json --split unseen --dry-run
python run_experiment3.py run --config config.formal.json
python run_experiment3.py summarize --run-dir runs/<run-id>
```

The released-checkpoint smoke run only verifies the pipeline. It is not formal
OOD evidence because those checkpoints were not trained with this 40/10 split.

## Adding Another Model

Add one adapter under `wm_eval/adapters/` and register it in
`wm_eval/adapters/__init__.py`. An adapter receives the common `EvalContext` and
implements:

```python
run(method, model, splits, run_root, dry_run)
```

It must consume the sealed manifests and emit the shared `result.json` schema
through `context.write_result(...)`. No Experiment 3 scheduling or reporting
code needs to be copied.

The same runtime and adapter layer can support Experiment 1. That experiment
needs a separate runner because it repeats one fixed initial state 32 times and
reports per-state probabilities instead of four aggregate cells.

## Git Hygiene

The `.gitignore` excludes runs, manifests, formal local configuration,
checkpoints, videos, and datasets. Do not commit upstream repositories, conda
environments, cuRobo source copies, or symlinks into local asset trees.
