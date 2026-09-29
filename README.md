# Experiment 3: RoboTwin Evaluation

Evaluation runner for the `Seen/Unseen x Clean/Randomized` experiment. FastWAM,
OpenWAM, and RoboTwin remain external repositories; this repository stores the
task split, shared episode manifests, model adapters, and result aggregation.

## Layout

```text
wm_eval/runtime.py             shared execution and result contract
wm_eval/adapters/fastwam.py    FastWAM in-process adapter
wm_eval/adapters/openwam.py    OpenWAM WebSocket adapter
tasks/                         fixed 40/10 split
patches/fastwam-eval.patch     minimal FastWAM evaluation-interface patch
run_experiment3.py             Experiment 3 protocol and reporting
```

The formal comparison is `no_wm` versus `wm` within each model family. FastWAM
and OpenWAM are not treated as identical backbones.

## Required Upstreams

The verified revisions are:

| Repository | Commit |
|---|---|
| FastWAM | `7faa71108368fbb3b6885649f112af607427a2d4` |
| OpenWAM | `f6d9f1059eb63a8a76dc60e07da2dad21615a161` |
| RoboTwin for OpenWAM | `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4` |

Check out these repositories next to this repository, or set their absolute
paths in the configuration. OpenWAM is used without source changes. FastWAM
needs the included evaluation patch; it does not change model inference or
checkpoint loading.

```bash
git -C /path/to/FastWAM apply /path/to/experiment3/patches/fastwam-eval.patch
python check_upstreams.py --config config.smoke.json
```

After filling in `config.smoke.json`, `check_upstreams.py` verifies the three
commits, rejects unexpected tracked changes, and confirms the FastWAM patch.

## Evaluation Protocol

RoboTwin generates a manifest containing each episode's task, mode, seed,
initial state, and instruction. Every model evaluates the same manifests.

Experiment 3 evaluates:

```text
Seen + Clean
Seen + Randomized
Unseen + Clean
Unseen + Randomized
```

Each condition is run for both `no_wm` and `wm`. The runner stores the resolved
configuration, repository commits, logs, per-task success rates, aggregate
rates, and WM improvement.

## Configuration

Create a machine-local configuration and fill in repository paths, Python or
Conda environments, GPU IDs, and checkpoint paths:

```bash
cp config.smoke.example.json config.smoke.json
# or, for the formal run:
cp config.formal.template.json config.formal.json
```

Paths may be absolute or relative to the configuration file. The two local
configuration files are ignored by Git.

## Commands

```bash
python run_experiment3.py validate --config config.smoke.json
python check_upstreams.py --config config.smoke.json
python run_experiment3.py run --config config.smoke.json --split unseen --dry-run
python run_experiment3.py run --config config.smoke.json --split unseen
python run_experiment3.py run --config config.formal.json
python run_experiment3.py summarize --run-dir runs/<run-id>
```

## One-command OpenWAM Formal Run

The formal OpenWAM protocol evaluates 50 tasks under clean and randomized
conditions, comparing No-WM and WM on 100 valid initial states per cell. Each
state receives one policy rollout, for 20,000 rollouts in total.

Configure the machine once:

```bash
cp config.openwam.formal.template.json config.openwam.formal.json
# Edit OpenWAM/RoboTwin/Python/checkpoint paths in config.openwam.formal.json.
python run_experiment3.py validate --config config.openwam.formal.json
```

On a 40-GPU machine, the default launcher creates 20 isolated workers. GPUs
0-19 run RoboTwin simulators and GPUs 20-39 host OpenWAM servers:

```bash
tmux new -s openwam-exp3
./run_openwam_formal.sh
```

Tasks are greedily balanced using RoboTwin's official per-task step limits.
Each worker gets a distinct model GPU, simulator GPU, port, config, log, and
result directory. Results are checkpointed every 10 episodes. If a process or
machine stops, run the same command again; validated batches are reused and
only missing batches are executed.

Override GPU pairs or the output directory without editing the script:

```bash
MODEL_GPUS=8-15 SIM_GPUS=0-7 RUN_DIR=/data/exp3/run_01 ./run_openwam_formal.sh
```

The launcher refuses to use missing GPUs or GPUs with more than 2 GiB already allocated.
Set `MAX_USED_MEMORY_MIB` to change that threshold, or pass `--allow-busy-gpus` only when
GPU sharing is intentional.

Progress is printed once per minute. Final artifacts are written to
`$RUN_DIR/summary/summary.md`, `all_results.csv`, and `status.json`. A complete
run has 200/200 task-condition-model cells and 20,000 rollouts.

The released-checkpoint smoke run only verifies the pipeline. It is not formal
OOD evidence because those checkpoints were not trained with this 40/10 split.

## Outputs

Each run is written under `runs/<run-id>/` and includes:

```text
config.json                 resolved configuration
metadata.json               commits, protocol, and hardware
<family>/<method>/<split>/  logs and normalized results
summary/summary.md           success-rate table
summary/success_rates.csv    machine-readable summary
```

`runs/` and generated manifests are intentionally not committed.

## Extending the Runner

To add another model, implement an adapter under `wm_eval/adapters/` and
register it in `wm_eval/adapters/__init__.py`. The adapter must consume the
shared manifests and write the common result format through
`EvalContext.write_result(...)`.

Experiment 1 can reuse the environment startup, manifest, and adapter approach,
but needs a separate runner and per-rollout result format for 32 samples of one
fixed simulator state.

## Git Hygiene

The `.gitignore` excludes runs, manifests, machine-local configuration,
checkpoints, videos, and datasets.
