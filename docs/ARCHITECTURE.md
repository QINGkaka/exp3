# Architecture

```text
Experiment runner
  -> EvalContext (tasks, conditions, environment, result schema)
      -> FastWAMAdapter -> FastWAM + vendored RoboTwin on one GPU
      -> OpenWAMAdapter -> model server GPU + official RoboTwin simulator GPU
      -> FutureAdapter  -> model-specific deployment only

Official RoboTwin manifest builder
  -> sealed task/mode/seed/instruction entries
  -> consumed by every adapter

result.json files
  -> model-independent Experiment 3 summarizer
```

## Ownership Boundary

The harness owns experimental choices and records. An adapter owns only the
translation between the shared episode contract and one model's inference API.
Upstream model code remains pinned and externally checked out.

FastWAM's upstream evaluator lacks three required controls, so the repository
ships a minimal patch for task files, explicit GPU IDs, and sealed manifests.
The patch is versioned separately to make that deviation visible and auditable.

OpenWAM already exposes the required server and Labtasker manifest interfaces,
so its source tree remains unchanged.

## Experiment 1 Extension

Experiment 1 should reuse `EvalContext`, manifest generation, adapters,
provenance, and per-episode success records. Its runner must add a sampling
seed distinct from the simulator seed and preserve:

```text
task, state_id, simulator_seed, model, sampling_seed, rollout_id, success
```

The simulator seed stays fixed across 32 rollouts of one state. The policy
sampling seed changes. This is deliberately not encoded as an Experiment 3
option because the output unit and statistical question are different.
