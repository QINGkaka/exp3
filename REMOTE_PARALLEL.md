# Experiment 3 Remote OpenWAM Parallel Run

`run_openwam_remote_formal.py` runs RoboTwin simulators on the local GPUs and
keeps one OpenWAM model server resident on each remote GPU. SSH ControlMaster
connections carry the eight server sessions and eight port forwards without
requiring changes to the remote SSH daemon.

The launcher runs globally phased methods (`no_wm`, then `wm`). Within a method,
all seen workers run in parallel, followed by all unseen workers. Set
`--clients-per-server` to run multiple isolated RoboTwin clients per model
server. Completed immutable part files are reused even when worker assignments
change between runs.

Apply the session-isolation patch to the local OpenWAM checkout before the first
run. The launcher synchronizes the patched server file to the remote checkout:

```bash
git -C /path/to/OpenWAM apply /path/to/exp3/patches/openwam-session-isolation.patch
```

## Formal command

```bash
python run_openwam_remote_formal.py \
  --config config.openwam.formal.json \
  --run-dir runs/openwam_remote_formal_100 \
  --remote-host openwam-inference-host \
  --remote-openwam-repo /srv/openwam/OpenWAM \
  --remote-python /srv/openwam/env/bin/python \
  --remote-no-wm-checkpoint /srv/checkpoints/no_wm \
  --remote-wm-checkpoint /srv/checkpoints/wm \
  --remote-gpus 0-7 \
  --sim-gpus 0-7 \
  --clients-per-server 4 \
  --remote-port-base 8848 \
  --local-port-base 9848 \
  --methods all \
  --condition all \
  --dynamic-pool
```

The command is restart-safe. Run it again with the same run directory after a
failure; valid `parts/*.json` files are preserved and skipped. The launcher
synchronizes only `openwam/deploy/server.py` to the remote OpenWAM checkout so
the server and local orchestration use the same session-isolation protocol.

The formal default is four clients per model server. A same-task A/B run on one
RTX 5090 simulation GPU and one H100 inference GPU completed 8/8 rollouts in
1007 seconds with four clients versus 2088 seconds with three clients. Peak
simulation-GPU memory was 22,568 MiB with four clients; five clients reached
29,482 MiB and left too little headroom for the formal run.

## tmux

```bash
tmux new-session -s exp3_remote_8x8
# Run the command above, then detach with Ctrl-b d.
tmux attach -t exp3_remote_8x8
```

Per-worker, server, and tunnel logs are under `<run-dir>/remote_logs/`. Monitor
aggregate committed and in-flight progress in another tmux window:

```bash
python watch_rollout_progress.py runs/openwam_remote_formal_100 --total 20000
```

## Smoke command

```bash
python run_openwam_remote_formal.py \
  --config config.openwam.formal.json \
  --run-dir runs/remote_parallel_smoke \
  --remote-host openwam-inference-host \
  --remote-openwam-repo /srv/openwam/OpenWAM \
  --remote-python /srv/openwam/env/bin/python \
  --remote-no-wm-checkpoint /srv/checkpoints/no_wm \
  --remote-wm-checkpoint /srv/checkpoints/wm \
  --remote-gpus 1 \
  --sim-gpus 7 \
  --clients-per-server 1 \
  --remote-port-base 8949 \
  --local-port-base 9949 \
  --methods no_wm \
  --condition clean \
  --tasks place_a2b_left \
  --episodes 1
```

`--session-isolation` is enabled on every remote server. Each WebSocket owns
its RESET state, action buffer, and request counters while all sessions on that
server share the loaded inference engine. Session isolation requires the
synchronous executor.

The rollout queue is dynamic, but inference endpoints are sticky: worker `N`
uses server `N % server_count` for the lifetime of a rollout job. This preserves
per-session action-buffer state; it is not request-level load balancing across
all inference GPUs.
