# Non-blocking Watch and live model reload

**Date:** 2026-08-23
**Status:** approved design, not yet implemented

## Summary

Replace synchronous Watch/save behaviour with a safe asynchronous checkpoint
pipeline. Checkpoints are captured only between optimizer updates, the GUI stays
responsive, and watched agents adopt the newest completed policy after death.

## Background

Three facts about the code as it stands motivate this.

`_save_to_path` queues a request and then blocks the caller on
`request.finished.wait(timeout=15)`. The caller is the Pygame event loop, so
Watch, named Save, and applying a game configuration all freeze the dashboard
until the trainer reaches a point where it can serve the request.

That point is narrower than it looks. `_process_save_requests` is only reached
from `_should_stop`, which runs inside `_on_step` — during rollout collection.
A save requested while `train()` is running is not served until the next
rollout begins.

`_completed_updates` is `_n_updates // n_epochs`. Stable-Baselines3 increments
`_n_updates` once per epoch, and `velocity-flow-v2` sets `target_kl`, which
breaks the epoch loop early. The quotient therefore under-counts completed
updates for v2, and every consumer of that number — the dashboard's update
figure, watch snapshot filenames, `exact_update` — is wrong for it today.

## Key changes

### Training and checkpoint pipeline

- Add a post-update callback to `DashboardPPO`, invoked only after `train()`
  completes successfully and the optimizer hook is restored.
- Track completed PPO updates explicitly rather than deriving them from
  `_n_updates / n_epochs`; v2's KL early stopping makes that calculation wrong.
- Service queued saves during rollout, while paused, and at the post-update
  safe point — never during an optimizer step.
- Replace GUI-side blocking waits with asynchronous requests and completion
  events.
- Use this pipeline for Watch, named model saves, and game-configuration
  restarts.
- On training failure or shutdown, complete every pending request with an error
  rather than leaving orphaned requests.

### Immutable watch snapshots

- Give each training session a private live-watch directory.
- Save to a unique hidden temporary ZIP, then atomically rename it to
  `update-XXXXXXXX.zip`.
- Never let watchers read the mutable `*-latest.zip` autosave.
- While at least one watcher process is alive, publish one snapshot after every
  completed PPO update.
- Retain the newest three snapshots; remove older ones on a best-effort basis so
  Windows file locks cannot crash training.
- Track watcher `Popen` handles and stop continuous publishing after all
  watchers exit.

### GUI behaviour

- `watch_agent()` returns immediately.
- If a checkpoint from the same session is already available, open it
  immediately — even if it is older — then let live reload catch up.
- If no safe checkpoint exists, show "Waiting for current update…" and launch
  automatically after the next safe point.
- Deduplicate repeated clicks while an initial Watch launch is pending;
  additional clicks after launch may open additional watcher windows.
- Named Save disables its button while pending, then refreshes and selects the
  saved checkpoint on completion.
- Game configuration is committed and training restarted only after its
  preservation checkpoint succeeds.
- Convert save, rename, and process-launch failures into GUI toasts; no
  exception from these operations may escape the event loop and close training.
- Add a small session event interface, such as
  `drain_events() -> list[SessionEvent]`, for asynchronous success/error
  notifications.

### Watched-agent reload

- Extend `watch_ppo.py` with live-directory and current-update arguments.
- At each death, before resetting the environment, scan for the highest
  completed update newer than the loaded one.
- Load and validate the new checkpoint, then swap the model and recompute the
  optional field extractor.
- Update the window title with the newly loaded update.
- If loading fails, keep the existing in-memory model, report the error, and
  retry at the next death.
- Never change models during an active episode.

## Test plan

- Click Watch during a deliberately slow learning phase and verify the call
  returns immediately, the GUI stays responsive, and training continues.
- Verify the pending Watch launches after the current PPO update completes.
- Verify an existing same-session checkpoint launches immediately and reloads a
  newer update after death.
- Confirm snapshots are invisible until their atomic rename and that partial
  files are ignored.
- Simulate ZIP-save, rename, and `Popen` failures and confirm they produce
  events/toasts without terminating training.
- Verify v2 KL early stopping increments the completed-update counter exactly
  once per `train()` call.
- Confirm one snapshot is published per update only while watchers are active,
  and that retention keeps the newest three.
- Verify named Save and game reconfiguration are non-blocking and apply their
  state changes only after successful saves.
- Verify a corrupt or incompatible newer checkpoint leaves the watcher on its
  current model.
- Run the complete existing suite plus a real temporary-directory integration
  test with the watcher process mocked.

## Assumptions

- Reloading occurs only between watched episodes.
- Watchers continue using CPU inference.
- Multiple watcher windows share the same published snapshot stream.
- "Use an older checkpoint" means the loaded source checkpoint or a prior
  snapshot from this training session, never an unrelated discovered model.
- A freshly initialised model with no safe checkpoint waits asynchronously for
  its first completed update.
- Publishing every update while watched is acceptable despite the additional
  checkpoint-write overhead.

## Open question carried into implementation

Publishing cost is assumed acceptable rather than measured. A velocity-flow
checkpoint is about 1.4MB and v2 completes an update roughly every 2.4 seconds,
so continuous publishing writes on the order of 35MB a minute, serialised on the
training thread at the safe point. Measure one save's duration during
implementation; if it is a material fraction of a cycle, publish on a cadence
rather than on every update.
