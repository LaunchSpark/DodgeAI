# Pattern replay viewer on Taylor Turner's death telemetry

**Date:** 2026-10-30
**Status:** design, awaiting approval

## Summary

Rebuild the dashboard's **Death paths** tab so it answers two questions about a
trained policy: *in what situations does it die*, and *how does it behave inside
each pattern*. A sampling run records every pattern appearance in a block of
fixed seeds — survived and died — using Taylor Turner's engine-side recorder,
ported into our engine fork. The viewer shows every appearance of the chosen
pattern at once as a diagram, and clicking one plays that appearance as the real
game picture, start to finish.

## Why

The current tab (`dodge/death_paths.py` + `web/src/components/death-paths.tsx`)
is a port of Taylor's viewer (`dodge-native`, branch
`t3/thread-progress-metaanalysis`, `paper/death_paths.template.html`), but it
rebuilds his data in Python from `lane_state` every frame. That reconstruction
guesses the killer (nearest enemy in the frame before death, traced back by a
6 px match), reports a killer "age" that is really frames tracked, cannot say
whether the killer was spawned by the anti-camping guard, keeps only deaths
during a pattern, and keeps only the last 120 frames. Taylor's recorder
(`dodge-native`, branch `t3/death-telemetry-fyou-patterns`) records all of this
exactly inside the game loop. The owner wants as much of his backend as
possible because it is the more mature of the two.

The owner looks at this view to find the situations the model dies in and how
it generally behaves in different patterns. Seeing only deaths answers the
first; the second needs the appearances it survived as well.

## Goals

1. **Engine recorder.** Port Taylor's death diagnostic, gameplay trace and enemy
   identity into our engine fork, wired into the game loop the way he wires
   them, off by default, readable per lane from Python.
2. **Sampling run records every pattern appearance**, survived or died: pattern
   id and type, start and end frame, outcome, the player's path for the whole
   appearance, and — for deaths — Taylor's full death context (exact cause,
   exact killer with true age and guard flag, killer path, enemies alive by
   kind, gaps, rectangles at death).
3. **Overview**: choose a pattern; every appearance of it is drawn at once on
   the 128×128 playfield over the pattern's rectangles. Deaths coloured by cause
   (rectangle blue `#0072B2`, enemy orange `#D55E00`, Taylor's colours) with a
   dot where the agent died; survived appearances as faint grey lines. A table
   lists every appearance.
4. **Replay**: clicking an appearance (its line, its dot, or its table row)
   swaps the diagram for the **real game picture** of that appearance, playing
   from the frame the pattern starts to the frame it ends or the agent dies.
   **No scrubber.** It plays straight through; at the end it shows the outcome
   over the last frame with a **Replay** button. A **← All appearances** button
   returns to the overview at any time.
5. **Front end in shadcn/ui**, matching Taylor's viewer in structure (pattern
   picker ordered by lethality, diagram, list, detail panel) and using the
   components already in `web/src/components/ui/` wherever one fits.

## Non-goals

- No change to training: observations, rewards, actions, the random generator,
  snapshot bytes and determinism of a game are identical with the recorder on
  or off. Training never turns it on.
- No change to the existing per-lane death telemetry (`death_cause`,
  `death_pattern` in `dodge-core/src/game.rs`), which the env and metrics use.
- No scrubber, frame stepping or speed control in the replay.
- No storing video. The replay is re-simulated on demand.
- No statistics beyond what the current tab shows plus Taylor's per-pattern
  death rate; the paper's bootstrap intervals and 1-frame/4-frame comparison
  are out of scope.
- The **Deadliest patterns** chart and other evaluation charts in `App.tsx` are
  untouched.

## Anti-goals

- A replay that is not the game that was recorded. If re-simulating a seed with
  the recorded actions does not reproduce the recorded outcome (same death
  frame and cause, or the same pattern end), the viewer must say so rather
  than play something else.
- Making the sampling run noticeably slower than today's death-paths job at the
  same game count.

## What exists

### Our engine

- Repo `D:/projects/dodge-ai-3`, branch `velocity-flow` (the owner's fork of
  Taylor's `TaylorTurnerIT/dodge-ai-3`). Crates under `native/crates/`:
  `dodge-core` (game, `game.rs` ≈4,100 lines), `dodge-batch`, `dodge-python`
  (the `dodge_native` module).
- Built with `maturin build --release` in `native/crates/dodge-python` and
  installed with `pip install` of the wheel (see `README.md`, Setup). The
  installed module is what every Python process imports; a running dashboard
  keeps the old module until it is restarted.
- `dodge.env.make_engine(...)` builds a batched engine; it has `step_ml_batch`,
  `observe_velocity`, `lane_state`, `reset_ml_batch`, `reset_ml_lanes`, and
  `step_pixels` (real rendered frames; used by `watch_ppo.py` with
  `pixels=True, step_frames=1`).

### Taylor's recorder (source to port)

Repo `TaylorTurnerIT/dodge-native`, branch `t3/death-telemetry-fyou-patterns`
(commits `78217f5`, `2e729b2`). The owner has confirmed licensing is not a
concern.

- `crates/dodge-core/src/death.rs` (121 lines): `DeathCause`,
  `DeathDiagnostic` (first lethal cause + same-frame contact counts).
- `crates/dodge-core/src/trace.rs` (338 lines): `RunTrace` — per-frame
  `FrameSample` (position, velocity, gap to nearest lethal rectangle, to nearest
  charging rectangle, to nearest enemy, enemy count, pattern id); 120-frame
  player ring; 60-frame enemy-position history; fyou triggers; `PatternSpan`
  (id, type, start, end); whole-game samples every 30 frames; `DeathContext` at
  the first lethal contact.
- `crates/dodge-core/src/enemy_rows.rs` (214 lines): `EnemyRows` wraps the enemy
  list with `EnemySerial` birth identities so the killer is known exactly. This
  must come across too: the trace depends on it, and neither his `main` nor our
  fork has it.
- Hooks in his `game.rs`: `trace.begin_frame` at the start of an in-game update,
  `trace.end_frame` + `trace_record_frame` at its end, `record_fyou` where the
  guard spawns, `pattern_start` / `pattern_end` at pattern selection and
  completion, `trace_death` at the first lethal contact,
  `set_trace_enabled`, `run_trace`, `death_diagnostic`.
- Python conversion in `crates/dodge-rl-env/src/python.rs` (`trace_dict`,
  `death_dict`) — the shape to expose from our `dodge-python`.
- His viewer `paper/death_paths.template.html` and `build_interactive.py` on
  `t3/thread-progress-metaanalysis` show the intended reading of the data.

### Our dashboard

- Job: `POST /api/death-paths {games}` (16..1024) runs
  `dodge.dashboard.record_death_paths` → `dodge.death_paths.record` in a
  `PolicyJob` thread on a frozen copy of the live policy, on seeds from
  `EVALUATION_FIRST_SEED` (1,000,000), frame cap 18,000, with the current game
  settings. `GET /api/death-paths` returns the result.
- UI: `web/src/components/death-paths.tsx`, a tab in `App.tsx`; React 19,
  Tailwind 4, shadcn components in `web/src/components/ui/`: badge, button,
  card, dialog, input, label, progress, scroll-area, select, separator, slider,
  sonner, switch, table, tabs, toggle-group, toggle, tooltip.

## Decided

**By the owner:**

- Use as much of Taylor's backend as possible; the front end looks like his but
  uses shadcn/ui as much as possible.
- Bring across all three of his engine pieces: `death.rs`, `trace.rs` and
  `enemy_rows.rs` (the enemy identity the trace depends on).
- Overview shows *all* appearances of the pattern at once, as the diagram.
- Clicking one switches to the real game picture of that appearance and plays
  its whole duration, straight through, with no scrubber.
- Licensing is not a concern.

**My calls (the owner can override):**

- **The replay window** is the pattern's lifetime: from its start frame to its
  end frame, or to the death frame if the agent died first.
- **Survived appearances** are in the overview as faint grey lines, with a
  Died / Survived / All filter defaulting to All.
- **The sampling run** stays a dashboard job on the live policy, same seeds,
  game cap and limits as today, and keeps its current trigger and progress UI.
- **Player path per appearance** is recorded for the whole appearance, not only
  the last 120 frames. This is our extension to Taylor's trace: append the
  per-frame player position to the active `PatternSpan` while a pattern runs.
  Budget: patterns are active in roughly half of all frames, so 1,024 games ≈
  2–3 M positions ≈ 10–20 MB server-side; the API returns one pattern's
  appearances at a time.
- **Actions are recorded per game** (one byte per decision) during the sampling
  run, so any appearance can be re-simulated.
- **Replay is computed on the server when clicked**: a fresh single-lane engine
  with the same game settings resets to the seed, steps the recorded actions
  without rendering up to the pattern start, then renders each frame of the
  window. Before rendering, it checks that the re-simulated game reaches the
  pattern start at the recorded frame with the recorded pattern id; after, that
  the outcome matches. A mismatch is reported in the UI (anti-goal above).
- **Per-frame line under the replay**: pattern state (charging / lethal),
  enemies on screen, gap to nearest enemy, gap to nearest lethal rectangle —
  Taylor's `FrameSample` fields, read from the replay engine's recorder.
- **End-of-replay summary**: outcome, frames into the pattern, score, and for
  deaths the killer (kind, true age, spawned by the guard or not).
- **`death_paths.py` becomes a thin reader** of the recorder; the Python
  killer-tracing heuristic is removed.
- **Deaths with no pattern active** get their own entry at the end of the
  pattern picker, "No pattern". Its overview draws each death's last 120 frames
  (Taylor's `recent` ring), and its replay plays the last 300 frames (5 s)
  before death. The non-pattern part of a game that ended in a pattern is not
  recorded.
- **Rectangles for every appearance, not only deaths.** Our extension to
  `PatternSpan` also stores the pattern's rectangles once per appearance, at
  the first frame they are lethal (or at the start for moving patterns, which
  are lethal throughout), in the same form as Taylor's `RectAtDeath`. This
  lets the overview draw a pattern nobody died in, and survived appearances
  of layouts that vary between games. It costs a few rectangles per
  appearance, which is negligible next to the paths.
- **Training speed is protected by a gate, not by changing Taylor's design.**
  `EnemyRows` is ported as is (identity kept on every game), with the frame
  hooks doing nothing while the recorder is off. Acceptance: training
  throughput on the GPU preset (64 envs × 256 steps, velocity-flow-v2) falls
  by no more than 2%. If it falls by more, identity tracking is switched off
  along with the recorder. Gameplay equality and snapshot comparisons used by
  our tests must ignore identities, as his trace and diagnostic already do.
- **Engine work lives in the `velocity-flow` branch of `dodge-ai-3`**; the
  dashboard work in this repository. The README's Setup section already
  describes building the wheel from that branch.

## Left to the builder

- How the hooks map onto our `game.rs`, which has diverged from Taylor's
  (ours ≈4,100 lines with the VelocityFlow additions, his ≈3,500). Keep his
  semantics: first lethal contact in source order, 1-based gameplay frames that
  exclude the startup frames, identity lost → `None` rather than a guess.
- Whether `EnemyRows` replaces our enemy `Vec` directly or sits beside it, as
  long as gameplay and snapshots are unchanged.
- The Python API shape on our engine (per-lane `trace(lane)` dict, a batched
  call, or fields on `lane_state`), and how the per-appearance path is pulled
  out efficiently.
- How replay frames reach the browser (a frame array of palette indices, PNGs,
  or a stream) and the playback rate (the game's 60 fps by default).
- How appearances are drawn when a pattern's layout varies between games;
  Taylor's approach (one shared layout when every game agrees, otherwise
  per-appearance rectangles) is a good default.
- Exact component choices within shadcn/ui, and the empty, loading and running
  states.

## How to know it works

- Engine: tests that turning the recorder on leaves a game's frames, score,
  death frame and snapshot bytes identical to recorder off, over a set of seeds
  and settings; tests of the killer, age, guard flag and pattern spans in
  constructed scenarios (Taylor's `trace_tests.rs` is the model).
- Replay: for every death in a sampling run of at least 256 games, re-simulating
  reproduces the recorded death frame and cause — zero mismatches (Taylor ran
  the same check on 2,048 games before trusting his traces).
- Sampling run time at 512 games is no worse than today's job.
- The dashboard test suite and the web build pass.
