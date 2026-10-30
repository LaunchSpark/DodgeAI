# Pattern Replay Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port Taylor Turner's engine-side death telemetry into our engine fork, record every pattern appearance in a sampling run, and rebuild the Death paths tab as an overview of all appearances plus a straight-through replay of the real game picture.

**Architecture:** Rust: `enemy_rows.rs`, `death.rs`, `trace.rs` come across from `TaylorTurnerIT/dodge-native@t3/death-telemetry-fyou-patterns` into `dodge-core`, hooked into `game.rs` as he hooks them, with one extension (each `PatternSpan` keeps the player's path and its rectangles). `dodge-batch`/`dodge-python` expose the trace per lane. Python: `dodge/death_paths.py` plays fixed seeds with the trace on, recording actions; `dodge/pattern_replay.py` re-simulates one appearance on a pixel engine. FastAPI serves a summary, one pattern's appearances, and a replay. React renders it with shadcn/ui.

**Tech Stack:** Rust (pyo3, maturin), Python 3.12, NumPy, FastAPI, React 19 + Tailwind 4 + shadcn/ui, Vite.

**Spec:** `docs/superpowers/specs/2026-10-30-pattern-replay-viewer-design.md`

## Global Constraints

- Two repositories. Engine: `D:/projects/dodge-ai-3`, branch `velocity-flow`, crates under `native/crates/`. Dashboard: `D:/projects/DodgeAI/.claude/worktrees/train-plan`, branch `train-plan`. Commit each task in the repository it touches.
- Rebuild and install the engine after any engine task whose Python tests need it: `cd D:/projects/dodge-ai-3/native/crates/dodge-python && maturin build --release && pip install --force-reinstall ../../target/wheels/dodge_native-*.whl`.
- Recorder on or off must not change a game: frames, score, death frame, RNG, rewards, observations, snapshot bytes. Training never turns it on.
- The existing `Telemetry` struct (`death_cause`, `death_pattern`, …) in `dodge-core/src/game.rs` stays as it is.
- Taylor's semantics are kept: first lethal contact in source order; 1-based gameplay frames counting only updates that began in `Mode::Game`; identity unavailable → `None`, never a guess.
- Sampling runs use the dashboard's existing seeds (`EVALUATION_FIRST_SEED = 1_000_000`), frame cap (`EVALUATION_FRAME_CAP = 18_000`), game limits (16..1024) and game settings without lives (lives = 0).
- Colours: rectangle `#0072B2`, enemy `#D55E00`, survived grey. Replay plays at 60 frames per second.
- No-pattern deaths: overview uses the last 120 frames; replay plays the last 300 frames before death.
- Training-speed gate: engine throughput with the port (recorder off) is within 2% of the baseline measured in Task 1.
- Python tests that need the engine skip themselves when `dodge_native` is missing, like the existing ones.

## Review Focus

- A game capped at 18,000 frames while a pattern is active: the appearance has no end; it must be listed as "unfinished" and replay to the cap, not crash or vanish. → Task 5 test `test_capped_pattern_is_unfinished`.
- Action repeat above 1: the replay must hold each recorded decision for `action_repeat` frames and stop at death exactly as `step_ml_batch` did. → Task 6 test `test_replay_matches_at_action_repeat_4`.
- Clicking an appearance while a new sampling run is replacing the data: the replay must refer to the run it came from or fail with a clear message, not replay another game. → Task 7 test `test_replay_of_stale_run_is_rejected`.
- A pattern nobody died in: the overview still draws its rectangles and its survived paths. → Task 5 test `test_survived_appearance_keeps_rectangles`.
- The same pattern id appearing twice in one game: two appearances, not one merged path. → Task 2 test `pattern_spans_are_separate_per_appearance`.

---

### Task 1: Engine baseline and enemy identity (`enemy_rows.rs`)

**Files:**
- Create: `D:/projects/DodgeAI/.claude/worktrees/train-plan/tools/engine_bench.py`
- Create: `D:/projects/dodge-ai-3/native/crates/dodge-core/src/enemy_rows.rs` (from Taylor's branch)
- Modify: `D:/projects/dodge-ai-3/native/crates/dodge-core/src/game.rs` (field `enemies` at line 250 and every mutation site: lines 375, 445, 1132, 1549, 1663, 1770, 2036, plus indexed writes)
- Modify: `D:/projects/dodge-ai-3/native/crates/dodge-core/src/lib.rs` (module + `pub use EnemySerial`)

**Interfaces:**
- Produces: `EnemyRows` with Taylor's API (`new`, `from_rows`, `push`, `remove`, `clear`, `serials() -> Option<&[EnemySerial]>`, `next_birth() -> Option<u64>`, `Deref<Target=[EnemyState]>`, mutable access to existing rows only); `EnemySerial::birth() -> u64`. `NativeGame::enemy_serials(&self) -> Option<&[EnemySerial]>`.
- Produces: `python tools/engine_bench.py` printing one line `frames/s: <float>` for 64 lanes, hard, patterns and powerups on, `step_frames=1`, `observe_velocity` every step, 3,000 steps of seeded random actions after a 200-step warm-up.

- [ ] **Step 1: Write `tools/engine_bench.py` and record the baseline**

Run three times with the currently installed engine: `python tools/engine_bench.py`. Write the median in the commit message of this task's DodgeAI commit; it is the gate for Task 3.

- [ ] **Step 2: Copy `enemy_rows.rs` and its tests from Taylor's branch**

`git -C <taylor clone> show origin/t3/death-telemetry-fyou-patterns:crates/dodge-core/src/enemy_rows.rs`. Keep his tests in the file.

- [ ] **Step 3: Write the failing equality test in `game.rs` tests**

```rust
#[test]
fn identities_do_not_affect_game_equality() {
    let game = NativeGame::new(NativeConfig::default());
    let restored = NativeGame::restore(&game.snapshot()).unwrap();
    assert_eq!(game, restored); // fresh identity namespace on restore
    assert_eq!(game.snapshot(), restored.snapshot());
}
```

Run: `cargo test -p dodge-core identities_do_not_affect_game_equality` — fails to compile until `EnemyRows` is in.

- [ ] **Step 4: Replace `enemies: Vec<EnemyState>` with `EnemyRows`**

`PartialEq` for `EnemyRows` compares `rows` only (Taylor derives it over identities too; ours must not, so restored games stay equal). Snapshots keep writing the plain rows.

- [ ] **Step 5: Run the whole engine suite**

Run: `cd D:/projects/dodge-ai-3/native && cargo test --workspace`
Expected: all pass, including `identities_do_not_affect_game_equality`.

- [ ] **Step 6: Commit both repositories**

```bash
git -C D:/projects/dodge-ai-3 commit -am "Give every enemy a birth identity (Taylor Turner's EnemyRows)"
git add tools/engine_bench.py && git commit -m "Add an engine throughput benchmark; baseline <median> frames/s"
```

### Task 2: Death diagnostic and gameplay trace in the engine

**Files:**
- Create: `native/crates/dodge-core/src/death.rs`, `trace.rs`, `trace_tests.rs` (from Taylor's branch)
- Modify: `native/crates/dodge-core/src/game.rs`, `lib.rs`

**Interfaces:**
- Consumes: Task 1 `EnemyRows`, `EnemySerial`.
- Produces (on `NativeGame`): `set_trace_enabled(&mut self, enabled: bool)`, `run_trace(&self) -> &RunTrace`, `death_diagnostic(&self) -> DeathDiagnostic`. Reset and restore keep the enabled flag (`RunTrace::begin(self.trace.enabled(), self.enemies.next_birth())`).
- Produces (our extension on `PatternSpan`): `path: Vec<(i32, i32)>` — player `(x, y)` raw fixed-point at the end of every gameplay frame from `start_frame` to its end; `rects: Vec<RectAtDeath>` and `rects_frame: Option<u32>` — captured once, at the first frame any rectangle is lethal (`pattern_type == 1 || sh == 2`), in `trace_rects()` form. `RunTrace::latest_sample(&self) -> Option<FrameSample>`.

- [ ] **Step 1: Copy `death.rs`, `trace.rs`, `trace_tests.rs` and wire the hooks**

Hooks exactly where Taylor places them in his `game.rs` (`begin_frame` before `update_game_frame` when the update began in `Mode::Game`; `end_frame` + `trace_record_frame` at the end of `update_game_frame`; `record_fyou` in `update_fyou`; `pattern_start` in `update_pattern_schedule` on selection; `pattern_end` on completion; `trace_death` at both lethal-contact sites; `DeathDiagnostic::record` beside our existing `Telemetry` death recording at line 882). Our `pattern_retry` restart (lives) is out of scope: the trace is only read with lives = 0.

- [ ] **Step 2: Write the failing tests in `trace_tests.rs`**

```rust
#[test] fn recorder_does_not_change_the_game()               // 32 seeds × hard, patterns, powerups, 1,500 frames, random inputs from a fixed RNG:
                                                            // snapshots, score and death frame equal with trace on vs off
#[test] fn pattern_spans_are_separate_per_appearance()      // a game where one id runs twice → two spans, each path.len() == end - start + 1
#[test] fn span_path_and_rects_are_recorded()               // a static pattern: rects_frame is the first lethal frame; every rect lethal
#[test] fn death_in_pattern_leaves_span_open()              // span.end_frame == None, death.frame == last path frame
#[test] fn killer_age_and_fyou_flag_are_exact()             // constructed: a guard-spawned enemy kills → fyou_spawned == Some(true), age == frames since its birth
```

Run: `cargo test -p dodge-core trace` — fail before the extension exists.

- [ ] **Step 3: Implement the `PatternSpan` extension and `latest_sample`**

Append to the active span in `record_frame` when `enabled`; capture rects inside `trace_record_frame` when the active span has `rects_frame == None` and a rectangle is lethal.

- [ ] **Step 4: Run** `cargo test --workspace` — all pass.

- [ ] **Step 5: Commit** `git -C D:/projects/dodge-ai-3 commit -am "Add Taylor Turner's death diagnostic and gameplay trace, with per-appearance paths"`

### Task 3: Expose the trace to Python; check the speed gate

**Files:**
- Modify: `native/crates/dodge-batch/src/lib.rs` (`BatchEnvironment::set_trace_enabled(&mut self, enabled: bool)` applying to every lane and to lanes created by later resets)
- Modify: `native/crates/dodge-python/src/lib.rs`
- Test: `D:/projects/DodgeAI/.claude/worktrees/train-plan/tests/test_engine_trace.py`

**Interfaces:**
- Produces (Python, on `NativeBatchEnv`), all positions and gaps in pixels as floats (raw / 65536), absent gaps (`-1`) as `None`, no pattern as `None`:
  - `set_trace_enabled(enabled: bool) -> None`
  - `lane_trace(lane: int) -> dict | None` (None when disabled):
    `{"frame": int, "fyou_triggers": [{"frame", "spawned"}], "patterns": [{"id", "type", "start", "end": int|None, "x": [float], "y": [float], "rects": [[x, y, w, h, lethal(0|1)]], "rects_frame": int|None}], "death": None | {"cause": str, "frame", "killer": None|{"personality", "x", "y", "size", "fyou_spawned": bool|None, "age": int|None}, "enemies_alive": [normal, kamikaze, explosion, powerup], "player": {"x", "y", "vx", "vy", "size"}, "active_pattern": None|{"id", "type", "start", "since"}, "rects": [[x, y, w, h, lethal]], "killer_path": [[frames_before_death, x, y]], "recent": {"frame", "x", "y", "vx", "vy", "lethal_rect_gap", "pending_rect_gap", "enemy_gap", "enemies", "pattern_id"} (lists)}, "contacts": [6 ints, DeathCause order]}`
  - `lane_sample(lane: int) -> dict | None`: the latest `FrameSample` with the same keys as one `recent` entry.

- [ ] **Step 1: Write the failing Python tests**

```python
def test_trace_is_off_by_default_and_on_after_enabling(): ...      # lane_trace(0) is None; after set_trace_enabled(True) and a reset, a dict
def test_death_trace_matches_existing_telemetry(): ...             # 16 seeds, random actions until death: trace["death"]["cause"] maps to
                                                                    # DEATH_CAUSES[observe_velocity()["death_cause"]] and active_pattern id == death_pattern
def test_lane_sample_frame_advances_one_per_frame(): ...
```

- [ ] **Step 2: Implement the bindings; rebuild and install the wheel; run** `pytest tests/test_engine_trace.py -v` — pass.

- [ ] **Step 3: Speed gate**

Run `python tools/engine_bench.py` three times. Expected: median within 2% of the Task 1 baseline. If not, make `EnemyRows` identity tracking follow the recorder flag (off → `identities: None`), re-run, and note it in the commit.

- [ ] **Step 4: Commit both repositories** (`"Expose the gameplay trace per lane"`, `"Test the engine trace from Python"`)

### Task 4: Commit the pending dashboard changes

The `train-plan` worktree has uncommitted work from earlier today (autosave over the active model, per-update metrics log, game-config fix, test isolation). Commit them as their own commit before this feature's commits.

- [ ] **Step 1:** `pytest tests -q -p no:cacheprovider --deselect tests/test_policy_parity.py` — all pass.
- [ ] **Step 2:** `git add README.md dashboard.py dodge/dashboard.py dodge/training.py tests/conftest.py tests/test_dashboard.py tests/test_training_session.py training_gui.py && git commit -m "Autosave over the active model and log metrics every update"`; add `.superpowers/` to `.gitignore` in the same commit.

### Task 5: Sampling run records every appearance (`dodge/death_paths.py`)

**Files:**
- Rewrite: `dodge/death_paths.py`
- Test: `tests/test_death_paths.py`

**Interfaces:**
- Consumes: Task 3 `set_trace_enabled`, `lane_trace`.
- Produces: `record(policy, seeds, *, frame_cap=18_000, progress=None, **game_settings) -> Recording`
  ```python
  @dataclass
  class Recording:
      run_id: str                      # uuid4 hex, new per run
      settings: dict                   # the game_settings used, incl. step_frames
      games: int
      actions: dict[int, np.ndarray]   # seed -> uint8 decisions in order
      appearances: list[dict]          # see below; index in this list is the appearance id
      def summary(self) -> dict        # {"run_id", "games", "patterns": [{"pid", "type", "encounters", "deaths", "rate"}]}
      def pattern(self, pid: int) -> dict  # {"run_id", "pid", "layout": rects|None, "appearances": [...]}
  ```
  Appearance: `{"id", "seed", "pid" (-1 = no pattern), "type": "static"|"moving"|None, "start", "end", "outcome": "survived"|"died"|"unfinished", "cause": "rect"|"enemy"|None, "kind": str|None, "x", "y", "rects", "death": None|{"killer", "alive", "since", "score", "frame"}}`.
  Patterns ordered deadliest first, those met fewer than 5 times after the rest (today's `MIN_RANKED` rule); `-1` last. `layout` is the shared rect list when every appearance of the pattern has the same rects, else `None`.

- [ ] **Step 1: Write the failing tests** (engine-backed, small seed counts, a seeded random policy)

```python
def test_every_pattern_appearance_is_recorded(): ...   # sum of encounters == len([a for a in appearances if a["pid"] >= 0])
def test_died_appearance_ends_at_death(): ...          # outcome "died": end == death frame, x/y length == end - start + 1
def test_no_pattern_death_keeps_last_120_frames(): ... # pid -1: len(x) <= 120, start == end - len(x) + 1
def test_capped_pattern_is_unfinished(): ...           # frame_cap=600: any open span → outcome "unfinished", end == last traced frame
def test_survived_appearance_keeps_rectangles(): ...   # outcome "survived" with rects_frame set → rects non-empty
def test_actions_are_recorded_per_seed(): ...          # len(actions[seed]) == decisions taken by that lane
```

- [ ] **Step 2: Implement.** Lanes run with the trace on; read `lane_trace(lane)` once when a lane's game ends (death or cap), then reset it with a filler seed as today. The Python killer tracing and per-frame `lane_state` reads are removed.
- [ ] **Step 3: Run** `pytest tests/test_death_paths.py -v` — pass.
- [ ] **Step 4: Commit** `"Record every pattern appearance from the engine trace"`

### Task 6: Replay one appearance (`dodge/pattern_replay.py`)

**Files:**
- Create: `dodge/pattern_replay.py`
- Test: `tests/test_pattern_replay.py`

**Interfaces:**
- Consumes: Task 5 `Recording`, appearance dicts; Task 3 `lane_sample`.
- Produces: `replay(recording: Recording, appearance_id: int) -> Replay`
  ```python
  @dataclass
  class Replay:
      frames: np.ndarray        # (n, 128, 128) uint8 PICO-8 palette indices
      samples: list[dict]       # one per frame: {"st": 0|1|2, "enemies", "enemy_gap", "rect_gap"}
      matched: bool
      reason: str | None        # why it did not match
  ```
- Engine: `dodge_native.NativeBatchEnv(step_frames=1, execution="serial", board=False, pixels=True, ml=True, …settings, lives=0)`, `reset_ml_batch([seed])` (verified to reproduce the sampling engine frame for frame), trace on. Each recorded decision is held `settings["step_frames"]` frames, stopping at death. Frames whose `lane_sample(0)["frame"]` lies in `[start, end]` are kept. Checks: at `start`, `pattern_id == pid` (skip for `pid == -1`); for `"died"`, the lane dies at `end` with the recorded cause; for `"survived"`, the pattern id changes after `end`.

- [ ] **Step 1: Write the failing tests**

```python
def test_replay_reproduces_every_death(): ...           # 64 seeds, random policy, action_repeat 1: every "died" appearance → matched, len(frames) == end - start + 1
def test_replay_matches_at_action_repeat_4(): ...       # same at step_frames=4
def test_replay_flags_a_tampered_action(): ...          # flip one action before start → matched False, reason set
def test_no_pattern_replay_is_last_300_frames(): ...    # pid -1 → len(frames) == min(300, death frame)
```

- [ ] **Step 2: Implement; run** `pytest tests/test_pattern_replay.py -v` — pass.
- [ ] **Step 3: Commit** `"Replay a recorded pattern appearance on the pixel engine"`

### Task 7: Dashboard API

**Files:**
- Modify: `dodge/dashboard.py` (`record_death_paths`, `Dashboard.death_path_data`, routes)
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: Task 5 `Recording`, Task 6 `replay`.
- Produces:
  - `GET /api/death-paths` → `{"status", "message", "finished", "total", "label", "data": Recording.summary() | None}`
  - `GET /api/death-paths/pattern/{pid}?run={run_id}` → `Recording.pattern(pid)`
  - `GET /api/death-paths/replay/{appearance_id}?run={run_id}` → `{"fps": 60, "count": n, "frames": base64(zlib.compress(frames.tobytes())), "samples": [...], "matched", "reason", "appearance": {...}}`
  - A `run` that is not the current recording → 409 `"That sampling run has been replaced; reopen the pattern."`
  - `POST /api/death-paths` unchanged (games 16..1024).

- [ ] **Step 1: Write the failing tests** with a fake `Recording` injected into `dashboard.death_paths.result`:
  `test_summary_lists_patterns`, `test_pattern_returns_its_appearances`, `test_replay_payload_decodes_to_frames` (zlib-decompress → `count*128*128` bytes), `test_replay_of_stale_run_is_rejected` (409).
- [ ] **Step 2: Implement; run** `pytest tests/test_dashboard.py -v` — pass.
- [ ] **Step 3: Commit** `"Serve appearances and replays from the death-paths recording"`

### Task 8: Front end

**Files:**
- Delete: `web/src/components/death-paths.tsx`
- Create: `web/src/components/death-paths/index.tsx` (tab: run controls, progress, pattern Select, Died/Survived/All ToggleGroup, overview or replay)
- Create: `web/src/components/death-paths/overview.tsx` (canvas diagram + Table in ScrollArea)
- Create: `web/src/components/death-paths/replay.tsx` (canvas player, per-frame line, end overlay)
- Create: `web/src/lib/pico8.ts` (`PICO8: [number, number, number][]`, copied from `dodge/palette.py`)
- Modify: `web/src/lib/api.ts` (types for the three responses), `web/src/App.tsx` (import path)

**Interfaces:**
- Consumes: Task 7 endpoints.
- Behaviour fixed by the spec:
  - Pattern Select ordered as the summary returns; each item reads `pattern {pid} ({type}) · {rate}% of {encounters}`; `-1` reads `No pattern`.
  - Overview canvas 128×128 playfield scaled to the card: `layout` rects (or nothing when `null`), every visible appearance's path; died paths in `#0072B2` / `#D55E00` by cause with a dot at the last point; survived and unfinished in muted grey. Filter defaults to All. Hover highlights; clicking a path, a dot, or a Table row opens the replay.
  - Table columns: Seed, Outcome (Badge: rectangle / enemy / survived / unfinished), Frames in pattern, Game frame.
  - Replay: fetch, `DecompressionStream("deflate")` → bytes, draw frame i at 60 fps with `requestAnimationFrame` using `PICO8`, `imageSmoothingEnabled = false`. No scrubber. Under it, the frame's line: pattern state, enemies, nearest enemy gap, nearest lethal rectangle gap. At the end: last frame dimmed, outcome Badge, the summary (frames into pattern, score; for deaths killer kind, age, "spawned by the guard"), and a **Replay** Button. **← All appearances** Button always visible. `matched === false` shows `reason` in place of the replay.
  - shadcn components only from `web/src/components/ui/` (Card, Select, ToggleGroup, Table, ScrollArea, Badge, Button, Progress, Tooltip); add `table`-adjacent pieces with `npx shadcn add` only if one is missing.

- [ ] **Step 1: Implement**
- [ ] **Step 2: Verify:** `cd web && npm run build && npm run lint` — no errors.
- [ ] **Step 3: Commit** `"Rebuild the Death paths tab: all appearances, then a straight replay"`

### Task 9: End-to-end check and rollout

- [ ] **Step 1: Zero-mismatch run.** In Python: `record()` 256 seeds with the live checkpoint (`artifacts/models/velocity-flow-v2-latest.zip`, CPU, argmax) and current game settings; `replay()` every died appearance. Expected: all `matched`.
- [ ] **Step 2: Timing.** `record()` at 512 games is no slower than the old job (time the old one first from the previous commit).
- [ ] **Step 3: Full suites.** `cargo test --workspace` (engine) and `pytest tests -q -p no:cacheprovider` (dashboard).
- [ ] **Step 4: README.** Setup section mentions the trace is part of the `velocity-flow` engine; Death paths description updated.
- [ ] **Step 5: Roll out.** Save the live model through `POST /api/save` under its active name, wait for the "Saved" toast, stop the dashboard process, relaunch it from `D:/projects/DodgeAI` with its previous arguments on port 8765, load the model back, and confirm the Death paths tab records and replays.
- [ ] **Step 6: Commit** `"Document the pattern replay viewer"`
