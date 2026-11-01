# Critic-Only Game State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give velocity-flow's value function the hidden game state the actor cannot see, so the critic can tell unavoidable deaths from bad play. The critic sees nothing the actor's own actions could have changed.

**Architecture:** The Rust engine (`../dodge-ai-3`, branch `velocity-flow`) encodes a fixed-length vector of critic features from each lane's `NativeGame` at the current frame. `DodgeVecEnv(critic_features=True)` appends it to the 3076-value observation. A new architecture, `velocity-flow-v3`, keeps v2's actor unchanged: it reads only the first 3076 values. Its critic's first layer is widened to take the extra columns, which start at zero, so a v3 model continued from v2 behaves exactly like v2 at its first update. A probe (Task 4) measures whether the features predict returns before any training is spent.

**Tech Stack:** Rust (dodge-core, dodge-batch, pyo3 dodge-python, maturin), Python 3.12, PyTorch, Stable-Baselines3 PPO, pytest.

**Spec:** no separate document. The design below was agreed in conversation on 2026-11-01.

## Design

**What is unbiased.** The actor is reactive: π(a | o_t), with o_t a function of the game state s_t. Two consequences:

- Any baseline that is a function of s_t, taken before a_t, leaves the policy gradient unbiased.
- `V(s_{t+k})` for a Markov s is exactly the expected return-to-go. So GAE's bootstrapping is also unbiased.

So the critic may see anything in `NativeGame` at frame t, including the RNG state and deterministic previews computed from it. It must never see hindsight: what actually happened after t, such as the enemy that really spawned or the pattern that really fired. Those outcomes depend on a_t…a_{t+k}.

The rule is enforced by the signature: every feature is computed by `encode_critic_features(game: &NativeGame, out: &mut [f32])`, an immutable read of one frame.

**Feature groups.** 295 floats, each in [-1, 1]:

| Group | Count | Contents |
|---|---|---|
| `clock` | 12 | survival_frames/18000; score/`SCORE_SCALE` (largest finite pattern `mins`); speed, enemy_est, bounce_cap_static, bounce_cap_moving as ratios to their difficulty-curve targets; enemy_max_size/6; the 5 `enemy_stats` weights normalised to sum 1 (all 0 if they sum to 0) |
| `timers` | 10 | pattern countdown, (delay − timer)/spawn_rate/600, clamped to [0, 1], 0 while a pattern is active or patterns are off; pattern_active; spawn countdown, 1 − enemy_timer/threshold, clamped; freeze_active; freeze_timer and size_timer, each over its maximum; freeze_rate; player size/4; friendly_timer over its maximum; friendly_enabled |
| `active_pattern` | 42 | one-hot over the 39 patterns; fraction of rects finished; fraction lethal (`sh == 2`, shown, not finished); pattern.timer/600, clamped |
| `next_pattern` | 78 | each of the 39 patterns' probability if the timer fired now (from the weighted list); one-hot of the pattern the current RNG state would pick (all 0 if none is eligible or a pattern is active) |
| `next_enemy` | 9 | one-hot of the size class (3, 4, 5, 6) and of the personality (0–4) the current RNG state would roll for the next spawn |
| `nearest_enemies` | 144 | the 8 enemies nearest the player, nearest first, 18 values each: present; dx/64 and dy/64 (clamped); vx and vy over `VELOCITY_SCALE`; size/6 and max_size/6; isizing; is_dying; inside; has_life; life over its starting value; one-hot personality over −1…4. Zero-padded |

The RNG previews are functions of s_t, but they are only guesses: draws between now and the event shift the actual result. The probe and the A/B decide whether they earn their place.

## Global Constraints

- The actor's input is the first `OBSERVATION_SIZE` (3076) values, unchanged. Nothing in `velocity-flow-v2` behaves differently.
- Every critic feature is computed from `&NativeGame` at the current frame. No feature may read a later frame, a trajectory or the trace recorder.
- Encoding critic features never changes game state: `NativeGame` before and after must compare equal (it derives `PartialEq`).
- The refactors in Task 1 leave the game bit-identical: every existing dodge-ai-3 test and DodgeAI's `tests/test_env_parity.py` still pass.
- New critic input columns start at zero. Continuing v2 → v3 reproduces v2's logits and values exactly.
- `velocity-flow-v3` uses v2's hyperparameter table and requires `half_resolution=True`.
- Work on a branch `critic-features` in both repositories. DodgeAI's `main` has unrelated uncommitted changes (`velocity-flow-v2.zip`, `dodge/charts.py`, `web/src/components/vega-chart.tsx`). Do not stage them.

## Review Focus

1. **Games with features switched off.** Patterns off, powerups off, or both enemy types off: features stay finite and in range, with an empty weighted list and `enemy_stats` summing to 0. Covered by a test in Task 2.
2. **`spawn_rate_percent` other than 100.** The pattern countdown divides by the spawn rate, so at 50% the countdown starts twice as far away. Covered by a test in Task 2.
3. **Restarted lanes.** Rows for a restarted game come from that new game in both the nursery and the batch restart paths, never a stale row. Covered by a test in Task 3.
4. **Actor-only callers on a v3 model.** Evaluation, watch, death paths and pattern replay pass 3076-wide observations. These must be zero-padded rather than crash. Covered by a test in Task 5.
5. **Loading a v3 checkpoint in the dashboard.** The env must be built with `critic_features=True` before `DashboardPPO.load`, or SB3 rejects the space. Covered by a test in Task 6.

---

### Task 1: Shared selection helpers and RNG previews in dodge-core

**Files:**
- Modify: `../dodge-ai-3/native/crates/dodge-core/src/game.rs`. That means `update_pattern_schedule` (~2339), `spawn_enemies` (~1878) and `random_personality` (~1940), plus new accessors.

**Interfaces:**
- Produces, all `pub` on `NativeGame`:
  - `fn pattern_weights(&self) -> Vec<usize>`: the weighted index list `update_pattern_schedule` builds.
  - `fn pattern_countdown_frames(&self) -> Option<f64>`: `None` while a pattern is active or patterns are off.
  - `fn spawn_threshold(&self) -> PicoFixed`.
  - `fn preview_pattern_pick(&self) -> Option<usize>`.
  - `fn preview_enemy_roll(&self) -> (PicoFixed, i8)`: the max size and personality.
  - Read accessors for the fields Task 2 needs that have none yet: `enemy_timer`, `enemy_est`, `enemy_stats`, `bounce_cap_static`, `bounce_cap_moving`, `enemy_max_size`, `freeze_active`, `freeze_timer`, `size_timer`, `patterns_enabled`, `pattern_timer`, `pattern_delay_frames`, `spawn_rate`.

- [ ] **Step 1: Branch.** `cd ../dodge-ai-3 && git checkout -b critic-features`.

- [ ] **Step 2: Write the failing tests** in `game.rs`'s `mod tests`:

```rust
#[test]
fn preview_pattern_pick_matches_the_selection() {
    // hard, patterns + powerups on; play 300 frames of a fixed input first
    let mut game = /* existing test constructor */;
    game.pattern_timer = game.pattern_delay_frames - 1;
    let preview = game.preview_pattern_pick();
    let before = game.clone();
    game.update_pattern_schedule();
    assert_eq!(game.active_pattern, preview);
    assert!(preview.is_some());
    assert_eq!(before.preview_pattern_pick(), preview); // pure
}

#[test]
fn preview_enemy_roll_matches_the_spawn() {
    let mut game = /* same setup */;
    game.enemy_timer = game.spawn_threshold();      // next add crosses it
    let (max_size, personality) = game.preview_enemy_roll();
    let mut events = Vec::new();
    game.spawn_enemies(&mut events);
    let spawned = game.enemies.iter().last().unwrap();
    assert_eq!((spawned.max_size, spawned.personality), (max_size, personality));
}

#[test]
fn previews_do_not_touch_the_game() {
    let game = /* same setup */;
    let copy = game.clone();
    let _ = (game.preview_pattern_pick(), game.preview_enemy_roll(),
             game.pattern_weights(), game.spawn_threshold());
    assert_eq!(game, copy);
}
```

- [ ] **Step 3: Run them to see them fail.** Run `rustup run stable cargo test --manifest-path native/Cargo.toml -p dodge-core preview`. Expected: compile errors, because the functions don't exist yet.

- [ ] **Step 4: Implement.** Move the weighted-list loop out of `update_pattern_schedule` into `pattern_weights`, and the threshold expression out of `spawn_enemies` into `spawn_threshold`. Change the originals to call them. Split `random_personality` into a free function `pick_personality(stats: &[PicoFixed; 5], rng: &mut PicoRng) -> i8`. The previews clone `self.rng` and run the same draws in the same order: one draw over `weighted.len()`; for an enemy, the size roll and then `pick_personality`. If `spawn_enemies` with training options zeroes `enemy_stats` entries before rolling, the preview applies the same zeroing to its copy.

- [ ] **Step 5: Run the whole workspace.** `rustup run stable cargo test --manifest-path native/Cargo.toml --workspace --locked`. Expected: all pass, including the existing snapshot and parity tests.

- [ ] **Step 6: Commit.** Message: "Expose pattern and spawn schedules and RNG previews for a critic".

### Task 2: `encode_critic_features` and the Python binding

**Files:**
- Create: `../dodge-ai-3/native/crates/dodge-batch/src/critic.rs`
- Modify: `dodge-batch/src/lib.rs`, to add `mod critic` and `NativeBatch::observe_critic`
- Modify: `dodge-python/src/lib.rs`, to add `observe_velocity(critic=False)` and the module constants

**Interfaces:**
- Consumes: Task 1's accessors and previews.
- Produces:
  - Rust: `pub const CRITIC_FEATURE_COUNT: usize = 295`; `pub const CRITIC_GROUPS: [(&str, usize, usize); 6]` (name, start, stop, in the Design table's order); `pub fn encode_critic_features(game: &NativeGame, out: &mut [f32])`; `NativeBatch::observe_critic(&self, out: &mut [f32])` (lane-major, parallel like `observe_velocity`).
  - Python: `NativeBatchEnv.observe_velocity(critic: bool = False)`. With `critic=True` the dict gains `"critic"`: `float32[lanes, CRITIC_FEATURE_COUNT]`. Module attributes `dodge_native.CRITIC_FEATURE_COUNT: int` and `dodge_native.CRITIC_GROUPS: dict[str, tuple[int, int]]`.

- [ ] **Step 1: Write the failing tests** in `critic.rs`:

```rust
#[test] fn groups_tile_the_vector() {
    assert_eq!(CRITIC_GROUPS[0].1, 0);
    for pair in CRITIC_GROUPS.windows(2) { assert_eq!(pair[0].2, pair[1].1); }
    assert_eq!(CRITIC_GROUPS[5].2, CRITIC_FEATURE_COUNT);
}
#[test] fn encoding_is_a_pure_read() { /* encode at frame 500; game == clone */ }
#[test] fn every_feature_stays_in_range() {
    // 4 configs: (hard, patterns+powerups), (patterns off), (powerups off),
    // (normal and kamikaze both off); 2000 frames of seeded random input each;
    // every value finite and in [-1, 1]
}
#[test] fn pattern_countdown_ticks_once_per_frame() {
    // patterns on, no freeze, spawn_rate 100: over 50 frames before the first
    // pattern, the countdown feature falls by exactly 1/600 per frame
}
#[test] fn pattern_countdown_scales_with_spawn_rate() {
    // spawn_rate_percent 50: the frame-0 countdown is twice the 100% one (clamped at 1)
}
#[test] fn one_hots_are_one_hot() {
    // the active_pattern one-hot sums to pattern_active; the next_enemy size and
    // personality one-hots each sum to 1; each present nearest-enemy slot has a
    // personality one-hot summing to 1
}
#[test] fn nearest_enemies_are_sorted() { /* present slots' dx²+dy² non-decreasing */ }
```

- [ ] **Step 2: Run them to see them fail.** `rustup run stable cargo test --manifest-path native/Cargo.toml -p dodge-batch critic`. Expected: compile errors.

- [ ] **Step 3: Implement `encode_critic_features`** following the Design table, group by group in table order. Find each "over its maximum" constant where the game sets that timer. `SCORE_SCALE` is computed from `init_patterns` once, in a `OnceLock`.

- [ ] **Step 4: Run the tests.** Same command. Expected: pass.

- [ ] **Step 5: Add the Python binding.** Add the `critic` keyword to `observe_velocity`, which calls `observe_critic` only when it is true, and add the two module attributes. Then build and install: `cd native/crates/dodge-python && maturin build --release && pip install --force-reinstall ../../target/wheels/dodge_native-*.whl`.

- [ ] **Step 6: Smoke test from DodgeAI.** Run `python -c "import dodge_native as d, numpy as np; from dodge.env import make_engine; e=make_engine(difficulty=3,patterns=True,powerups=True); e.reset_ml_batch(np.arange(4,dtype=np.uint32)); s=e.observe_velocity(critic=True); print(s['critic'].shape, d.CRITIC_FEATURE_COUNT, list(d.CRITIC_GROUPS))"`. Expected: `(4, 295) 295 ['clock', 'timers', 'active_pattern', 'next_pattern', 'next_enemy', 'nearest_enemies']`.

- [ ] **Step 7: Run everything.** Workspace tests, plus `cargo fmt --check` and clippy as in the `justfile`. Then commit: "Encode critic-only game state per lane".

### Task 3: `DodgeVecEnv(critic_features=True)`

**Files:**
- Create: `dodge/critic.py`
- Modify: `dodge/env.py` (`DodgeVecEnv.__init__`, `_start_all`, `_fresh_games`, `step_wait`, `config`)
- Test: `tests/test_critic_env.py`

**Interfaces:**
- Produces:
  - `dodge/critic.py`: `CRITIC_FEATURE_COUNT: int` (0 when `dodge_native` lacks it); `CRITIC_GROUPS: dict[str, tuple[int, int]]`; `WIDE_OBSERVATION_SIZE = OBSERVATION_SIZE + CRITIC_FEATURE_COUNT`; `CRITIC_ARCHITECTURES = ("velocity-flow-v3",)`.
  - `DodgeVecEnv(..., critic_features: bool = False)`. When true: `observation_space` is `Box(-1, 1, (WIDE_OBSERVATION_SIZE,), float32)`; every returned observation and every `terminal_observation` is `[observation | critic]`; `game_settings["critic_features"]` records it.

- [ ] **Step 1: Branch.** In DodgeAI: `git checkout -b critic-features`.

- [ ] **Step 2: Write the failing tests:**

```python
pytestmark = pytest.mark.skipif(not native_available(), reason="needs dodge_native")
SETTINGS = dict(difficulty=3, patterns=True, powerups=True, action_repeat=1)

def test_actor_columns_match_the_plain_env():
    plain = DodgeVecEnv(n_envs=4, seed=7, **SETTINGS)
    wide = DodgeVecEnv(n_envs=4, seed=7, critic_features=True, **SETTINGS)
    a, b = plain.reset(), wide.reset()
    rng = np.random.default_rng(0)
    for _ in range(1500):
        assert np.array_equal(b[:, :OBSERVATION_SIZE], a)
        actions = rng.integers(0, 9, 4)
        a, ra, da, _ = plain.step(actions)
        b, rb, db, ib = wide.step(actions)
        assert np.array_equal(ra, rb) and np.array_equal(da, db)
        for info in ib:
            if "terminal_observation" in info:
                assert info["terminal_observation"].shape == (WIDE_OBSERVATION_SIZE,)

@pytest.mark.parametrize("mode", ["nursery", "batch"])
def test_restarted_lane_carries_its_new_games_features(mode):
    # force env._restart_observe = mode; step until a lane dies; its new row's
    # critic columns equal observe_velocity(critic=True)["critic"] of a fresh
    # make_engine reset on that lane's new seed (env.game_seeds[lane])

def test_observation_space_is_wide():
    env = DodgeVecEnv(n_envs=2, critic_features=True, **SETTINGS)
    assert env.observation_space.shape == (WIDE_OBSERVATION_SIZE,)
    assert env.game_settings["critic_features"] is True
```

- [ ] **Step 3: Run them to see them fail.** `pytest tests/test_critic_env.py -v`. Expected: `TypeError: unexpected keyword 'critic_features'`.

- [ ] **Step 4: Implement.** Add one private method, `_observation(state) -> np.ndarray`: it returns `state["observation"]`, or that array with `state["critic"]` appended horizontally. Every `observe_velocity()` call in the env passes `critic=self.critic_features`, the nursery engine's included. Each place that reads `state["observation"]` reads `self._observation(state)` instead.

- [ ] **Step 5: Run the tests.** `pytest tests/test_critic_env.py tests/test_env_parity.py -v`. Expected: pass.

- [ ] **Step 6: Commit.** Message: "Let DodgeVecEnv append critic-only game state to the observation".

### Task 4: Probe whether the features predict returns (go/no-go gate)

**Files:**
- Create: `tools/critic_probe.py`
- Test: `tests/test_critic_probe.py`

**Interfaces:**
- Consumes: `DodgeVecEnv(critic_features=True)`, `CRITIC_GROUPS`.
- Produces:
  - `discounted_returns(rewards: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray`, with shape (steps, lanes); the tail that has no terminal is returned as NaN for its last `ceil(5 / (1 - gamma))` steps.
  - `probe_r2(train: dict, test: dict, columns: slice | None) -> float`.
  - CLI: `python tools/critic_probe.py MODEL.zip --steps 20000 --envs 32`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_discounted_returns_stop_at_terminals():
    r = np.array([[1.0], [1.0], [1.0]]); d = np.array([[0], [1], [0]])
    g = discounted_returns(r, d, 0.5)
    assert g[0, 0] == 1 + 0.5 * 1 and g[1, 0] == 1
    assert np.isnan(g[2, 0])

def test_probe_finds_a_planted_signal():
    # residual = 3 * feature[10]; with columns covering 10 the R² gain > 0.5,
    # and with columns=None (V only) the gain is ~0
```

- [ ] **Step 2: Run them to see them fail.** `pytest tests/test_critic_probe.py -v`. Expected: ImportError.

- [ ] **Step 3: Implement.** Load the model with `DashboardPPO.load` and no env. Play `--steps` steps on the wide env (hard, patterns and powerups on, `action_repeat` 1, `--envs` lanes), feeding `obs[:, :OBSERVATION_SIZE]` to `model.policy`, sampling stochastically. Record V, rewards, dones and the critic columns. Use the model's `gamma`. Split by lane: 3/4 of lanes train, 1/4 test.

  The probe is a 2-layer MLP (64 hidden, LeakyReLU, Adam 1e-3, 30 epochs) that predicts G − V from `[V, features[columns]]`. Report the test-set explained variance of G, using V alone and V plus the probe. Do this for all features together and for each group in `CRITIC_GROUPS`, one line per group, plus the baseline.

- [ ] **Step 4: Run the tests.** Expected: pass.

- [ ] **Step 5: Run the probe** on `artifacts/models/velocity-flow-v2.zip` and save the output to `docs/superpowers/plans/2026-11-01-critic-probe-result.txt`.

  **Gate:** continue to Task 5 only if "all groups" raises held-out explained variance by at least 0.02 over V alone. Otherwise stop and report. Groups that add less than 0.002 on their own are listed for removal and decided with the user before Task 5.

- [ ] **Step 6: Commit** the tool, its test and the result file. Message: "Probe whether critic-only game state predicts returns".

### Task 5: The `velocity-flow-v3` architecture

**Files:**
- Modify: `dodge/velocity.py`. Refactor `VelocityFlowV2Extractor.forward` (lines 464-481), and add `VelocityFlowV3Extractor` and `VelocityFlowV3Policy`.
- Modify: `dodge/policies.py`, in `ARCHITECTURES`, `EXTRACTOR_ARCHITECTURE`, `policy_class` and `policy_kwargs`.
- Modify: `dodge/training.py`, in `GPU_ARCHITECTURES` and the hyperparameter table (v3 maps to v2's entry).
- Test: `tests/test_velocity_v3.py`

**Interfaces:**
- Consumes: `WIDE_OBSERVATION_SIZE` and `CRITIC_FEATURE_COUNT` from `dodge/critic.py`.
- Produces:
  - `VelocityFlowV2Extractor._danger_and_value_inputs(observations) -> tuple[Tensor, Tensor]`. `forward` becomes `cat(-danger, value_net(inputs))`, numerically identical to before.
  - `VelocityFlowV3Extractor(observation_space, action_repeat=4)`. It requires `observation_space.shape == (WIDE_OBSERVATION_SIZE,)` and half resolution. `value_net[0]` is a `Linear` whose input width is v2's plus `CRITIC_FEATURE_COUNT`, with the new columns zeroed at construction. `field(observations)` slices to the first `OBSERVATION_SIZE` values.
  - `VelocityFlowV3Policy(VelocityFlowPolicy)`. Its `obs_to_tensor` zero-pads numpy observations whose last axis is `OBSERVATION_SIZE` up to `WIDE_OBSERVATION_SIZE`.
  - The architecture name `"velocity-flow-v3"`, appended to `ARCHITECTURES` (the default stays first).

- [ ] **Step 1: Write the failing tests:**

```python
def _policy():
    space = spaces.Box(-1, 1, (WIDE_OBSERVATION_SIZE,), np.float32)
    return VelocityFlowV3Policy(space, spaces.Discrete(9), lambda _: 3e-4,
                                **policy_kwargs("velocity-flow-v3"))

def test_actor_ignores_critic_columns():
    policy = _policy()
    with torch.no_grad():  # make the zero-initialised critic columns live
        policy.features_extractor.value_net[0].weight[:, -CRITIC_FEATURE_COUNT:].normal_()
    obs = torch.rand(16, WIDE_OBSERVATION_SIZE) * 2 - 1
    other = obs.clone(); other[:, OBSERVATION_SIZE:] = torch.rand(16, CRITIC_FEATURE_COUNT)
    a, va = policy._distribution_inputs(obs)
    b, vb = policy._distribution_inputs(other)
    assert torch.equal(a, b) and not torch.equal(va, vb)

def test_new_critic_columns_start_at_zero():
    w = _policy().features_extractor.value_net[0].weight
    assert torch.count_nonzero(w[:, -CRITIC_FEATURE_COUNT:]) == 0

def test_predict_accepts_actor_only_observations():
    policy = _policy()
    narrow = np.random.default_rng(0).uniform(-1, 1, (4, OBSERVATION_SIZE)).astype(np.float32)
    wide = np.concatenate([narrow, np.zeros((4, CRITIC_FEATURE_COUNT), np.float32)], 1)
    assert np.array_equal(policy.predict(narrow, deterministic=True)[0],
                          policy.predict(wide, deterministic=True)[0])

def test_v2_forward_unchanged_by_refactor():
    # tests/test_velocity_trunk.py's frozen route still passes; also a v2
    # extractor's forward equals cat(-danger, value_net(inputs)) from the helper
```

- [ ] **Step 2: Run them to see them fail.** `pytest tests/test_velocity_v3.py -v`. Expected: ImportError.

- [ ] **Step 3: Implement** to the interfaces above. `policy_kwargs("velocity-flow-v3")` is v2's dict with `features_extractor_class=VelocityFlowV3Extractor`.

- [ ] **Step 4: Run the tests.** `pytest tests/test_velocity_v3.py tests/test_velocity_trunk.py tests/test_policies.py -v`. Expected: pass.

- [ ] **Step 5: Commit.** Message: "Add velocity-flow-v3: v2's actor, a critic that also reads hidden game state".

### Task 6: Train v3 in the dashboard and continue it from v2

**Files:**
- Modify: `dodge/training.py`, in `_run` (~981): choose `critic_features` before building the env.
- Create: `tools/continue_with_critic.py`
- Test: `tests/test_critic_training.py`

**Interfaces:**
- Consumes: Tasks 3 and 5.
- Produces:
  - `checkpoint_architecture(path) -> str` in `dodge/training.py`: reads `policy_kwargs` via `load_from_zip_file(path, device="cpu")` and returns `architecture_from_kwargs`.
  - `_run` sets `env_kwargs["critic_features"] = arch in CRITIC_ARCHITECTURES`, where `arch` is the checkpoint's architecture when loading and `self.architecture` otherwise.
  - `continue_with_critic(source: Path, dest: Path) -> None`. It builds a v3 `DashboardPPO` on a 1-lane critic env and copies every policy tensor from the v2 source by name. For `features_extractor.value_net.0.weight` it copies the leading columns and leaves the critic columns at zero. It carries over `num_timesteps`, `_n_updates` and `completed_updates`, then saves to `dest`. The optimizer restarts. CLI: `python tools/continue_with_critic.py SRC.zip DEST.zip`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_continuation_reproduces_v2(tmp_path):
    # save a small fresh v2 DashboardPPO to tmp_path/"v2.zip"; continue_with_critic
    # it to "v3.zip"; load both policies; on 32 random wide observations, v3's
    # log_pmf and value equal v2's (fed the first 3076 values) to 1e-6, whatever
    # is in the critic columns

def test_session_builds_a_wide_env_for_a_v3_checkpoint(tmp_path):
    # PPOTrainingSession(architecture=None, autostart=False, n_envs=2, ...);
    # pending_load = the v3 zip; run _run's env-construction step (factor it
    # into _env_for(load_path) -> DodgeVecEnv if that is what makes it testable)
    # and assert env.observation_space.shape == (WIDE_OBSERVATION_SIZE,) and
    # that _load_model(path, env) returns a model without raising

def test_v2_checkpoint_still_gets_a_plain_env(tmp_path): ...
```

- [ ] **Step 2: Run them to see them fail.** `pytest tests/test_critic_training.py -v`. Expected: ImportError.

- [ ] **Step 3: Implement** to the interfaces above.

- [ ] **Step 4: Run the tests.** `pytest tests/ -q`. Expected: the full suite passes.

- [ ] **Step 5: Make the starting point.** Run `python tools/continue_with_critic.py artifacts/models/velocity-flow-v2.zip artifacts/models/velocity-flow-v3.zip`.

- [ ] **Step 6: Commit.** Message: "Train velocity-flow-v3 from the dashboard and continue it from v2". Do not commit the generated `.zip`.

### Task 7: A/B against a v2 control (run by the user, not code)

- [ ] **Step 1: Run two continuations from the same `velocity-flow-v2.zip`** with identical settings (`training.json`: lr 3e-5, γ 0.997) and the same number of updates. Control: v2 as it is. Treatment: `velocity-flow-v3.zip`.
- [ ] **Step 2: Compare.**
  - Explained variance over time. Expect v3 to rise above v2's ~0.70–0.75.
  - The dev-cohort score.
  - The `docs/paper/eval` harness: hard difficulty, 1-frame decisions, argmax, 512 seeds; report mean and Q1.
- [ ] **Step 3: Read the result.**
  - Explained variance up but the policy no worse: the critic is better. Give it longer.
  - Explained variance up and the policy worse: suspect the shared trunk losing value-loss gradient (the concern raised in conversation). Next experiment: add a second, auxiliary value head that reads only the trunk (no critic columns) and train it on the same returns. The trunk then gets the full value signal again, while PPO's advantages still come from the head that sees the critic features.
  - Explained variance flat: drop the groups the probe found weakest.
