# AI Compatibility Contract

**Status:** normative. **Applies to:** any fork, rewrite, or extension of the
Dodge game that must keep working with the DodgeAI training stack
(`pico8/`, `dodge/`, `train_ppo.py`, `training_gui.py`, `watch_ppo.py`) and with
the `.zip` policy checkpoints already trained.

This document exists so a new game project can be developed *away* from this
repository and still drop into the existing trainer. It is written as a set of
rules against the code that enforces them, so every rule can be checked.

---

## 0. How to use this

1. Read §1 to understand what is actually coupled. It is less than you think,
   but it is not what you would guess.
2. Before designing any feature or game mode, check it against §12
   (change-cost tiers). Tier 0 is free. Tier 2 kills every checkpoint you own.
3. Before shipping, run §15 (conformance checklist).
4. If you must break something, use §14 (migration recipes) rather than
   improvising — some breaks are silent.

The word **MUST** means the trainer or a trained policy breaks without it.
**SHOULD** means it degrades learning or transfer but still runs.
**MAY** means genuinely free.

---

## 1. What the AI is coupled to

The AI does **not** look at your screen. It never sees a pixel. Rendering is a
no-op during training (`NullBackend.draws = False`, which binds every drawing
builtin to a Lua no-op in `pico8/api.py`).

The AI is coupled to five things, in descending order of how expensive they are
to break:

| # | Coupling | Enforced by | Break cost |
|---|---|---|---|
| 1 | **The shape of the observation** — 3076 floats | `dodge/observe.py` | Every checkpoint dies |
| 2 | **The names and layout of Lua game state** | `dodge/observe.py`, `dodge/gradient.py` | Trainer crashes at step 1 |
| 3 | **The player movement model** | `dodge/gradient.py:simulate_player` | Silent: the policy mis-aims |
| 4 | **The set of Lua functions that get monkey-patched** | `pico8/machine.py` | Trainer crashes at reset |
| 5 | **The PICO-8 runtime subset** | `pico8/api.py`, `pico8/prelude.lua` | Cart fails to boot |

Everything else — art, sound, menus, HUD, scoring presentation, palettes,
particles, screen shake, difficulty text, achievements — is invisible to the AI
and free to change.

---

## 2. Layer 1 — The runtime: what Lua you may write

The game runs as **PICO-8 Lua hosted on LuaJIT**, not on real PICO-8. The
runtime is partial. Using a builtin that is not implemented is a boot failure,
not a graceful degradation.

### 2.1 Cartridge delivery

- The game MUST be loadable by `pico8/cart.py:read_rom` — a `.p8.png`
  cartridge or a PICO-8 HTML5 `.js` export. Both legacy and PXA code
  compression are supported.
- The Lua source MUST survive `pico8/preprocess.py`, which translates the
  PICO-8 dialect to Lua 5.1. It handles: button glyphs (left/right/up/down/O/X
  → 0..5), `!=` → `~=`, `0b` binary literals, `?` print shorthand, compound
  assignment (`+= -= *= /= %=`), and single-line `if (cond) body`.
- Anything else in the PICO-8 dialect is **not** translated and must not be
  used. In particular there is no rule for `\` integer division, and the
  short-if rule only fires when the condition and body sit on one line — a
  short-if whose condition wraps is left as an ordinary `if` and fails to
  compile.

### 2.2 Builtins that exist

From `pico8/prelude.lua` (pure Lua):
`add del all foreach count mid sgn sin cos tostr srand rnd flr ceil abs min max
sqrt sub`

From `pico8/api.py` (Python-backed):
`btn btnp stat peek poke time t dget dset cls pset rect rectfill circfill
ovalfill line spr print pal fillp camera flip`

No-ops (callable, do nothing): `cartdata menuitem cursor sfx music printh palt`

`stat(n)` returns a value only for `n = 32, 33, 34` (mouse x, y, buttons).
Everything else returns `0`.

### 2.3 Builtins that do NOT exist

Using any of these is a crash:

`atan2  tonum  chr  ord  split  sget  sset  fget  fset  mget  mset  map  sspr
circ  oval  clip  color  pget  peek2  peek4  poke2  poke4  memcpy  memset
reload  cstore  serial  extcmd  tline`, and the two-player form `btn(i, p)`.

If a new feature needs one of them, the correct move is to **add it to
`pico8/prelude.lua` or `pico8/api.py` in the AI project**, not to work around it
in game code. That is a Tier 0 change (§12).

### 2.4 Determinism

- `srand` is Lehmer / Park-Miller, **not** PICO-8's generator. Sequences differ
  from real PICO-8. This is fine, but it means "it plays the same in PICO-8" is
  not a valid test of trainer behaviour.
- The game MUST be fully deterministic given the seed passed to
  `Machine(..., seed=N)`. Any non-determinism (wall-clock, table iteration order
  affecting gameplay) makes advantage estimates noisy and makes `watch_ppo.py`
  unreproducible.
- `time()` is `frame / 60.0`, driven by the frame counter — safe to use.
- `dget`/`dset` write to an in-memory 64-float array that is **discarded when
  the Machine is rebuilt**. Persistent high scores across episodes are not a
  thing during training. Do not build mechanics on cartdata persistence.

---

## 3. Layer 2 — The frame loop

`pico8/machine.py:Machine.step()` is the entire game clock:

```python
self.frame += 1
self._update_btnp()
if self.globals._update60 is not None:
    self.globals._update60()
self.globals._draw()
self.backend.flip()
```

- The game **MUST** define `_update60`. A cart that defines only `_update`
  (30 fps) will **never advance** — `_update60` is nil, so only `_draw` runs.
  This fails silently as an agent that appears frozen.
- The game **MUST** define `_draw`. It is called unconditionally; a nil `_draw`
  is a crash every frame.
- `_init` is optional but is called once at construction if present.
- One `step()` is exactly one game frame at 60 Hz. The environment's decision
  interval is `action_repeat` frames (default 4 → 15 decisions/second). Do not
  introduce internal frame skipping, a variable timestep, or a second update
  rate.

---

## 4. Layer 3 — Round entry

`Machine.start_game()` skips the menus. It requires this exact surface:

```lua
smallsettings          -- table; [2]=difficulty(1..3), [3]=patterns(bool), [4]=powerups(bool)
updategame, drawgame   -- the in-round update and draw functions
_upd, _drw             -- the dispatch slots _update60/_draw call through
reset()                -- starts a round
est                    -- seconds between enemy spawns (dynamic)
ed                     -- enemy spawn countdown accumulator
frz                    -- freeze factor (1 = normal time)
input                  -- 1 selects the keyboard movement branch
prevxpos, prevypos     -- mouse-mode position memory
```

Rules:

- `_update60` MUST dispatch through `_upd`, and `_draw` through `_drw`. The
  trainer sets `_upd, _drw = updategame, drawgame` to bypass the menu state
  machine. If you replace this with a different state machine, you MUST keep a
  way to jump straight into a round by assigning two globals, or you MUST
  update `start_game` in the AI project to match (Tier 1).
- `reset()` MUST leave the game in a playable first frame of a round.
- `input == 1` MUST select a keyboard/button movement branch with velocity and
  friction. `env.step()` **re-forces `input = 1` on every frame** — a game mode
  that requires mouse or analogue input is invisible to the agent.
- The trainer builds a **fresh `Machine` per cartridge run**, not `reset()` in a
  loop, precisely because the cart's own reset carries state across restarts.
  Keep `reset()` cheap, but do not rely on it being total.

### 4.1 Configurable dimensions the dashboard already exposes

These are wired into `training_gui.py`'s Game Config panel and
`TrainingSession.configure_game`. If your game keeps the same knobs, the panel
keeps working for free:

| Knob | Range | Reaches the game as |
|---|---|---|
| difficulty | 1, 2, 3 | `smallsettings[2]` |
| patterns | bool | `smallsettings[3]` |
| powerups | bool | `smallsettings[4]` |
| normal enemies | bool | `training_normal_enemies` (hook) |
| kamikaze enemies | bool | `training_kamikaze_enemies` (hook) |
| spawn rate | 50–150 % | `training_spawn_rate` (hook) |
| decision interval | 1–8 frames | `action_repeat` (env-side) |
| training lives | 0–5 | `training_lives` (hook) |

**A new game mode is best expressed as a new value in this table, not as a new
top-level concept.** A mode that is "difficulty 3, patterns on, normal enemies
off" costs nothing. A mode that changes the arena size costs everything (§12).

---

## 5. Layer 4 — The observation state contract

This is the contract. `dodge/observe.py:observe(machine)` reads Lua globals
directly, every decision. Every name below MUST exist, MUST be a number (or the
stated type), and MUST mean what it says here.

### 5.1 Player

| Lua global | Type | Meaning | Read by |
|---|---|---|---|
| `px`, `py` | number | Player position in arena pixels, 0..127 | `observe`, `gradient` |
| `dpx`, `dpy` | number | Player velocity, pixels per frame | `observe`, `gradient` |
| `size` | number | Player collision size in pixels | `gradient` |

`px`/`py` are normalised by `127.0` and clipped to `[0, 1]`.
`dpx`/`dpy` are divided by `VELOCITY_SCALE = 4.0` and clipped to `[-1, 1]`.

**MUST:** if the player can move faster than 4 px/frame in a new mode, the
velocity channels saturate and the agent goes blind to speed. Either cap the
player at ≤ 4 px/frame or raise `VELOCITY_SCALE` (Tier 1 — it rescales every
input a trained policy has ever seen).

### 5.2 World

| Lua global | Type | Meaning |
|---|---|---|
| `spd` | number | Global difficulty speed multiplier |
| `frz` | number | Freeze factor; 1 = normal time, 0 = frozen |
| `isdead` | truthy | The cartridge run is over |
| `score` | number | Current score |
| `enemies` | 1-based Lua array | Every mobile entity (see 5.3) |
| `cp` | table or nil | The currently active pattern, nil when none |
| `ftimer` | number | Frames spent inside the anti-camp box |
| `fy` | bool | Anti-camp check armed |

**MUST:** `enemies` is indexed `1..#enemies` and iterated by index, not by
`pairs`. Holes in the array truncate the agent's view of the world silently.

**MUST:** `frz` multiplies into every velocity the observation reports. Any new
time-dilation mechanic MUST go through `frz`, not through a separate
multiplier, or reported velocities become lies.

### 5.3 Entities (`enemies` elements)

Every element MUST carry:

| Field | Type | Meaning |
|---|---|---|
| `x`, `y` | number | Position in arena pixels |
| `vx`, `vy` | number | Direction/velocity components |
| `s` | number | Size in pixels (square) |
| `p` | number | Personality — see the table below |
| `spd` | number | Per-entity speed multiplier |

**Real displacement per frame is `vx * spd * frz * e.spd`** (`dodge.lua:1043`).
The observation computes exactly this. A new entity whose motion is applied some
other way will report a wrong velocity, and the danger field will aim at the
wrong future.

Personality codes are **frozen**:

| `p` | Entity | Observation cell type | Rect painted |
|---|---|---|---|
| `0` | Normal enemy | `ENEMY_NORMAL = 1` | `(x, y, s, s)` — top-left anchored |
| `1` | Kamikaze | `ENEMY_KAMIKAZE = 2` | `(x, y, s, s)` |
| `-1` | Kamikaze blast | `ENEMY_BLAST = 3` | `(x, y, s, s)` |
| `2` | Explode pickup | `PICKUP_EXPLODE = 4` | `(x-4, y-4, 8, 8)` — **centred** |
| `3` | Freeze pickup | `PICKUP_FREEZE = 5` | `(x-4, y-4, 8, 8)` |
| `4` | Shrink pickup | `PICKUP_SHRINK = 6` | `(x-4, y-4, 8, 8)` |

**MUST NOT** reuse or renumber these codes. `ENEMY_CELL` in `observe.py` maps
`p` → cell type, and an unknown `p` falls through to `ENEMY_NORMAL` — meaning a
new entity type silently masquerades as a normal enemy. That is the single
easiest way to poison a training run without any error appearing.

**MUST:** the anchor convention flips at `p >= 2`. Pickups are centred on
`(x, y)`; everything else is top-left anchored at `(x, y)`. Keep it, or fix
`cell_grid` in the AI project.

`enemiestats` is the spawn-weight array: index `i` is the weight for personality
`i - 1`, so `{normal, kamikaze, explode, freeze, shrink}`.

### 5.4 Patterns (`cp`)

When `cp` is not nil it MUST carry:

| Field | Type | Meaning |
|---|---|---|
| `tpe` | number | `1` = lethal from spawn; `0` = telegraphed, then lethal |
| `timer` | number | Frames since activation |
| `rects` | 1-based array | The pattern's rectangles |

Each rect MUST carry:

| Field | Type | Meaning |
|---|---|---|
| `x`, `y`, `w`, `h` | number | Axis-aligned rectangle in arena pixels |
| `sh` | number | Telegraph state; `2` means **lethal** |
| `dx`, `dy` | number | Unit direction (may be nil → treated as 0) |
| `spd` | number | Rate; real velocity is `dx * spd * frz` |
| `warnings` | 1-based array | Telegraph segments, each `{_, x1, y1, x2, y2}` |

Rules:

- A rect is lethal iff `cp.tpe == 1` **or** `rect.sh == 2`. Lethal rects paint
  `HAZARD_LETHAL = 8`; non-lethal ones paint `HAZARD_WARNING = 7`.
- `warnings` are painted only while `cp.timer < 125` (`WARNING_FRAMES`), which
  mirrors what the cart draws (`dodge.lua:1240`). If you change how long a
  telegraph is shown to a human, you MUST change `WARNING_FRAMES` to match, or
  the agent is told about a threat for a different length of time than the
  player is.
- Side-entering rects spend their whole telegraph parked **outside** the arena
  and are clamped away by the painter. The `warnings` segments are the only
  thing that tells the agent they are coming. **A new pattern that enters from
  off-screen MUST populate `warnings`** or it is invisible until it is already
  lethal.
- `patterns` MUST be a 1-based array holding the pattern tables, and the active
  `cp` MUST be identity-equal to one of its elements — the training-lives hook
  finds `cp` by scanning `patterns` for identity.
- `spawns` and `fy` MUST exist; the pattern-restart hook snapshots and restores
  them.

### 5.5 Paint priority

When two entities want the same 4 px cell, the higher rank wins the cell **and
supplies its velocity**:

```
HAZARD_LETHAL 6  >  ENEMY_BLAST 5  >  ENEMY_KAMIKAZE 4
              >  ENEMY_NORMAL 3  >  HAZARD_WARNING 2  >  any pickup 1
```

This exists because only two overlaps can persist in the game: a kamikaze blast
passes through other entities, and enemies sit inside a pattern's warning
telegraph. If a new mode creates a third kind of persistent overlap, add it to
`PAINT_PRIORITY` deliberately — the naive "highest ID wins" gets it wrong (it
ranked a pickup above a lethal blast, and a harmless warning above a lethal
enemy).

---

## 6. Layer 5 — Geometry

**All of these are frozen.** They are baked into the observation size, which is
baked into every checkpoint.

| Constant | Value | Consequence of changing |
|---|---|---|
| `ARENA_SIZE` | 128 | Observation size changes → all checkpoints unloadable |
| `CELL_SIZE` | 4 | Same |
| `GRID_SIZE` | 32 | Same |
| `CELL_COUNT` | 1024 | Same |
| `CELL_TYPES` | 8 | CNN input channels change → first conv unloadable |
| `PLAYER_VALUES` | 4 | Same |
| `OBSERVATION_SIZE` | **3076** | `4 + 3 × 1024` |
| `VELOCITY_SCALE` | 4.0 | Rescales every input a policy was trained on |
| `WARNING_FRAMES` | 125 | Agent's telegraph window desyncs from the player's |

Observation layout, in order:

```
[0]         px / 127                 clipped 0..1
[1]         py / 127                 clipped 0..1
[2]         dpx / 4                  clipped -1..1
[3]         dpy / 4                  clipped -1..1
[4:1028]    cell category / 8        32x32 row-major, category values 0..8
[1028:2052] cell vx / 4              32x32, velocity of the cell's occupant
[2052:3076] cell vy / 4              32x32
```

**The category channel budget is full.** IDs run 1..8 and `CELL_TYPES` is 8; the
extractor one-hots to `CELL_TYPES + 1 = 9` classes and drops the empty class,
giving exactly 8 hazard channels. There is no spare ID. A ninth entity type is a
Tier 2 change (§12) — and worse, it is an *insertion*, not an append: the CNN's
channel order is `[8 hazard | 1 player | 2 velocity]`, so a ninth hazard channel
shifts the player and velocity channels and defeats the append-only migration in
`training.py:migrate_widened_inputs`.

**Design consequence: new entity types must be reskins of existing categories,
not new categories.** A laser is a lethal hazard rect. A mine is a normal enemy
with `spd = 0`. A shield pickup is `p = 2/3/4` with different game logic behind
it. All free, all work today.

---

## 7. Layer 6 — Actions

`dodge/actions.py`. The action space is `Discrete(9)`:

```
0 idle   1 left   2 right   3 up   4 down
5 up-left   6 up-right   7 down-left   8 down-right
```

Each maps to a PICO-8 button mask over bits `LEFT=1, RIGHT=2, UP=4, DOWN=8`,
delivered via `machine.set_buttons(mask)` and read by the game through `btn()`.

Rules:

- The agent **never presses O (bit 4) or X (bit 5)**. Any mechanic gated behind
  those buttons — a dash, a bomb, a weapon, a mid-round confirm — does not exist
  as far as the AI is concerned. Design modes so that direction input alone is
  sufficient play, or extend the action space (Tier 2: the head width changes,
  and `velocity-flow` emits exactly `N_ACTIONS` field readings).
- `btnp` repeat is emulated (first repeat at 15 frames, then every 4). Do not
  build core movement on `btnp`.
- Buttons are held for the whole `action_repeat` window. There is no way for the
  agent to tap for one frame while `action_repeat > 1`.
- The mask is `& 0x3F`. Nothing above bit 5 reaches the game.

---

## 8. Layer 7 — Physics

`dodge/gradient.py` replicates the game's movement model **in Python**, and
`velocity-flow` — the default architecture — bakes the result into a frozen
lookup table (`action_offsets`) registered as a torch buffer and saved inside
every checkpoint.

The replicated constants:

```python
PLAYER_FRICTION      = 0.8    # dpx *= 0.8
PLAYER_ACCELERATION  = 0.5    # dpx += 0.5 per held direction
ARENA_MIN, ARENA_MAX = 2.0, 125.0   # position clamped, velocity NOT cleared
ENEMY_FRICTION       = 0.99
ENEMY_HOMING         = 0.01   # per frame, toward the player
BUTTON_DIRECTIONS    = ((-1,0), (1,0), (0,-1), (0,1))   # dirx/diry order
```

Per frame, in this order: `dpx *= 0.8`, then `dpx += 0.5` per held direction,
then `px += dpx`, then clamp `px` to `2..125` **without** zeroing `dpx`.

Rules:

- **MUST NOT** change the player movement model without regenerating
  `action_offsets` and retraining. This is the most dangerous break in the
  document because it is completely silent: the game runs, the trainer reports
  numbers, and the policy simply aims at positions the player never reaches.
- **Worse:** `offsets` is a persistent buffer in the policy `state_dict`. A
  loaded checkpoint restores the **old** offsets even after you fix the Python
  constants. Fixing the physics requires retraining or explicitly overwriting
  the buffer after load (§14).
- The clamp must not clear velocity. `simulate_player` relies on a player held
  against a wall retaining the speed it needs to leave.
- The anti-camp mechanic is part of the physics contract as far as the analytic
  controller is concerned: linger with the player box overlapping `60..68` on
  both axes for 50 frames and the game spawns an enemy at every spawn point,
  then resets the timer. `gradient.py` prices this explicitly because **it is
  invisible in the observation** — nothing about four enemies appearing says
  they were summoned. If a new mode changes or removes anti-camping, update
  `CAMP_BOX_MIN/MAX`, `CAMP_FRAMES`, and `camp_weight`, and keep `ftimer`/`fy`
  meaning what they mean now.

---

## 9. Layer 8 — Reward and termination

### 9.1 Termination

- `isdead` truthy ends the cartridge run. That is the only real terminal.
- `max_episode_frames` (default `None` — no cap) truncates the run, not the
  episode.
- Training lives (`training_lives > 0`) end the RL *episode* while the run
  continues, so the next episode starts from a mid-game state.

### 9.2 Score attribution

The reward function distinguishes score the agent caused from score it did not:

- The game credits **0.5** whenever an enemy dies to another enemy or to a
  pattern (`UNCONTROLLED_SCORE_PER_ENEMY = 0.5`). This is detected by hooking
  `difficultycurve(half)` — `half == true` means an enemy died without the
  player causing it.
- `uncontrolled_score = 0.5 × enemies_destroyed`, and
  `agent_score = score_delta − uncontrolled_score`. They are weighted
  separately.

**MUST:** if a new mode changes what score is worth, or adds a score source the
agent does not control, either keep the 0.5-per-uncontrolled-kill convention or
update `UNCONTROLLED_SCORE_PER_ENEMY` and the `difficultycurve` hook. Otherwise
the agent is paid for luck it cannot influence, which flattens the signal.

**MUST:** score must be monotonic within a run. `score_delta` is taken as a raw
difference; a mode that resets or decrements score feeds the agent negative
reward for events it did not cause.

### 9.3 Reward weights

`rewards.json` at the project root, re-read by every environment process at the
start of every episode. Fields, ranges, and defaults live in
`dodge/rewards.py:SLIDERS`:

| Weight | Range | Default | Meaning |
|---|---|---|---|
| `survival_per_frame` | 0 – 0.05 | 0.01 | Paid per frame survived, scaled by position |
| `edge_penalty` | 0 – 1 | 0.5 | Survival credit lost against a wall |
| `death_penalty` | 0 – 5 | 1.0 | Charged once, on death |
| `score_weight` | 0 – 2 | 1.0 | Multiplies agent-caused score |
| `uncontrolled_score_weight` | 0 – 1 | 0.0 | Multiplies luck score |

Survival credit uses `center_survival_scalar`: flat across the inner 60 % of the
half-arena, tapering to `1 − edge_penalty` at the walls. Adding a weight is a
Tier 0 change (add a dataclass field plus a `SliderSpec`); `load()` tolerates
unknown and missing keys, and `RENAMED` handles renames.

---

## 10. Layer 9 — The hook surface

`pico8/machine.py` layers training behaviour over the game by **replacing Lua
globals at runtime**. Every name below MUST remain a global function with the
stated signature, and MUST NOT be localised, inlined, or renamed.

### 10.1 Spawn controls (`_install_spawn_controls`)

| Name | Signature | Requirement |
|---|---|---|
| `addenemy` | `(x, y, isdying, size, max_size, personality)` | Global. **Every** creation path must go through it, including forced anti-camp waves. Personality is the 6th argument. |
| `spawnenemies` | `()` | Global. Must increment `ed` by exactly 1 per call, and spawn when `ed > est * 60 * spp * (2 - frz)`. |
| `updatepatterns` | `()` | Global. Must increment `cpt` by exactly 1 per call, and only while `dopatterns and cp == nil and not freeze`; a pattern starts when `cpt >= pt * 60`. |
| `enemiestats` | array | Index `i` is the spawn weight for personality `i - 1`. Zeroing an index must suppress that personality. |
| `ed`, `est`, `cpt`, `pt`, `dopatterns`, `freeze` | globals | Read and written by the hooks. |

The "+1 per call" property is load-bearing: the spawn-rate control works by
adding `rate - 1` on top of the game's own increment, which changes cadence
without touching the dynamic `est`, freeze, or special-pattern multipliers. A
game that increments `ed` by a variable amount breaks the spawn-rate slider
silently.

### 10.2 Event counting and training lives (`install_training_hooks`)

| Name | Signature | Requirement |
|---|---|---|
| `collide` | `(enemy)` | Global. Called on player↔entity contact. `enemy.p ∈ {2,3,4}` are pickups. |
| `collisioncheck` | `()` | Global. Wraps a full collision pass; may reach `die()` more than once for one large rect. |
| `die` | `()` | Global. The single death path. MUST be called from inside `collide` for enemy deaths and from inside `collisioncheck` for pattern deaths — the hook distinguishes them by which wrapper is on the stack. |
| `difficultycurve` | `(half)` | Global. `half == true` ⇒ an enemy died to something other than the player. |

The hook injects these globals:

```lua
training_events = { enemies_destroyed, explosion_pickups,
                    freeze_pickups, shrink_pickups, lives_spent }
training_lives  = <int>
```

**MUST NOT** define globals named `training_*` in game code — they get
overwritten.

**MUST:** `die()` must be the only way the run ends. Training lives work by
returning early from `die()`; a second death path bypasses that, and the run
ends when the environment expected a life to be spent.

**MUST:** pattern collisions must be survivable by restoring the pattern from a
deep copy of its activation state. That requires pattern tables to be plain data
plus functions, deep-copyable by `pairs` recursion, and requires `patterns`,
`spawns`, and `fy` to be the complete restart state. If a pattern gains state
stored anywhere else, training lives will restart it incorrectly.

---

## 11. Checkpoint compatibility

A saved `.zip` is a Stable-Baselines3 PPO checkpoint. It pins:

- `observation_space` = `Box(-1, 1, (3076,), float32)` — SB3 **refuses** to load
  against a different shape.
- `action_space` = `Discrete(9)`.
- Every layer shape, including field-trunk input channels
  (`HYBRID_INPUT_CHANNELS = 11`) and coordinate-branch input width
  (`COORDINATE_INPUTS = 10`).
- The frozen `offsets` and `sample_weights` buffers (§8).

Architectures, from `dodge/policies.py:ARCHITECTURES`: `velocity-flow-v2`,
`velocity-flow`, `hybrid-cnn`, `mlp`. Loading resolves the architecture from the
pickled `features_extractor_class` name and **imposes the architecture's
hyperparameters**, ignoring whatever the checkpoint was saved with.

Note: `velocity-flow` is fully convolutional and uses adaptive pooling, so its
*weights* are arena-size agnostic — only SB3's observation-space check and the
module-level constants pin it to 128×128. That is the one door left open if a
future mode genuinely needs a different arena.

---

## 12. Change-cost tiers

### Tier 0 — Free. Change at will.

- All art, sprites, palettes, particles, screen shake, HUD, fonts.
- All audio (`sfx`/`music` are no-ops).
- Menus, title screens, options screens, achievements, high-score display.
- Game feel that does not alter the movement model or entity kinematics.
- New patterns built from the existing rect vocabulary (`x,y,w,h,sh,dx,dy,spd`
  plus `warnings`).
- New enemy behaviour through existing personalities: different `spd`, different
  `vx/vy` steering, different `s`, different spawn placement.
- New pickups mapped onto `p = 2/3/4`.
- New difficulty tuning **within** `smallsettings[2] ∈ {1,2,3}`.
- New reward weights (a dataclass field plus a `SliderSpec`).
- Adding missing builtins to `prelude.lua` / `api.py`.

### Tier 1 — Costs a retrain, but checkpoints still load.

- Changing `VELOCITY_SCALE`, `WARNING_FRAMES`, or the anti-camp constants.
- Changing enemy kinematics (`ENEMY_FRICTION`, `ENEMY_HOMING`) — update
  `gradient.py` to match.
- Changing scoring conventions or `UNCONTROLLED_SCORE_PER_ENEMY`.
- Changing the `action_repeat` default — regenerate `action_offsets`.
- Adding a game-config knob (needs a matching hook plus a dashboard field).
- Restructuring the menu state machine (needs `start_game` updated).

### Tier 2 — Every existing checkpoint becomes unloadable.

- Changing arena size, cell size, or grid size.
- Adding a 9th entity category.
- Adding player observation values.
- Changing the action space (adding O/X, reordering, adding actions).
- Changing player movement physics.

Tier 2 is not forbidden. It is a decision to retrain from scratch, and it should
be made once, deliberately, for a batch of changes — not incrementally.

---

## 13. Designing game modes inside the contract

Worked examples, all Tier 0:

| Mode idea | Compliant implementation |
|---|---|
| **Boss fight** | A pattern (`cp`) with many rects and a long `timer`. Boss "health" is game-side bookkeeping the AI never needs to see. |
| **Endless / marathon** | `max_episode_frames = None` (already the default) and let difficulty ramp. |
| **Bullet hell** | Normal enemies (`p = 0`) with small `s` and high `spd`. Bullets are just fast small enemies. |
| **Laser grid** | Lethal pattern rects (`tpe = 1`), or telegraphed ones with `sh` ramping 0→2. |
| **Mines / static hazards** | `p = 0` with `vx = vy = 0`. The velocity planes report 0, which is exactly right. |
| **Time attack** | Score conventions unchanged; bound it with `max_episode_frames`. |
| **Shrinking arena** | Lethal hazard rects along the closing edges. Do **not** change `ARENA_SIZE`. |
| **Slow-motion powerup** | Drive it through `frz` so reported velocities stay honest. |
| **New pickup effect** | Reuse `p = 2/3/4`; the effect behind it is free. |
| **Waves / rounds** | Game-side sequencing; keep `isdead` as the only terminal. |

Modes that are **not** compliant without a Tier 2 decision:

- Anything requiring a button press (dash, shoot, bomb, block).
- Anything where the player is not a single point-like body with the 0.8/0.5
  movement model (vehicles, grappling, teleport, altered momentum).
- Multiple simultaneous player-controlled bodies.
- A scrolling camera, or a world larger than the 128×128 arena.
- Entities whose danger cannot be expressed as an axis-aligned rectangle.

---

## 14. Migration recipes

**Adding an observation input.** Append it. Player values go after `dpy`; grid
channels go after the existing hazard channels but before the player and
velocity channels — which is why a new *hazard* channel is not an append.
`training.py:needs_input_migration` plus `migrate_widened_inputs` copy an older
checkpoint into the leading slice of each widened layer and let the new
connections start from fresh initialisation. Progress (`num_timesteps`,
`_n_updates`) is preserved; optimizer state is discarded.

**Renaming a reward weight.** Add `old: new` to `dodge/rewards.py:RENAMED`.
Existing `rewards.json` files then carry their value over instead of silently
reverting to a default.

**Changing player physics.** Update the `gradient.py` constants, then either
retrain, or after loading each checkpoint overwrite the frozen buffer:

```python
extractor = model.policy.features_extractor
extractor.offsets.copy_(torch.as_tensor(action_offsets(action_repeat)))
```

Do not skip this — the saved buffer wins on load.

**Adding a Lua builtin.** Pure Lua → `pico8/prelude.lua`. Needs Python state →
`pico8/api.py:install`, and add it to `DRAWING_BUILTINS` if it only draws.

---

## 15. Conformance checklist

Run this against any new or extended game before pointing the trainer at it.

**Boot**
- [ ] `read_rom` loads the cartridge and `decompress_code` returns Lua.
- [ ] `preprocess(code)` produces Lua 5.1 that LuaJIT compiles.
- [ ] `Machine(rom, code, NullBackend(), seed=1)` constructs without error.
- [ ] No unimplemented builtin is referenced (grep the source against §2.3).

**Loop**
- [ ] `_update60` and `_draw` both exist as globals.
- [ ] 600 `step()` calls advance `frame` to 600 and never raise.
- [ ] Two Machines with the same seed produce identical `px, py, score` traces.

**Entry**
- [ ] `start_game(difficulty=d, ...)` for `d ∈ {1,2,3}` leaves the game in-round.
- [ ] `smallsettings[3]` / `[4]` actually gate patterns and powerups.
- [ ] `input = 1` selects keyboard movement.

**Observation**
- [ ] `observe(machine).shape == (3076,)` and `dtype == float32`.
- [ ] All values in `[-1, 1]`; no NaN, no inf.
- [ ] Every `p` value the game can produce is a key of `ENEMY_CELL`.
- [ ] With an entity on screen, its cell is non-zero, and its velocity plane is
      non-zero while it is moving.
- [ ] An off-screen incoming pattern paints `HAZARD_WARNING` cells via
      `warnings` before it becomes lethal.
- [ ] `cell_grid` velocity for a moving enemy equals `vx * spd * frz * e.spd`.

**Actions**
- [ ] All 9 actions produce distinct movement over 24 frames from centre.
- [ ] Core play reads no button above bit 3.

**Physics**
- [ ] `simulate_player(64, 64, 0, 0, mask, 24, held=4)` matches the live game to
      < 0.01 px on every action. (This repo's version matches to 0.000 px.)

**Hooks**
- [ ] `install_training_hooks(lives=3)` runs without error.
- [ ] With `lives=3`, three enemy collisions do not end the run and
      `training_events.lives_spent == 3`.
- [ ] A pattern collision with lives remaining restarts the pattern rather than
      ending the run.
- [ ] `_install_spawn_controls(normal_enemies=False, ...)` produces zero `p == 0`
      entities over 3600 frames.
- [ ] `spawn_rate = 1.5` produces measurably more spawns than `1.0`.
- [ ] `training_events.enemies_destroyed` increments only on non-player kills.

**Termination**
- [ ] `isdead` is set by exactly one code path, reachable only via `die()`.
- [ ] `score` never decreases within a run.

**End to end**
- [ ] `DodgeEnv(...).reset()` then 1000 `step()` calls, no exception.
- [ ] An existing `.zip` checkpoint loads and `model.predict` runs.
- [ ] `python train_ppo.py --steps 2000` completes.
- [ ] `python watch_ppo.py <checkpoint>` renders and the agent visibly dodges.

The existing test suite in `tests/` is the executable form of most of this —
`test_dodge_observe.py`, `test_dodge_env.py`, `test_dodge_actions.py`,
`test_machine.py`, and `test_gradient.py` in particular. **Port those tests
first.** They are the cheapest compliance check available, and several of them
encode facts that are not written down anywhere else.
