# Dodge — local playable build and training environment

**Date:** 2026-08-11
**Status:** approved design, phase 1 not yet implemented

## Background

`logic.js` is the PICO-8 game `dodge.p8` exported to HTML5 via Emscripten. About 97% of
its 4.3 MB is the compiled PICO-8 runtime; the game itself is a 32,768-byte ROM stored in
the `_cartdat` array on line 5.

`tools/extract_cart.py` pulls that ROM out and decompresses its PXA-compressed code
section, producing:

- `dodge.lua` — 2,263 lines of PICO-8 Lua, the game's real source
- `dodge.p8` — the full cartridge in PICO-8's plain-text format

The cart uses no map and no sprite flags, so all world state lives in Lua tables.

## Goal

Two deliverables, in order:

1. A **local playable copy** of the game that runs without PICO-8 or a browser.
2. A **training environment** exposing internal game state — never pixels — so an agent
   can be trained without solving vision first.

The user does not own a PICO-8 licence. Both a headless mode and a watchable rendered
mode are required.

## Approach

Host the real, unmodified `dodge.lua` in LuaJIT via `lupa`, and implement the PICO-8 API
around it in Python.

This was chosen over a hand-port to Python because it is faithful by construction. A
survey of the cart shows the surface area is small: **38 PICO-8 builtins are used, only 14
of them drawing-related**, and the dialect deviations from standard Lua are bounded (86
compound assignments, 31 `!=`, 28 one-line `if (cond) stmt`, 2 `?` print shorthands, 2
fixed-point binary literals).

A native Python port remains open as a later optimisation if throughput demands it. This
environment then serves as the reference oracle to diff it against, frame by frame.

`lupa 2.8` ships a prebuilt `cp312-win_amd64` wheel, so no C compiler is required.

## Architecture

```text
dodge.lua                    unmodified game (regenerable from logic.js)
│
pico8/preprocess.py          PICO-8 Lua dialect → standard Lua 5.1
pico8/api.py                 the 38 builtins, seeded PRNG, 128-byte GPIO memory
pico8/backends.py            NullBackend (no-op draws) │ PygameBackend (real draws)
pico8/machine.py             boots the cart, steps one frame, reads globals
│
play.py                      phase 1: human plays with the keyboard
│
dodge/observe.py             phase 2: all internal-state reads live here
dodge_env.py                 phase 2: Gymnasium Env
watch.py                     phase 2: run a policy through PygameBackend
```

`dodge.lua` is never edited. It is treated as a build artefact of `extract_cart.py`.

### `preprocess.py`

A line-oriented text transform, not a parser. Handles:

- compound assignment (`+=`, `-=`, `*=`, `/=`, `%=`), including on table fields
- `!=` → `~=`
- one-line `if (cond) stmt` → `if cond then stmt end`
- `?expr` → `print(expr)`
- PICO-8 fixed-point binary literals (`0b1010010110100101.1`)

This is the component most likely to harbour subtle bugs, so it is tested directly
against snippets lifted from the real cart.

### `api.py`

Implements the 38 builtins the cart calls. Notable decisions:

- **Seeded PRNG.** `rnd`/`srand` are supplied by us so episodes are reproducible.
- **Persistence is local.** The game's `store`/`get_data` poke and peek `0x5f82+i` — GPIO,
  which in the shipped build crosses to `pico8_gpio` for the host page to persist. Backed
  here by a 128-byte `bytearray`, so settings and highscores work with no browser.
- **Audio is stubbed.** `sfx` and `music` are no-ops in phase 1. A PICO-8 synthesiser
  (waveforms, note effects, 4-channel pattern sequencer) is out of scope; the SFX and
  music data is present in the ROM if it is added later.

### `machine.py`

- **The frame entry point is `_draw()`, not `_update60()`.** `_update60` is empty in this
  cart; all game logic runs inside `_draw`. Stepping the simulation means calling `_draw`
  once.
- **Phase 2 boots straight into gameplay**, bypassing menus: set
  `_upd, _drw = updategame, drawgame`, write difficulty/patterns/powerups into the
  `settings` and `smallsettings` tables, then call the game's own `reset()`. Phase 1 boots
  normally so the menus are playable.
- **Particles are switchable.** `part` is written every frame but only ever *read* by
  `drawgame`, so under `NullBackend` particle work is skipped entirely — a speed win with
  no behavioural change.

### Rendering

`PygameBackend` draws to a 128×128 surface scaled up, using PICO-8's 16-colour palette.
The renderer is an approximation of PICO-8's output; the *simulation* is exact. Both
backends run identical Lua.

- **`pal(c, c2, 1)` must be honoured.** The game draws everything in colours 12 and 1 and
  remaps them at the end of every `_draw`; that is how the 13 themes work. Without it the
  game renders flat blue.
- **A 3×5 font is bundled.** PICO-8's font lives in `data.pod` inside the runtime, not in
  the cart. Without a font, score and every menu label are invisible.
- **`fillp` is approximated.** Used by `line2` to draw telegraphed hazards as dotted
  outlines. Cosmetic.

## Phase 1 — playable

**Deliverable:** `python play.py` boots the cart to its own menu, and the game is
playable with the keyboard.

Scope: `preprocess.py`, `api.py`, `PygameBackend`, `machine.py`, `play.py`.

The menus, settings screen, themes and highscore tracking require no extra work — they
are cart code, and a complete shim runs them.

**This phase validates the entire stack.** If the game boots, the settings screen works,
and play feels correct, the Lua shim is correct.

## Phase 2 — trainable

Built on a stack already known to work.

### Observation

Built by **rasterising entity geometry** — the same rectangles the game's own
`collisioncheck` uses — never by reading pixels. Works identically headless.

Three components, all selectable so any can be ablated:

**`grid`** — `(9, 32, 32)`, 4px cells over the 128×128 arena. One channel per entity type:

| Channel | Source | Note |
| --- | --- | --- |
| player | `px, py, size` | `size` drops to 2 with the shrink pickup |
| enemy normal | `e.p == 0` | |
| enemy kamikaze | `e.p == 1` | accelerates when near the player |
| kamikaze blast | `e.p == -1` | the expanding lethal square it leaves |
| pickup explode | `e.p == 2` | clears the screen |
| pickup freeze | `e.p == 3` | slows everything via `frz` |
| pickup shrink | `e.p == 4` | sets `size = 2` |
| hazard telegraphed | `r.sh < 2` | visible but harmless |
| hazard lethal | `r.sh == 2` **or** `cp.tpe == 1` | the second clause matters — moving-wall patterns are lethal the whole time and never telegraph |

**`nearest`** — the K nearest threats (default K = 8, zero-padded) as exact floats
(`dx, dy, vx, vy, size, type`). A
32×32 grid cannot express velocity or sub-cell position, and collisions are pixel-exact,
so this restores precision where it decides outcomes.

**`scalars`** — `dpx, dpy`, `size`, `frz`, `freeze`, `score`, and difficulty state
`spd`/`est`.

`dodge/observe.py` holds **every** read of internal state. The env and renderers consume a
documented dict and never touch Lua, so changing representations means editing one file.

`render_ascii()` prints the grid with one character per type (`@` player, `e` enemy,
`K` kamikaze, `*` explode, `F` freeze, `S` shrink, `#` lethal, `:` telegraph). It is
near-free once the grid exists, and is the fastest way to confirm the observation matches
reality.

### Actions

`Discrete(9)` — eight directions plus no-op — mapped onto `btn(0..3)`.

Keyboard control is the real game: `dpx *= 0.8` friction with `+= 0.5` per held direction,
so the agent steers a body with inertia.

**Mouse control is deliberately not supported.** Under mouse input the game does
`px, py = stat(32), stat(33)` — the player teleports to the cursor every frame with no
momentum, which makes it a degenerate control problem.

### Reward

Survival plus score: `+0.01` per frame survived, plus the delta in the game's own `score`
(`+0.5` per enemy destroyed, `+1` per pickup), and `-1.0` on death.

Two properties of this game shape that choice:

- **`score` drives difficulty.** Every point calls `difficultycurve()`, permanently ramping
  enemy speed and spawn rate. Rewarding score rewards making the game harder for yourself —
  an intentional tension, since it pushes the agent to engage rather than turtle.
- **Camping is already punished by the game.** Lingering at the centre of the arena — the
  60–68 box, adjusted by player size — for 50 frames triggers `fyou()`, spawning enemies at
  every spawn point at once. Pure survival reward will therefore not collapse into standing
  still.

### Episodes

Terminate on `isdead`. Truncate at a configurable frame cap (default 18,000 ≈ 5 minutes).
Reset calls the game's own `reset()` with a fresh seed rather than rebuilding Lua state,
so resets are cheap.

### Configuration

The cart's own settings are the curriculum knobs: `difficulty` (easy/normal/hard),
`patterns` on/off, `powerups` on/off. Patterns-off + powerups-off + easy is pure enemy
dodging; hard + everything is the full game. Also configurable: action repeat (default 1).

## Testing

- **`preprocess.py`** against snippets lifted from the real cart — all 28 one-line `if`s,
  both binary literals, compound assignment on table fields.
- **Determinism:** same seed + same action sequence → byte-identical trajectory hashes.
  This is the test that catches subtle shim bugs.
- **Observation correctness** (phase 2): inject an enemy at a known position into the Lua
  table, assert the correct grid cell in the correct channel lights up.
- **Boot smoke test:** 1,000 frames with no input, no Lua errors, the player eventually dies.
- `gymnasium.utils.env_checker` (phase 2), and a steps/sec benchmark.

## Error handling

- Lua errors surface as Python exceptions carrying the Lua traceback.
- Preprocessor failures fail loudly with the source line number.
- A missing `dodge.lua` instructs the user to run `tools/extract_cart.py`.
- Phase 2 asserts the machine is in the gameplay state before stepping.

## Dependencies

`lupa`, `numpy`, `pygame`. Phase 2 adds `gymnasium`.

## Out of scope

- Any training script or RL algorithm. The env is library-agnostic by design.
- PICO-8 audio synthesis.
- Driving the shipped browser build via Playwright. This was considered as a transfer-
  validation harness — the export exposes `pico8_gpio` and `pico8_buttons` bridges — but it
  is capped near real-time and is a poor trainer.

## Known limitations

- **Fixed-point drift.** PICO-8 uses 16.16 fixed-point arithmetic; we use doubles.
  Trajectories may diverge microscopically from the real cart.
- **Grid occlusion.** The observation grid cannot represent two entities stacked in one
  cell. The `nearest` float channel covers that case.
- **Renderer fidelity.** Font, `fillp` patterns and any audio are approximations or
  absent. Gameplay is unaffected.
