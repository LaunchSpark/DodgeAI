# Dodge — local playable build and training environment

**Date:** 2026-08-11
**Status:** approved design, phase 1 not yet implemented

## Background

`dodge.p8.png` is Artridge's public PICO-8 release of Dodge from Lexaloffle. It
stores a 32,768-byte ROM in the low bits of the cartridge PNG's pixels.

`tools/extract_cart.py` pulls that ROM out and decompresses its legacy-compressed code
section, producing:

- `dodge.lua` — 2,254 lines of PICO-8 Lua, the game's real source
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
dodge.lua                    unmodified game (regenerable from dodge.p8.png)
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

A paren-aware scanner that skips string literals and block comments. Not a full parser,
but more than line-oriented regex — see the constraints below. Handles:

- compound assignment (`+=`, `-=`, `*=`, `/=`, `%=`), including on table fields
- `!=` → `~=`
- one-line `if (cond) body` → `if cond then body end`
- `?expr` → `print(expr)`
- PICO-8 fixed-point binary literals (`0b1010010110100101.1`)
- **button glyph constants** → button numbers:
  `⬅️`=0, `➡️`=1, `⬆️`=2, `⬇️`=3, `🅾️`=4, `❎`=5

Four properties of the real cart constrain the implementation:

1. **Glyphs appear both as constants and as text.** 9 occurrences are bare identifiers
   (`btnp(❎)` at lines 16, 215, 316, 322; `btnp(🅾️)` at 24, 207; arrows at 253, 267, 285,
   286) and must become numbers. 5 occurrences are inside string literals (35, 411, 413,
   509, 511) — HUD text telling the player which button to press — and must be left alone.
   Substituting those would render "press 5 to play again". This also means **the bundled
   font must include the 6 glyph shapes**, or the menus lose their button prompts.
2. **A one-line `if` body extends to the end of the line and may hold several statements.**
   Line 2131 is `if (r.sh>1.99) r.sh=2 initspawns()` — two statements, both in the body.
3. **Conditions may nest parentheses.** Line 397 is
   `if ((isdead or not hasplayed) or n!="gameplay") print2(...)`, so the closing paren must
   be found by balance, not by first match.
4. **Conditions may span lines and end in `then`.** Lines 248 and 262 open `if (` and run
   six lines before closing with `then`. These are ordinary `if`s and must not be
   transformed. The rule that separates them: a short `if` has its opening paren matched on
   the *same physical line*, with non-`then` content after it. Both multi-line conditions
   leave their parens open at end of line, so the check is local.

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
  `settings` and `smallsettings` tables, then `hard_reset(seed)` (see Episodes — the
  cart's own `reset()` alone does not give independent episodes). Phase 1 boots normally so
  the menus are playable.
- **The full `_draw` always runs, including `drawgame`.** This is a correctness
  requirement, not a preference. `drawgame` calls `spawntrail` (line 457), which consumes
  two `rnd()` calls per powerup enemy per frame, and `shatter` consumes eleven per enemy
  death. Because the cart draws from a single RNG stream shared with gameplay, skipping any
  of that shifts every later `rnd()` — changing which enemy types spawn. So under
  `NullBackend` the drawing *primitives* become no-ops, but every function that can consume
  RNG still executes. This is what keeps headless and rendered runs identical.

  An optional `fast` mode may skip particle work for speed, and is documented as breaking
  cross-mode determinism. If profiling later shows it matters, the principled fix is to
  give cosmetic effects their own RNG stream rather than to skip them.

### Rendering

`PygameBackend` draws to a 128×128 surface scaled up, using PICO-8's 16-colour palette.
The renderer is an approximation of PICO-8's output; the *simulation* is exact. Both
backends run identical Lua.

- **`pal(c, c2, 1)` must be honoured.** The game draws everything in colours 12 and 1 and
  remaps them at the end of every `_draw`; that is how the 13 themes work. Without it the
  game renders flat blue.
- **Text is drawn as a scaled overlay.** PICO-8's font lives in `data.pod` inside the
  runtime, not in the cart, and glyphs are 3×5 px — unreadable if rendered at native scale
  with a system font. `print()` therefore records its calls during the frame, and they are
  drawn with a system monospace font onto the *upscaled* window after the 128×128 surface
  is blitted. Legible and cheap, at the cost of pixel authenticity. The 6 button glyphs are
  drawn as simple icons, since they appear in HUD strings like `"press ❎ to play again"`.
- **`fillp` is approximated.** Used by `line2` to draw telegraphed hazards as dotted
  outlines. Cosmetic.
- **The sprite sheet is loaded from the ROM.** `spr()` is called 11 times — the menu logo,
  the mouse cursor, the settings header, the button icons and the death art. `machine.py`
  passes the 128×128 sheet from ROM `0x0000–0x1fff` to the backend, decoded the same way
  `extract_cart.py` already decodes it (two pixels per byte, low nibble first). Without it
  phase 1 has no menus worth looking at.

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

**The cart's own `reset()` is not sufficient for independent episodes.** It deliberately
carries pattern history across restarts: it snapshots every `p.counter`, calls
`initpatterns()`, then recomputes each `p.prob` *from the carried counter* and adds `+2`.
Recently-played patterns therefore stay suppressed into the next run, and boot state never
equals post-reset state even when counters are zero. Left alone, this makes a seed's
behaviour depend on which episodes preceded it — a silent, hard-to-diagnose source of
non-reproducibility.

`machine.hard_reset(seed)` instead performs a fixed sequence every episode:

1. `srand(seed)`
2. zero `patterns[i].counter` for all patterns
3. `initpatterns()` — note this itself consumes RNG (line 1946 and the `range()` calls in
   specials 6 and 7), which is why it must sit at a fixed position in the sequence
4. restore each `p.prob` to its default (`(tpe==1 and 17 or 15) / flr(#p.var+1)`)
5. call the cart's `reset()`
6. reset the 128-byte GPIO memory, so highscore display does not vary between episodes

This runs as an injected Lua chunk owned by `machine.py`. `dodge.lua` stays unmodified.

### Configuration

The cart's own settings are the curriculum knobs: `difficulty` (easy/normal/hard),
`patterns` on/off, `powerups` on/off. Patterns-off + powerups-off + easy is pure enemy
dodging; hard + everything is the full game. Also configurable: action repeat (default 1).

## Testing

- **`preprocess.py`** against lines lifted verbatim from the real cart, chosen because each
  defeats a naive rule:

  | Line | Case |
  | --- | --- |
  | 285 | short-`if` + compound assignment + glyph constant (`if (btnp(⬇️)) cursi+=1`) |
  | 397 | nested parens in the condition — needs balance matching, not first-`)` |
  | 919 | compound assignment on a global inside a short-`if` body |
  | 1052 | fixed-point binary literal inside a short-`if` body |
  | 2131 | two-statement body (`r.sh=2 initspawns()`) |
  | 248 | **negative case** — multi-line condition ending in `then`, must be left unchanged |
  | 411 | **negative case** — glyph inside a string literal, must not become a number |

- **Determinism:** same seed + same action sequence → byte-identical trajectory hashes.
  This is the test that catches subtle shim bugs. It must include an
  **episode-independence** case: episodes A, B, A′ run in sequence, where A and A′ share a
  seed, must produce identical trajectories — this is what catches carried pattern state.
- **Backend parity:** the same seed and action sequence under `NullBackend` and
  `PygameBackend` must produce identical trajectories, including an episode that spawns
  powerups (which is where the `drawgame` RNG consumption bites).
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
- **Single RNG stream.** The cart draws gameplay and cosmetic randomness from one stream,
  so any change to how often visual effects run alters the game. This constrains what can
  be optimised away; see the `_draw` note under `machine.py`.
- **The preprocessor is correct for this cart, not for arbitrary PICO-8 source.** Three
  shapes are knowingly unhandled. All three are absent from `dodge.lua`, verified by
  grepping the whole file, and the cart is a frozen build artefact rather than something
  edited by hand — so the risk is accepted rather than designed around:
  1. Two compound assignments on one line produce overlapping edits.
  2. A compound assignment followed by more code on the same line has the trailing tokens
     swallowed into its right-hand side, because the RHS runs to end of line:
     `if a then c+=1 end` would become `if a then c = c + (1 end)`. Fixing this needs
     statement-terminator awareness (`end`, `then`, `;`).
  3. A short `if` whose body contains a nested long-form `if ... then ... end` is left
     untransformed, because the "no top-level `then` on this line" rule cannot tell that
     `then` belongs to a body statement rather than to the condition.

  A fourth was fixed rather than accepted, because it did occur: a condition built from
  several parenthesised groups (`if (a) or (b) then`, `dodge.lua:193`) was misread as a
  short `if`, since the first balanced paren is not the end of the condition.
