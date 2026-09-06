# Task 11 report: the playable entry point

## What I implemented

- **`play.py`** (new, repo root): the brief's `play.py` verbatim, plus one
  authorized addition — a `--frames N` option that runs the main loop for a
  fixed number of frames and then exits cleanly (`pygame.quit(); return 0`),
  instead of running until the window is closed. Default is `None`
  (unlimited), so normal interactive use (`python play.py`) is unchanged.
  `button_mask(pressed)` and the `KEYMAP` table are kept as plain,
  window-free functions/data, separable from `main()`.
- **`pico8/backends.py`**: added `pygame.event.pump()` to
  `PygameBackend.flip()`, immediately before `pygame.display.flip()`, exactly
  as authorized — this is the only change to that file, and it's the only
  otherwise-protected file touched.
- **`README.md`**: replaced the placeholder with the brief's content
  verbatim ("Play it", "Re-extract the cart", "Layout" sections).
- **`tests/test_play.py`** (new): unit tests for `button_mask()` covering
  every mapped key, both bound keys per button, OR-combination across
  buttons, unmapped keys, and the "both keys of one button held" case. No
  window or display is created for these tests.

## Verification — item by item

All of this was verified by execution (subprocess runs and direct Python
driving of `pico8.machine.boot`/`PygameBackend`), with `SDL_VIDEODRIVER=dummy`
throughout since this environment has no interactive display for me to
watch. I could not visually confirm rendering on an actual screen — only
that the correct pixel data was produced.

1. **`play.py --frames N` runs headless as a subprocess, exits 0, no
   traceback.** Verified. Ran `python play.py --frames 20 --seed 1` and
   `python play.py --frames 150 --seed 4 --scale 3` as real subprocesses
   under `SDL_VIDEODRIVER=dummy`; both exited with status 0, stderr was
   empty in both cases (checked stdout/stderr separately, not merged).

2. **The boot splash actually presented frames before the main loop
   began.** Verified. `dodge.lua` lines 49-72 run a `for i=0,65 do ...
   flip() end` loop from top-level code, i.e. during `boot()`, before
   `main()`'s loop or a single `machine.step()` ever runs. I subclassed
   `PygameBackend` to count `flip()` calls and confirmed exactly 66 calls
   happened during `boot()`, and that `machine.frame == 0` immediately
   after `boot()` returns (no `step()` has run yet) — i.e. the splash truly
   precedes the main loop, matching the ordering the brief and dispatch
   both call out as load-bearing.

3. **Reach the title screen; rendered surface is not blank (more than one
   colour).** Verified. `dodge.lua` line 47 sets `_upd,_drw =
   updatemenu,drawmenu` before the splash loop even runs, so the very first
   `machine.step()` after `boot()` renders the title screen. One `step()`
   produced 3 distinct colours on the 128x128 surface:
   `(29,43,83)`, `(41,173,255)`, `(255,241,232)`.

4. **`button_mask` produces the right bits for each mapped key.** Verified,
   both as pytest tests (`tests/test_play.py`, 11 tests, all passing) and by
   deliberately mutating `play.py` (`any` -> `all` in the OR-check) and
   confirming 8 of the 11 tests failed as expected, then reverting.

### Step 3's checklist (splash / title / X / momentum / O-settings), driven programmatically

Since I can't watch a window, I drove the actual cart's Lua state through
each beat using `machine.set_buttons()` — the same call `play.py`'s loop
makes every frame from `button_mask()` — and asserted on the cart's own
global state:

- **Splash, then title screen**: covered above (items 2-3).
- **X starts a round**: pressed button 5 (X) once from the title screen;
  within a few frames `globals._upd` became `updategame` (compared via
  `str()` on the underlying lupa `_LuaFunction`, since lupa hands back a
  fresh Python proxy per attribute access and `==` compares proxy identity,
  not the Lua function).
- **Momentum**: held right for 10 frames (`px`: 64.5 -> 80.07, monotonic
  increase), released, then stepped 6 more frames — `px` kept increasing
  for at least the first frame after release (81.86 -> 83.29 -> ... ->
  86.66), confirming the player keeps drifting rather than stopping dead.
- **O opens Settings**: pressed button 4 (O) once from the title screen;
  `globals._upd` became `updatesettings` within a few frames.
- **Left/right changes theme, colours change immediately**: from Settings,
  the cursor (`cursi`) defaults to 1, the theme row. Pressing right once
  changed `settings[1][1].current` from 1 to 2, and in that same step the
  rendered display palette changed: `display_pal[12]`: 12 -> 1,
  `display_pal[1]`: 1 -> 0 (these feed `pal(12, smallsettings[1][1], 1)` /
  `pal(1, smallsettings[1][2], 1)` in the cart's global `_draw()`, dodge.lua
  lines 2260-2261).
- **Difficulty / patterns / powerups also toggle** (called out in the
  plan's "Done when", not just Step 3): navigated the settings cursor with
  down/up taps and pressed right on each row; `settings[2][1].current`
  (difficulty), `settings[2][2].current` (patterns) and
  `settings[2][3].current` (powerups) each changed, and each change
  reached `smallsettings[2..4]`, which feed `dopatterns`/`powerups` and the
  movement-tuning constants in `_draw()`/`updatefyou()`.

What I could **not** verify: the actual visual appearance on a real
monitor (glyph legibility, animation smoothness, whether the CoolMathGames
splash "looks right" to a human, whether 60Hz timing feels good). Frame
content and state transitions are confirmed correct; subjective look-and-feel
is not.

### Whole-suite and reproducibility checks

- `python -m pytest tests/ -v`: **100 passed**, 1 warning (the
  known/pre-existing `pkg_resources` `DeprecationWarning` from
  `pygame/pkgdata.py` — not new, not touched).
- `python tools/extract_cart.py`: re-ran it; `git diff --stat dodge.lua
  dodge.p8` was empty both before and after, confirming byte-for-byte
  reproduction and that `dodge.lua` was never modified.

## TDD evidence

`tests/test_play.py` was written against the already-drafted `play.py`
(the brief's code was the starting point, and `button_mask` is copied
verbatim from it), so this wasn't a strict red-green-refactor from an empty
implementation. To still get real evidence the tests check real behaviour, I
mutated the implementation (`any` -> `all` in the per-button OR check) and
reran: 8 of 11 tests failed with clear assertion messages pointing at the
mutated line, then all 11 passed again after reverting. This confirms the
tests are not vacuous.

## Files changed

- `D:\projects\DodgeAI\play.py` (new)
- `D:\projects\DodgeAI\pico8\backends.py` (modified — `pygame.event.pump()`
  added to `PygameBackend.flip()`)
- `D:\projects\DodgeAI\README.md` (modified — brief's content)
- `D:\projects\DodgeAI\tests\test_play.py` (new)

## Self-review findings

- `play.py` matches the brief's code exactly except for the authorized
  `--frames` addition; no unrequested scope crept in (no config file, no
  extra CLI flags, no gamepad support, etc. — YAGNI held).
- `button_mask`/`KEYMAP` are free functions with no pygame-display
  dependency, so they're directly unit-testable, per the task's
  organization requirement.
- Naming: `frames_run` / `--frames` read clearly against the existing
  `--scale`/`--seed`/`--js` options; help text documents the default.
  No renames of anything in the protected files.
- No new warnings introduced; the one pre-existing `DeprecationWarning` is
  unchanged and not mine to fix per the dispatch.
- `pico8/backends.py` diff is a 5-line addition (4 comment lines + 1 call),
  nothing else touched in that file.
- Confirmed `dodge.lua`, `dodge.p8`, `pico8/preprocess.py`,
  `pico8/prelude.lua`, `pico8/api.py`, and `pico8/machine.py` are all
  untouched (`git diff --stat` shows nothing for any of them).

## Issues or concerns

None. This is the last task of the phase and I did not encounter anything
that needed escalation — no bug surfaced in the brief's `play.py` code
itself (unlike prior tasks in this project); the only corrections needed
were the two already authorized in the dispatch (`--frames`, the event
pump). The one genuine snag was in my own verification harness, not the
implementation: driving `btnp()`-style presses via `set_buttons()` requires
an explicit released frame between two presses of the same button, or the
second press is seen as a continued hold rather than a new press-edge. That
is correct PICO-8 `btnp()` semantics (already covered by
`tests/test_machine.py::test_btnp_repeats_after_15_frames_then_every_4`),
not a bug — I mention it only because it cost some back-and-forth while
writing the manual verification scripts.
