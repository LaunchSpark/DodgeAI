# Task 9: Pygame rendering — Report

## What I implemented

- `pico8/palette.py` — the 16 PICO-8 RGB triples (`PICO8`), verbatim from the brief.
- `pico8/backends.py` — appended `PygameBackend(Backend)`: a 128×128 8-bit
  indexed `pygame.Surface` that drawing calls write palette *indices* into
  (not RGB), a separate `draw_pal` (pal mode 0) and `display_pal` (pal mode
  1) table, camera offset, a sprite sheet built from `cart.sprite_sheet()`
  rows with colour 0 as the transparent colourkey, and a `flip()` that
  reapplies the display palette to the surface, scales, and blits to the
  window. `_draw_text` is a deliberate `pass` stub per the task scope (Task
  10 implements it); `text_queue` plumbing (`print_` appends, `flip` clears)
  is intact and untouched.
- `tests/test_pygame_backend.py` — the brief's 9 tests, plus one test I
  added for a bug I found (`test_camera_persists_across_cls`, see below).

Implemented from the brief essentially verbatim, with two corrections found
by direct experiment / code reading (both described below, neither was
guessed — both were verified against real behaviour before fixing).

## Corrections to the brief

### 1. Bug: `cls()` reset the camera offset (genuine gameplay-rendering bug)

The brief's `cls()` included `self.cam_x = self.cam_y = 0`. I traced how
`camera()` and `cls()` are actually used in `dodge.lua`'s `_draw()` (the
function `machine.step()` calls every frame, regardless of menu or
gameplay state):

```
camera(shakex,shakey)   -- line 2245
...
cls(12)                 -- line 2258
_drw()                  -- line 2259 (drawgame or drawmenu)
pal(12,smallsettings[1][1],1)   -- line 2260
pal(1,smallsettings[1][2],1)    -- line 2261
```

`camera(shakex,shakey)` is called *every single frame* (shake decays
toward 0 but is applied whenever a hit/shake is active), immediately
followed by `cls(12)` and then the actual draw call. Real PICO-8's `cls()`
does not touch the camera offset — only an explicit `camera()` call does.
If the backend's `cls()` reset the camera to (0,0), every screen-shake
effect in the game (triggered on hits, used in the pause/settings screen
too) would be silently discarded on the very next line, every frame it's
active — the camera offset applied one line earlier would never reach the
draw calls that use it.

This doesn't affect the parity test (camera is a pure rendering concern
that never consumes `rnd()`), which is why the brief's own test suite
couldn't have caught it. I removed the reset and added
`test_camera_persists_across_cls`, which fails on the brief's original code
(confirmed — see TDD Evidence) and passes with the fix.

**Fix:** removed `self.cam_x = self.cam_y = 0` from `cls()`; added a
comment explaining why, citing the exact `dodge.lua` line numbers.

### 2. Bug: `flip()`'s `pygame.transform.scale(..., self.window)` raises

Verified by direct experiment (see below) that scaling the 8-bit indexed
surface directly into the window surface (which pygame gives a 32-bit
format even under `SDL_VIDEODRIVER=dummy`) raises:

```
ValueError: Source and destination surfaces need the same format.
```

**Fix:** scale into a freshly-allocated surface (`pygame.transform.scale(self.surface, size)`,
no destination arg — this preserves 8-bit depth, verified), then
`self.window.blit(scaled, (0, 0))`, which does the format conversion. This
matches exactly what the dispatch instructions predicted might happen and
told me to verify rather than assume.

## What I tested and the results

Ran `python -m pytest tests/test_pygame_backend.py -v`: **10 passed**
(9 from the brief + 1 I added for the camera/cls bug).

Ran the full suite `python -m pytest -v`: **85 passed, 1 warning** (an
unrelated `pkg_resources` `DeprecationWarning` from pygame's own
`pkgdata.py`, nothing to do with this change). Output is pristine — no
"Windows fatal exception" dumps, no window opened (confirmed no visible
window process; `SDL_VIDEODRIVER=dummy` was in effect throughout, set at
test-module import time before any `pygame.init()`/`display.set_mode()`
call happens).

Tests genuinely ran, not skipped: `logic.js` is present in the working
tree (confirmed via `git status` / `Path("logic.js").exists()`), so the
`test_backends_produce_identical_trajectories` parity test and
`test_machine.py`'s cart-boot tests all executed against the real cart
rather than skipping.

## TDD Evidence

**RED** — before `pico8/palette.py` / `PygameBackend` existed:

```
$ python -m pytest tests/test_pygame_backend.py -v
ERROR collecting tests/test_pygame_backend.py
ImportError: cannot import name 'PygameBackend' from 'pico8.backends'
```

**GREEN** — after implementing `palette.py` and `PygameBackend`:

```
$ python -m pytest tests/test_pygame_backend.py -v
...
10 passed in 3.88s
```

**RED for the camera/cls bug fix** (temporarily reintroduced
`self.cam_x = self.cam_y = 0` into `cls()` to confirm the new test
actually catches the bug):

```
$ python -m pytest tests/test_pygame_backend.py -k test_camera_persists_across_cls -v
FAILED tests/test_pygame_backend.py::test_camera_persists_across_cls
AssertionError: assert (0, 0, 0) == (255, 0, 77)
```

**GREEN** after reverting to the fix:

```
$ python -m pytest tests/test_pygame_backend.py -v
...
10 passed in 2.78s
```

## Direct observation of the real cart rendering

Booted the machine with `PygameBackend` (not `NullBackend`), called
`start_game()`, and stepped real gameplay frames under
`SDL_VIDEODRIVER=dummy`, then inspected `backend.surface` pixel-by-pixel
with `pygame.Surface.get_at` (exact RGB reads, not visual screenshots,
which turned out to be misleading — see note below):

- **Palette indices are preserved, not flattened to one colour.** Over 900
  stepped frames (seed 21, sampled every 30 frames), the surface used
  exactly 3 distinct colours: PICO8[12] (background blue, ~481k px),
  PICO8[7] (white, ~8k px — sprites/UI), PICO8[1] (dark navy, ~2.3k px —
  outlines/shadow accents). More than one colour is present, ruling out
  the "flat blue" failure mode the task warns about; the exact 3-colour
  count also matches the brief's own description that "the cart draws the
  entire game in colours 12 and 1" (plus white for UI/text sprites) — this
  is the game's actual limited palette by design, not an under-rendering
  bug.
- **The palette-swap-at-flip-time design works end-to-end against the real
  cart.** `dodge.lua`'s `_draw()` calls `pal(12, smallsettings[1][1], 1)`
  every frame, where `smallsettings[1][1]` defaults to `12` (identity — no
  visible change). I forced a non-default theme by setting
  `m.globals.smallsettings[1][1] = 8` before stepping; after 30 frames the
  background pixel read back as `(255, 0, 77)` — exactly `PICO8[8]` (red).
  This confirms the real cart's own `pal(..., 1)` call is correctly wired
  through `Machine` → `api.py`'s `g.pal` lambda → `PygameBackend.pal` →
  `_apply_display_palette` → the surface's colour table, all the way to
  the rendered pixel, without the backend consuming any extra randomness
  (the parity test already covers that separately).
- **Sprites blit correctly with colour-0 transparency.** Letting a game
  actually run to death, the "GAME OVER" text (drawn via sprite blits, not
  `print()` — consistent with `_draw_text` being a stub in this task) and
  enemy sprites rendered correctly against the background, confirming
  `spr()`'s colourkey-based transparency works against real sprite-sheet
  data decoded by `cart.sprite_sheet()`, not just the synthetic sheets used
  in unit tests.
- **Caveat on visual screenshots:** I initially saved `pygame.image.save()`
  screenshots and visually inspected them via the image reader, which
  intermittently appeared to show an extra "peach" colour not present in
  an exact pixel Counter of the same surface at the same frame. Repeating
  the exact same run twice with an exact `Counter` (not a viewer-rendered
  image) gave identical, reproducible results with no peach — I attribute
  the visual artifact to the image viewer's own downscaling of a 512×512
  screenshot for display, not a rendering bug in the backend. I'm flagging
  this so it isn't mistaken for a bug: the pixel-level evidence (`Counter`,
  reproduced 3× identically) is what the "3 colours" and "palette remap"
  findings above are based on, not the screenshots.

## Files changed

- `d:\projects\DodgeAI\pico8\palette.py` (new)
- `d:\projects\DodgeAI\pico8\backends.py` (modified — appended `PygameBackend`)
- `d:\projects\DodgeAI\tests\test_pygame_backend.py` (new)

## Self-review findings

- Confirmed `_draw_text` stub and `text_queue` plumbing left exactly as
  specified (not implemented, not deleted).
- Confirmed no edits to `dodge.lua`, `dodge.p8`, `pico8/preprocess.py`,
  `pico8/prelude.lua`, `pico8/api.py`, `pico8/machine.py`.
- Confirmed no `git checkout`/`switch`/`branch`/`reset`/`rebase` was run;
  stayed on `design/dodge-env` throughout.
- `import pygame` sits mid-file in `backends.py` (after `NullBackend`,
  before `PygameBackend`) rather than at the top, matching the brief's
  "append to backends.py" instruction verbatim. This is a minor style
  deviation from PEP 8 top-of-file imports but not a correctness issue —
  `pygame` is already an unconditional project dependency
  (`requirements.txt: pygame>=2.5`), so `pico8/machine.py`'s existing
  unconditional `from .backends import NullBackend` isn't newly put at
  risk by this. Left as-is to avoid unrequested restructuring; flagging in
  case the reviewer prefers it moved.
- Verified tests exercise real behaviour, not vacuous assertions: each
  colour/index assertion reads back through `pygame.Surface.get_at`
  against the real `PICO8` RGB values (not mocked), and the parity test
  boots and runs the actual cart for 900 frames per backend.
- No skipped tests in this run: `logic.js` present, `lupa.luajit21`
  importable, `pygame` importable.

## Issues or concerns

None blocking. The one style note above (import placement) is the only
open item, and it's cosmetic.
