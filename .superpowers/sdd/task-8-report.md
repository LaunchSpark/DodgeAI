# Task 8 Report: Backend protocol, NullBackend, and the machine

## What I implemented

- `pico8/backends.py` — `Backend` interface (13 draw/flip methods, all `raise NotImplementedError`) and `NullBackend(Backend)`, which implements every method as an explicit no-op. Per the dispatch's mandatory decision, these are individually written stub methods, not `__getattr__` magic, because the machine must call every draw entry point every frame (the cart's `spawntrail()` inside `drawgame()` consumes `rnd()`, so skipping a call would change which enemies spawn).
- `pico8/api.py` — installs the Python-backed PICO-8 builtins as Lua globals: `btn`/`btnp` (with PICO-8's 15-frame-then-4-frame repeat timing), `stat` (32/33/34 → mouse x/y/buttons), `peek`/`poke` against a shared byte buffer, `time`/`t`, no-ops for `sfx`/`music`/`printh`/`palt`, and the drawing calls (`cls`, `pset`, `rect`, `rectfill`, `circfill`, `ovalfill`, `line`, `spr`, `print`, `pal`, `fillp`, `camera`, `flip`), each dispatched to the machine's backend.
- `pico8/machine.py` — `Machine(rom, code, backend, seed=0)`: owns `mem` (a 64KB `bytearray`, backing both `peek`/`poke` and the cart's memory-mapped persistence via `store`/`get_data` at `0x5f82+i`), button/mouse state, and the LuaJIT runtime. Boot order: install the Python API → execute `prelude.lua` → `srand(seed)` → execute the preprocessed cart (this is where the 66-frame splash animation runs and calls `flip()` mid-construction, and where the cart pokes `0x5ffe`/`0x5f2d` at load — both work because the API and `mem` buffer exist before the cart's code runs). `.step()` advances one frame: updates `btnp` state, calls `_update60()` (present but empty in this cart) then `_draw()` (where all real game logic lives), then flips the backend. `.start_game()` skips the menu/transition state machine by directly wiring `_upd`/`_drw` to `updategame`/`drawgame` and calling `reset()`. `boot(js_path=Path("logic.js"), backend=None, seed=0)` reads the ROM, decompresses the code, and constructs a `Machine` (defaulting to a fresh `NullBackend()`).
- `tests/test_machine.py` — the six smoke tests from the brief (one rewritten; see below), plus a module-scoped `autouse` fixture that silences a Windows/LuaJIT SEH artifact (see "Cart behaviour that surprised me").

**Mandatory corrections applied, verbatim as instructed:**
- `machine.py` imports `from lupa.luajit21 import LuaRuntime` (not `from lupa import LuaRuntime`, which defaults to Lua 5.5 in lupa 2.8 and would change integer/float stringification and `unpack`'s location).
- `tests/test_machine.py` uses `luajit = pytest.importorskip("lupa.luajit21")`, matching `test_prelude.py`/`test_preprocess_cart.py`.
- No `DIFFICULTY` dict — confirmed absent from `machine.py`; nothing in phase 1 reads it.

## What I tested, and the results

Ran the full suite, not just the new file, with `python -m pytest -v`:

```
73 passed in 1.24s
```

- `tests/test_machine.py`: 6 passed (all six from the brief; none skipped — `logic.js` and `lupa.luajit21` are both present, so the `skipif`/`importorskip` guards never trigger).
- Pre-existing suites (`test_cart.py`, `test_prelude.py`, `test_preprocess.py`, `test_preprocess_cart.py`): all still pass, 67 tests, unchanged.
- No skips, no warnings, no stray output — confirmed by running with `-v` and inspecting the full log (see below re: the SEH dump I found and silenced).

## TDD evidence

**RED** — wrote `tests/test_machine.py` (brief's test verbatim, with the `lupa.luajit21` import correction) before any of the three source files existed:

```
$ python -m pytest tests/test_machine.py -v
...
ImportError while importing test module 'D:\projects\DodgeAI\tests\test_machine.py'.
tests\test_machine.py:5: in <module>
    from pico8 import machine
E   ImportError: cannot import name 'machine' from 'pico8'
```
Correct failure: missing module, not a typo.

**GREEN (first pass)** — implemented `backends.py`, `api.py`, `machine.py` per the brief (with the mandated corrections) and re-ran:

```
$ python -m pytest tests/test_machine.py -v
...
FAILED tests/test_machine.py::test_same_seed_gives_the_same_trajectory
5 passed, 1 failed in ~1.7s
```

`test_same_seed_gives_the_same_trajectory` failed on `assert run(11) != run(12)` — both seeds produced byte-identical 300-frame traces of `(px, py, len(enemies))`.

**Diagnosis** (see "bug found" below): the implementation is correct; the test's chosen observables can never diverge for this cart under these conditions. I rewrote the test's trace to fingerprint each spawned enemy's `.x`, `.y`, `.p` (personality), and `.maxs` (size) instead of just `len(enemies)`.

**GREEN (final)**:

```
$ python -m pytest tests/test_machine.py -v
tests/test_machine.py::test_cart_boots_and_defines_its_entry_points PASSED
tests/test_machine.py::test_stepping_runs_frames_without_error PASSED
tests/test_machine.py::test_gameplay_state_advances_and_kills_the_idle_player PASSED
tests/test_machine.py::test_enemies_actually_spawn PASSED
tests/test_machine.py::test_same_seed_gives_the_same_trajectory PASSED
tests/test_machine.py::test_buttons_move_the_player PASSED
6 passed in 1.06s
```

Full suite:
```
$ python -m pytest -q
73 passed in 1.24s
```

## Bug found in the brief's own test code

`test_same_seed_gives_the_same_trajectory` traced `(round(px,6), round(py,6), len(enemies))` per frame and asserted two different seeds produce different traces. That assertion can never pass for this cart, for two independent reasons I verified directly against `dodge.lua`:

1. **`px`/`py` never depend on `rnd()`.** In mouse mode (`input==0`, the default — confirmed by `anybtn()`/`anymouse()` control-mode logic, see below), `px,py = stat(32),stat(33)`, i.e. the fixed default mouse position `(64,64)` when nothing calls `set_mouse`. In button mode, movement is `dpx += dirx[i]*pspd` — also deterministic given fixed button state. Neither path touches the PRNG, and collisions never apply RNG-driven knockback to the player's position either (`collide()` only sets flags/score, or calls `die()`).
2. **Enemy spawn *timing* is not random.** `spawnenemies()` gates on `ed > est*60*spp*(2-frz)`, a frame counter against a fixed threshold — no `rnd()` involved. So `len(enemies)` at any given frame is identical for every seed when the player is stationary (both seeds' 300-frame traces were confirmed byte-identical, including a `fyou()` anti-camping burst of 5 simultaneous spawns at frame 50 in both cases).

What the seed *does* drive is *which* enemy spawns: `es = flr(rnd(3))+3` (size) and `ep = pers[flr(rnd(#pers))+1]` (personality), both drawn from `rnd()` inside `spawnenemies()`. I fixed the test to track each enemy's `.maxs` and `.p` fields. Empirically verified (via a throwaway script) that seeds 11 and 12 diverge for the first time at frame 181, in exactly one enemy's `.maxs` (4 vs. 3) — a genuine RNG-driven difference, not a spurious one.

This is a real bug in the brief, not a misunderstanding on my part: I checked the actual game source (`dodge.lua` lines 311–343 for movement, 563–594 for collision, 640–682 for spawning) before concluding the test's premise was wrong rather than the implementation.

## Cart behaviour that surprised me

**A Windows/LuaJIT SEH artifact, not a real crash.** Running the real 2,263-line cart (as opposed to the small prelude snippets in earlier tasks) tripped `faulthandler`'s "Windows fatal exception: code 0xe24c4a02" dump *hundreds of times* (489, across the full suite) — pointing at `machine.py`'s `step()` and `Machine.__init__`. This is the same phenomenon already documented and worked around in `test_preprocess_cart.py`'s `faulthandler_disabled()` helper ("LuaJIT uses structured exception handling internally; on Windows this trips faulthandler's fatal-error dump even though nothing actually crashed") — just far more frequent here because real gameplay code triggers LuaJIT's trace compiler repeatedly, apparently once per JIT trace. All 73 tests still passed; this was purely a diagnostic-output problem, not a correctness one. I added a module-scoped `autouse` fixture in `test_machine.py` (`_quiet_luajit_seh`) that disables `faulthandler` for the duration of the module's tests, restoring it afterward, so the run stays pristine without masking crash dumps in unrelated test files. Confirmed 0 occurrences after the fix, 73 still passing.

**Everything else matched the dispatch's warnings exactly:** the 66-frame splash `flip()` calls during `lua.execute(code)` worked fine because the API and `mem` buffer are installed first; `smallsettings` is correctly populated at load via the unconditional `updatesettings()` call in top-level code (I traced through and confirmed the resulting `{{12,1},2,true,true}` matches the literal default, since `get_data` reads all-zero memory → `defcur(0)=1` for every setting); and the idle-player death does come from the anti-camping `fyou()` mechanic (confirmed: at frame 50, `enemies` jumps from 0 to 5 simultaneously — one spawn per corner plus one regular spawn — for a player sitting at the screen center).

## Files changed

- `d:\projects\DodgeAI\pico8\backends.py` (new)
- `d:\projects\DodgeAI\pico8\api.py` (new)
- `d:\projects\DodgeAI\pico8\machine.py` (new)
- `d:\projects\DodgeAI\tests\test_machine.py` (new)

Commit: `d409f2f` "feat: PICO-8 machine running the cart headless" on branch `design/dodge-env`. Only these four files were staged/committed (matches the brief's Step 7 exactly); `logic.js` remains untracked as it was before this task, untouched.

## Self-review findings

- Confirmed `backends.py`/`api.py`/`machine.py` match the brief's code with only the mandated corrections (LuaJIT import, no `DIFFICULTY` dict) — no scope creep, no unused constants added.
- Confirmed `NullBackend` keeps 13 explicit no-op methods (no `__getattr__`), per the correctness requirement about randomness consumption.
- Verified by grepping `dodge.lua` that every call site into `rect`/`rectfill`/`circfill`/`ovalfill`/`line`/`pset`/`pal`/`camera`/`spr` in the cart supplies all the arguments `api.py`'s lambdas default — so the `_num()` defaulting logic in `api.py`, while present per the brief, is defensive/unexercised dead-ish code for this cart. Left as-is since it matches the brief's specified interface and is harmless (the interface must accept nil per the `Backend` protocol regardless of what this one cart happens to call).
- Found and fixed one genuine bug in the brief's test code (`test_same_seed_gives_the_same_trajectory`, detailed above) — the implementation was correct, the test's assertion was unsatisfiable as written.
- Found and fixed one environmental/tooling issue (SEH dump noise) not mentioned in the brief, using the same pattern already established elsewhere in this repo's test suite.
- Ran the full suite (73 tests) clean, no skips, no warnings, no stray output, before committing.

## Issues or concerns

None outstanding. Status: DONE.

## Fix round 1

Two review findings fixed before the rendering work starts. Both are one-line changes; nothing else touched.

### Fix 1: `btnp` key repeat fires three frames late

**Derivation.** `api.py` documents "first repeat after 15 frames, then every 4," i.e. firing frames `[1, 16, 20, 24, 28, ...]` while a button is held continuously (`held` = number of consecutive frames the button has been down, `held == 1` on the press frame).

Old condition: `held == 1 or (held > 15 and (held - 15) % 4 == 0)`. The modulo term is first satisfied at `held == 19` (`19 - 15 = 4`, `4 % 4 == 0`), giving `[1, 19, 23, 27, ...]` — three frames late, and off the documented cadence.

New condition: `held == 1 or (held > 15 and (held - 1 - 15) % 4 == 0)`, i.e. `(held - 16) % 4 == 0`. Checking small values: `held=16 → 0 % 4 == 0` ✓ (first repeat, exactly at the documented 15-frame delay after the press). `held=17,18,19 → 1,2,3 % 4 ≠ 0` (no fire). `held=20 → 4 % 4 == 0` ✓. Continuing every 4th frame: `24, 28, 32, ...`. So held values `1, 16, 20, 24, 28, ...` fire — matches `[1, 16, 20, 24, 28, ...]` exactly.

**Fix applied** (`pico8/machine.py`, `_update_btnp`):
```python
if held == 1 or (held > api.BTNP_DELAY and
                 (held - 1 - api.BTNP_DELAY) % api.BTNP_INTERVAL == 0):
```

**TDD evidence.** Added `test_btnp_repeats_after_15_frames_then_every_4` to `tests/test_machine.py`: holds button 0 for 40 frames via a freshly booted machine, collects every frame on which `btnp_mask` reports the button, and asserts the set equals `[1, 16, 20, 24, 28, 32, 36, 40]`; also asserts `bool(m.btnp_mask & 1) == bool(m.globals.btnp(0))` every frame (two reads within the same frame agree), and that releasing for 5 frames then re-pressing fires immediately on the next step.

Ran before the fix: failed exactly as predicted —
```
AssertionError: assert [1, 19, 23, 27, 31, 35, ...] == [1, 16, 20, 24, 28, 32, ...]
```
(list actually diverges at index 1: `19 != 16`, and old code produces one fewer fire in the 40-frame window). Applied the one-line fix; test then passed.

### Fix 2: drawing calls early-bind the backend while `flip` late-binds it

`pico8/api.py`'s `install()` captured `b = machine.backend` once, and all twelve drawing lambdas (`cls`, `pset`, `rect`, `rectfill`, `circfill`, `ovalfill`, `line`, `spr`, `print`, `pal`, `fillp`, `camera`) closed over that local `b`. `Machine.flip()`/`.step()` already resolved `self.backend.flip()` fresh each call, so `flip` was already late-bound — only the draw calls had the bug. Swapping `machine.backend` after construction would silently keep drawing into the discarded backend forever.

**Fix applied**: removed the `b = machine.backend` capture; every drawing lambda now reads `machine.backend.<method>(...)` at call time instead of the captured `b`, e.g. `g.cls = lambda col=None: machine.backend.cls(int(_num(col)))`. Defaults, coercions (`_num`, `int()`), and argument shapes are unchanged — only the backend lookup moved from install-time to call-time.

**TDD evidence.** Added `RecordingBackend(Backend)` to `tests/test_machine.py` (records method names called, does nothing else) and `test_swapping_backend_after_construction_redirects_draw_calls`: boots a machine on `old_backend`, records how many calls landed on it during boot (booting this cart runs a splash/attract animation that legitimately draws before any test code runs — that baseline is not what's under test), swaps `m.backend = new_backend`, calls `m.step()`, then asserts no *additional* calls landed on `old_backend` after the swap and that `new_backend` did receive calls.

Ran before the fix (verified by temporarily stashing only the `api.py` change and re-running just this test, to make sure the *btnp* fix wasn't masking anything): failed exactly as predicted —
```
AssertionError: draw calls made after the backend swap should not land on the stale backend, but 14 more landed there
assert 806 == 792
```
Restored the `api.py` fix; test then passed, and 14 draw calls per step now correctly land on the swapped-in backend instead of the stale one.

### Full suite

Command: `python -m pytest -v` (also `python -m pytest -q` for the terse form)

Result: `75 passed in 1.43s` — no skips, no warnings, no stray console output (checked with `-v`, full log inspected). 73 pre-existing tests unchanged and passing; 2 new tests added and passing.

### Commit

`git diff` on the two source files touched confirms the changes are exactly as prescribed: `pico8/machine.py` — one line changed (the modulo subtrahend); `pico8/api.py` — the `b = machine.backend` line removed, twelve lambda bodies changed from `b.<method>(...)` to `machine.backend.<method>(...)`, nothing else.

Files changed:
- `d:\projects\DodgeAI\pico8\machine.py` (btnp timing fix)
- `d:\projects\DodgeAI\pico8\api.py` (late-bind backend in drawing lambdas)
- `d:\projects\DodgeAI\tests\test_machine.py` (two new regression tests + `RecordingBackend` helper)

No other files modified. `logic.js` remains untracked, untouched.
