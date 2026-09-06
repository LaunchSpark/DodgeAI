# Task 6 Report: Whole-cart syntax check

## What I implemented

1. **`tests/test_preprocess_cart.py`** — integration test matching the brief's
   Step 1 code, with the mandatory correction applied: uses
   `lupa.luajit21.LuaRuntime()` instead of `lupa.LuaRuntime()` everywhere,
   and `pytest.importorskip("lupa.luajit21")` instead of
   `pytest.importorskip("lupa")`, so the skip-if-lupa-absent behavior still
   works but the runtime that actually gets constructed is LuaJIT 2.1
   rather than lupa's default (Lua 5.5, which has no `loadstring`).
   Three tests:
   - `test_whole_cart_is_valid_lua` — runs `preprocess()` over the full
     `dodge.lua` and compiles (not runs) the result via `loadstring` under
     LuaJIT.
   - `test_no_pico8_dialect_remains` — masks literals/comments in the
     translated output and asserts no `!=`, `0b`, bare button glyphs, or
     compound-assignment operators survive.
   - `test_hud_glyphs_survive_inside_strings` — confirms the `"❎"` HUD
     prompt and "to play again" text are untouched (they're inside string
     literals, not dialect syntax).

2. **`tools/check_lua.py`** — developer diagnostic, same shape as
   `tools/extract_cart.py` (module docstring with description, examples/usage,
   `def main()`, `if __name__ == "__main__":` guard, `sys.path.insert` so it
   runs from the `tools/` directory without install). Preprocesses
   `dodge.lua`, attempts `loadstring` under `lupa.luajit21.LuaRuntime()`, and
   on failure prints the compiler's error plus five lines of context around
   the reported line (`repr()`'d, per the dispatch note, since dodge.lua
   contains cp1252-unencodable glyphs). Exits 0 on success, 1 on failure.

No changes were needed to `pico8/preprocess.py` — see below.

## What I tested and the results

- `python -m pytest tests/test_preprocess_cart.py -v` → **3 passed**, no
  skips. Confirmed no skip by also running with `-rs` (skip-reason report)
  across the whole suite — nothing listed, and the per-file summary line
  reads `tests\test_preprocess_cart.py ...` (three dots = three tests
  actually executed, not `sss`).
- `python -m pytest tests/ -v` → **52 passed** (49 pre-existing +
  3 new), 0 failed, 0 skipped, in under 1s. The brief's stale "Expected: N
  passed" counts (from Step 2/4) don't match; 52 is the real, current total.
- `python tools/check_lua.py` → `ok: translated cart compiles`, exit code 0,
  confirming at the tool level (independent of lupa/pytest fixture plumbing)
  that the real cart's translation compiles clean under LuaJIT.

Per the dispatch note's prediction, the whole-cart test needed **zero fixes**
to `pico8/preprocess.py` — it passed on the first run. Step 2 of the brief
(run test, expect it to fail) was skipped as moot; the dispatch instructions
explicitly said the brief's Step 3 fix cycle should be a no-op here, and it
was.

## Verification that check_lua.py's error-line parsing is correct

Per instructions, I fed it a deliberately broken input rather than trusting
the brief's example code:

1. Backed up `dodge.lua` to `dodge.lua.bak`.
2. Replaced line 50 (1-indexed; a `for i=0,65 do` loop body line) with the
   garbage `local x = ((( this is not valid lua`.
3. Ran `python tools/check_lua.py`. Output:

   ```
   [string "..."]:50: ')' expected near 'is'
      48  'size=0'
      49  'for i=0,65 do'
      50> 'local x = ((( this is not valid lua'
      51  '\tif size<1 then'
      52  '\t\tsize = size + (0.045)'
   exit: 1
   ```

   The reported line (50) exactly matches the corrupted line, the `>` marker
   lands on it, and two lines of context print on each side as designed.
4. Restored `dodge.lua` from the backup, deleted the backup, and re-ran
   `python tools/check_lua.py` to confirm it reports `ok: translated cart
   compiles` again (exit 0). Reran the full test suite afterward (52 passed)
   to confirm the working tree was left byte-for-byte clean —
   `git status --porcelain` shows no diff on `dodge.lua`, only the
   pre-existing untracked `logic.js` (unrelated to this task, present before
   I started).

The parsing logic (`int(str(error).split(":")[1])`) works because LuaJIT's
error format is `[string "..."]:<line>: <message>` — splitting on `:` and
taking index 1 reliably extracts the line number for this error shape.

## Files changed

- `d:\projects\DodgeAI\tests\test_preprocess_cart.py` (new)
- `d:\projects\DodgeAI\tools\check_lua.py` (new)
- `pico8/preprocess.py` — **unchanged** (verified via `git diff`, no output)

## Self-review findings

- **Runtime correction applied consistently**: both files construct
  `LuaRuntime` from `lupa.luajit21`, not the bare `lupa` module. Verified
  `loadstring` exists on that runtime (`Lua 5.1` reported by `_VERSION`)
  before writing any test logic.
- **`importorskip` target**: used `"lupa.luajit21"` rather than `"lupa"` so
  the skip path is honest — if the luajit21 extension specifically is
  missing (even if base `lupa` is present), the test skips cleanly instead
  of raising `AttributeError` on `lupa.LuaRuntime`.
- **No `import lupa` left dangling in check_lua.py** — only
  `from lupa.luajit21 import LuaRuntime` is imported, matching what's
  actually used (leftover generic `import lupa` from the brief would be
  dead code).
- **Encoding safety**: both files use `encoding="utf-8"` explicitly on the
  `dodge.lua` read; `check_lua.py`'s context-line printing uses `!r` so a
  console using cp1252 (Windows default) won't raise `UnicodeEncodeError`
  on the HUD glyphs embedded in the source — confirmed by actually running
  it in this PowerShell/cp1252 environment without incident.
  the `print(error)` line itself is safe since Lua error messages here are
  ASCII.
  the `!r` gives readable line content while avoiding this repo's known
  cp1252 hazard.
- **Test file doesn't run vacuously**: confirmed via `-rs` (no skip reasons
  reported) and via the dot-count in the per-file summary that all three
  cart tests actually executed against the real, extracted `dodge.lua`, not
  skipped due to a missing file or missing dependency.
- **check_lua.py structural consistency**: matches `extract_cart.py`'s
  conventions (docstring, `main()`, `sys.path.insert` bootstrap, `__main__`
  guard) without importing `argparse` for zero-argument usage — YAGNI
  respected, no unused CLI surface.
- **Verified the diagnostic tool independently catches real breakage**: the
  inject-then-revert experiment (see above) proves `check_lua.py` isn't
  just printing "ok" unconditionally; it genuinely parses LuaJIT's
  compile-time error and locates the right line.
- Test output is pristine: 52 passed, 0 warnings, 0 skipped, 0 errors on
  final full-suite run.

## Issues or concerns

None. The task was, as predicted in the dispatch, a no-op fix cycle: the
existing `pico8/preprocess.py` already produces a translation of the full
cart that compiles cleanly under LuaJIT 2.1. The only real engineering work
was writing the integration test and the diagnostic tool correctly (with the
LuaJIT-runtime correction applied) and independently verifying the
diagnostic's line-number logic against a deliberately broken input.

## Fix round 1

### What changed and why

`tests/test_preprocess_cart.py::test_whole_cart_is_valid_lua` had a verified
vacuous-assertion bug. The Lua wrapper was:

```lua
function(src) return loadstring(src) end
```

`return loadstring(src)` is a tail call, so on a compile failure it forwards
*both* of `loadstring`'s return values (`nil, "error message"`) out of the
Lua function. Lupa surfaces multiple Lua return values as a Python tuple, and
`chunk is not None` is `True` for a tuple regardless of whether the first
element is `nil` — so the assertion passed unconditionally, whether or not
the cart actually compiled. This made the phase's flagship compile-check
test give zero protection against a compile regression.

Two changes were made to `tests/test_preprocess_cart.py`:

1. **`test_whole_cart_is_valid_lua`** — rewrote the wrapper to capture both
   `loadstring` return values into Lua locals and return only the error:
   `function(s) local f, e = loadstring(s) return e end`. The test now
   asserts `error is None`, with the actual compiler error included in the
   failure message if not. This matches `tools/check_lua.py`'s existing
   (correct) wrapper shape exactly, so the two are consistent.
2. **Added `test_compile_check_rejects_invalid_lua`** — a guard test that
   proves the compile-check wrapper can actually detect invalid Lua, so this
   class of bug (a check that can only pass) cannot silently reappear in a
   future refactor. It feeds a deliberately invalid Lua string
   (`"this is not valid lua $$$"`, held only in memory, never written to
   disk) through the same corrected wrapper and asserts the returned error
   is not `None` and is a `str`.

Also applied the requested clarity fix: renamed the module-level binding of
`lupa.luajit21` from `lupa` to `luajit` (`luajit = pytest.importorskip(...)`,
`runtime = luajit.LuaRuntime()`), so the code no longer reads like the
default `lupa.LuaRuntime()` (Lua 5.5, which this project must avoid). The
runtime actually constructed is unchanged — still LuaJIT 2.1 via
`lupa.luajit21`.

No changes were made to `pico8/preprocess.py`, `dodge.lua`, or `dodge.p8`.

### TDD evidence

**RED** — before applying the Part 1 fix, `test_compile_check_rejects_invalid_lua`
was first written using the *buggy* (tail-call) wrapper shape, to prove the
bug class is real and that the guard test can catch it:

```
python -m pytest tests/test_preprocess_cart.py::test_compile_check_rejects_invalid_lua -v
...
    assert error is not None
>   assert isinstance(error, str)
E   assert False
E    +  where False = isinstance((None, '[string "this is not valid lua $$$"]:1: \'=\' expected near \'is\''), str)
FAILED tests/test_preprocess_cart.py::test_compile_check_rejects_invalid_lua
1 failed in 0.16s
```

The tuple return is visible directly in the assertion diff — exactly the
bug the task describes. This confirms `error is not None` alone (the old
`chunk is not None` shape) would have passed vacuously; only the added
`isinstance(error, str)` check exposed the defect.

**GREEN** — after rewriting both `test_whole_cart_is_valid_lua` and
`test_compile_check_rejects_invalid_lua` to use the corrected wrapper
(capturing `f, e` into locals and returning only `e`):

```
python -m pytest tests/test_preprocess_cart.py -v
tests/test_preprocess_cart.py::test_whole_cart_is_valid_lua PASSED
tests/test_preprocess_cart.py::test_compile_check_rejects_invalid_lua PASSED
tests/test_preprocess_cart.py::test_no_pico8_dialect_remains PASSED
tests/test_preprocess_cart.py::test_hud_glyphs_survive_inside_strings PASSED
4 passed in 0.10s
```

`test_whole_cart_is_valid_lua` passing here is a real pass: the translated
cart genuinely compiles under LuaJIT with a wrapper that can now actually
fail. This is not the old test masking a problem — the assertion is now
meaningful and still passes because `dodge.lua`'s translation is in fact
valid Lua.

(Note: both runs above print a `Windows fatal exception: code 0xe24c4a03`
traceback from the interpreter's `faulthandler` when `loadstring` hits
invalid syntax inside `lupa.luajit21`. This occurs identically pre- and
post-fix, exit code is 0, and pytest reports `PASSED`/`FAILED` correctly in
both cases — it is a benign artifact of how LuaJIT's C-level error handling
surfaces through Python's fault handler on this Windows/Python 3.12 box, not
a real crash, and it does not affect any test outcome.)

### Full suite

```
python -m pytest
53 passed in 0.32s
```

0 failed, 0 skipped, 0 errors. (53 = 49 pre-existing + 4 in
`test_preprocess_cart.py`, i.e. the original 3 plus the new guard test.)

### Working tree confirmation

```
git status --short
 M tests/test_preprocess_cart.py
?? logic.js
```

Only `tests/test_preprocess_cart.py` is modified. `dodge.lua` and `dodge.p8`
do not appear in the status output — unmodified. `logic.js` is a
pre-existing untracked file present before this task began and unrelated to
it; left untouched.

### Disagreements

None. The task instructions were unambiguous and the fix matched
`tools/check_lua.py`'s existing wrapper exactly, as directed.

## Fix round 2

### What changed and why

The prior "Fix round 1" note above observed that
`test_compile_check_rejects_invalid_lua` printed a
`Windows fatal exception: code 0xe24c4a03` stack dump from pytest's
`faulthandler` plugin, triggered by `loadstring` hitting invalid syntax
inside `lupa.luajit21`. It was benign (exit code 0, correct PASSED/FAILED
reporting) but noisy, and noise here is a real cost: later tasks run a lot
of LuaJIT, and a suite that always prints a fatal-looking dump will mask a
genuine crash when one occurs.

Fixed by suppressing `faulthandler` narrowly around the single deliberate
`loadstring` call that triggers it, in
`tests/test_preprocess_cart.py`:

1. Added `import faulthandler` and `from contextlib import contextmanager`.
2. Added a small named context manager, `faulthandler_disabled()`, defined
   right after the `translated` fixture. It records whether faulthandler was
   enabled, disables it, yields, and re-enables it in a `finally` block only
   if it was enabled beforehand (so the fixture doesn't force it on if
   something upstream had it off).
3. Wrapped only the invalid-Lua compile call in
   `test_compile_check_rejects_invalid_lua` with
   `with faulthandler_disabled(): error = compile_error(...)`. Nothing else
   changed in that test — it still compiles `"this is not valid lua $$$"`
   and still asserts `error is not None` and `isinstance(error, str)`, i.e.
   the guard test's actual behavior (proving the compile check can fail) is
   untouched.

No changes to `test_whole_cart_is_valid_lua` (compiles valid Lua, never
triggers the SEH path), `pico8/preprocess.py`, `dodge.lua`, or `dodge.p8`.
No global pytest config was added — `faulthandler` remains fully active for
every other test.

### Before

```
python -m pytest
...
tests\test_preprocess_cart.py .Windows fatal exception: code 0xe24c4a03

Current thread 0x000077ac (most recent call first):
  File "D:\projects\DodgeAI\tests\test_preprocess_cart.py", line 29 in test_compile_check_rejects_invalid_lua
  ... (25+ more stack frames from pytest/pluggy internals) ...
...                                       [100%]

============================= 53 passed in 0.28s ==============================
```

### After

```
python -m pytest
============================= test session starts =============================
platform win32 -- Python 3.12.2, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\projects\DodgeAI
plugins: anyio-4.12.1
collected 53 items

tests\test_cart.py ...                                                   [  5%]
tests\test_preprocess.py ..............................................  [ 92%]
tests\test_preprocess_cart.py ....                                       [100%]

============================= 53 passed in 0.38s ==============================
```

`grep -i "fatal\|traceback"` over the captured output returns nothing (exit
code 1 / no match). 53 passed, 0 skipped, 0 failed, no stack dump anywhere.

### Guard test still genuine

```
python -m pytest tests/test_preprocess_cart.py -v
tests/test_preprocess_cart.py::test_whole_cart_is_valid_lua PASSED       [ 25%]
tests/test_preprocess_cart.py::test_compile_check_rejects_invalid_lua PASSED [ 50%]
tests/test_preprocess_cart.py::test_no_pico8_dialect_remains PASSED      [ 75%]
tests/test_preprocess_cart.py::test_hud_glyphs_survive_inside_strings PASSED [100%]

============================== 4 passed in 0.10s ==============================
```

`test_compile_check_rejects_invalid_lua` reports PASSED (not skipped), and
its body is unchanged apart from the `with faulthandler_disabled():` wrapper
around the `compile_error(...)` call — it still feeds genuinely invalid Lua
through the same wrapper as the real compile check and still asserts a
non-`None` string error comes back.

### Working tree confirmation

```
git status --short
 M tests/test_preprocess_cart.py
?? logic.js
```

Only `tests/test_preprocess_cart.py` is modified; `logic.js` is the
pre-existing untracked file from before this task, unrelated. `dodge.lua`
and `dodge.p8` do not appear — unmodified.

### Disagreements

None. Suppressing `faulthandler` narrowly around the one call known to
trigger LuaJIT's SEH-based fatal-exception dump worked on the first attempt
and fully removed the dump without touching faulthandler's behavior for any
other test.
