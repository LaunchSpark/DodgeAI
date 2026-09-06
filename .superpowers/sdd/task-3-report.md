# Task 3 Report — Token transforms: glyphs, `!=`, binary literals, `?`

## What I implemented

Added the first real translation pass to `pico8/preprocess.py`, on top of Task 2's `mask_literals`/`apply_edits` foundation:

- `BUTTONS: dict[str, int]` — maps the six PICO-8 button glyphs (`⬅️➡️⬆️⬇️🅾️❎`) to their numeric constants (0–5). Two of the glyphs (`⬅️`, `🅾️`) are multi-codepoint sequences (base codepoint + U+FE0F variation selector); the dict keys carry the full sequence and `_GLYPH_RE` is built from `re.escape` over the dict keys, so no glyph is split character-by-character.
- `_NEQ_RE` — matches `!=` for rewriting to Lua's `~=`.
- `_BIN_RE` — matches PICO-8 binary literals `0b<bits>[.<bits>]`, converting the fixed-point fractional part (`bits / 2^len(bits)`) and collapsing to a plain integer when there's no fraction.
- `_SHORTHAND_RE` — matches PICO-8's `?` print shorthand at the start of a line (after optional leading whitespace) and rewrites it to `print(...)`, wrapping through to end-of-line.
- `_token_edits(src, mask) -> list[tuple[int, int, str]]` — runs all four regexes against the **mask** (never the original source) and returns edits with positions from the mask, which are valid against `src` because `mask_literals` preserves length and only replaces literal contents.
- `preprocess(src)` — the pipeline entry point: masks, generates token edits, applies them to the original source via `apply_edits`.

This is implemented exactly per the brief, verbatim, since the code was given in full in Step 3.

## What I tested and the results

Appended the brief's test code verbatim to `tests/test_preprocess.py`:
- 6 parametrized cases for bare glyphs → button numbers (covering all six buttons, including the two multi-codepoint glyphs).
- 1 case confirming glyphs *inside strings* (the `print2("press ❎ to play again")` HUD text) are left untouched — this is the property the whole mask-based design exists to guarantee.
- 2 cases for `!=` → `~=`, one bare and one inside a string (left alone).
- 2 cases for binary literals: one fixed-point (`0b1010010110100101.1` → `42405.5`), one plain integer (`0b1010` → `10`).
- 1 case for the `?` print shorthand.

All pass. Full test suite (`pytest`, no path filter) also passes: 20 passed (5 pre-existing `mask_literals`/`apply_edits` tests + 12 new preprocess tests + 3 pre-existing cart tests).

## TDD Evidence

**RED** — ran before implementing `preprocess`/`_token_edits`/`BUTTONS`:

```
python -m pytest tests/test_preprocess.py -v
```

```
ERROR collecting tests/test_preprocess.py
...
E   ImportError: cannot import name 'preprocess' from 'pico8.preprocess' (D:\projects\DodgeAI\pico8\preprocess.py)
=========================== short test summary info ===========================
ERROR tests/test_preprocess.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

This is exactly the failure the brief predicted (`ImportError: cannot import name 'preprocess'`) — expected, since `preprocess` did not exist yet.

**GREEN** — after implementing `BUTTONS`, `_token_edits`, `preprocess`:

```
python -m pytest tests/test_preprocess.py -v
```

```
collected 17 items

tests/test_preprocess.py::test_mask_preserves_length_and_newlines PASSED
tests/test_preprocess.py::test_mask_hides_string_contents PASSED
tests/test_preprocess.py::test_mask_hides_line_comments PASSED
tests/test_preprocess.py::test_mask_hides_block_comments_but_keeps_newlines PASSED
tests/test_preprocess.py::test_apply_edits_runs_right_to_left PASSED
tests/test_preprocess.py::test_bare_glyphs_become_button_numbers[...] PASSED (x6)
tests/test_preprocess.py::test_glyphs_inside_strings_are_left_alone PASSED
tests/test_preprocess.py::test_not_equal_becomes_lua_inequality PASSED
tests/test_preprocess.py::test_not_equal_inside_string_is_left_alone PASSED
tests/test_preprocess.py::test_fixed_point_binary_literal PASSED
tests/test_preprocess.py::test_plain_binary_literal PASSED
tests/test_preprocess.py::test_question_mark_shorthand PASSED

============================= 17 passed in 0.04s ==============================
```

Full suite, run once before committing:

```
python -m pytest
```

```
collected 20 items
tests\test_cart.py ...
tests\test_preprocess.py .................
============================= 20 passed in 0.17s ==============================
```

Note: the brief's Step 4 says "Expected: 15 passed" — actual is 17. The discrepancy is in the brief's count, not a bug: the parametrized glyph test alone expands to 6 cases, and there are 6 more standalone new tests (7 test functions total, 12 test cases), plus the 5 pre-existing tests from Task 2 = 17. I flagged this per the "spec, not proven-correct code" instruction rather than silently adjusting anything to force 15.

## Files changed

- `d:\projects\DodgeAI\pico8\preprocess.py` — added `import re`, `BUTTONS`, the four compiled regexes, `_token_edits`, `preprocess`.
- `d:\projects\DodgeAI\tests\test_preprocess.py` — added `import pytest` and `preprocess` import, plus the 7 new test functions (12 test cases) from the brief.

## Self-review findings

- Diff matches the brief's Step 1/Step 3 code verbatim (confirmed via `git diff`) — no deviation.
- `_token_edits` correctly operates only on `mask`, never `src`, for pattern matching (the `src` parameter is unused inside the function but is part of the specified interface signature, kept as-is since the module is meant to stay `(src, mask)`-shaped for future tasks 4/5).
- Verified by hand that the multi-codepoint glyphs (`⬅️` = U+2B05 U+FE0F, `🅾️` = U+1F17E U+FE0F) round-trip correctly — confirmed via the passing parametrized test using `\U0001f17e️` literal.
- Verified the `?` shorthand edit math by hand: for `" ?_txt,_x,_y+1,1"`, the match `" ?"` spans [0,2); edit `(1,2,"print(")` replaces just the `?`, and a second edit `(len, len, ")")` appends the closing paren at end-of-line/end-of-string. Confirmed against the test's expected output.
- Confirmed no accidental double-blank-line issue or formatting drift; `git diff` output is a clean additive diff with standard PEP8 spacing between top-level definitions.
- Test output is pristine — no warnings, no unicode console errors (test collection is fine since pytest doesn't print glyphs to the failing-safe console by default; I did not print glyphs directly to console at any point, per the encoding guidance).
- File did not grow beyond the brief's intent — still one module, one responsibility (dialect translation), no restructuring performed.

## Issues or concerns

None. The brief's "15 passed" figure in Step 4 undercounts the parametrized test expansion; actual is 17 (all passing), which I've noted above rather than silently massaging test counts to match. No other discrepancies between the brief's test code and its implementation code were found — the mask-based skip-strings-and-comments design worked as intended on first run, both RED and GREEN transitions occurred cleanly with no debugging needed beyond the expected import-error step.

## Fix round 1

### What changed and why

Two verified correctness bugs in `pico8/preprocess.py`, both from the plan's example code:

1. **`?` shorthand closing paren landing inside a trailing comment.** `_token_edits` located the insertion point for the closing `)` via `mask.find("\n", m.end())` — the next newline. Since `mask_literals` preserves comment positions (only blanking contents to `CMT_MASK`), a trailing `-- comment` sits between the statement's real end and that newline, so the `)` was inserted inside the comment, producing an unclosed `print(` call in the emitted Lua.

   Fix: added a module-level `_statement_end(mask, pos)` helper that finds the line end and then trims back over trailing `CMT_MASK` and space/tab characters (never `STR_MASK`, so a statement legitimately ending in a string literal is untouched). `_token_edits` now calls `_statement_end(mask, m.end())` instead of the raw newline search.

2. **`_BIN_RE` matching `0b` inside identifiers.** The pattern `r"0b([01]+)(?:\.([01]+))?"` had no left boundary, so `x0b101` matched `0b101` mid-identifier and got rewritten to `x5`.

   Fix: added a negative lookbehind `(?<![\w.])` so `0b` only starts a match when not preceded by a word character or a dot: `r"(?<![\w.])0b([01]+)(?:\.([01]+))?"`.

### TDD evidence

Added 5 new tests to `tests/test_preprocess.py` (2 covering bug 1, 2 covering bug 2, 1 additional regression guard) before touching `preprocess.py`. Ran `python -m pytest tests/test_preprocess.py -v` and confirmed 3 failed for exactly the stated reasons:

```
tests/test_preprocess.py::test_question_mark_shorthand_with_trailing_comment FAILED
AssertionError: assert ' print(_txt,...draw label)\n' == ' print(_txt,... draw label\n'
-  print(_txt,_x,_y+1,1) -- draw label
+  print(_txt,_x,_y+1,1 -- draw label)

tests/test_preprocess.py::test_question_mark_shorthand_with_trailing_comment_no_newline FAILED
AssertionError: assert ' print(_txt,...- draw label)' == ' print(_txt,...-- draw label'
-  print(_txt,_x,_y+1,1) -- draw label
+  print(_txt,_x,_y+1,1 -- draw label)

tests/test_preprocess.py::test_binary_literal_not_matched_inside_identifier FAILED
AssertionError: assert 'local x5 = 5' == 'local x0b101 = 5'
- local x0b101 = 5
+ local x5 = 5
```

The pre-existing `test_fixed_point_binary_literal` and `test_question_mark_shorthand` (no trailing comment) passed both before and after, confirming no regression.

After applying both fixes, `python -m pytest tests/test_preprocess.py -v` — all 21 tests pass, including all 5 new ones. Full suite `python -m pytest` — 24 passed, 0 failed.

### Exact test command and result

```
$ python -m pytest
============================= test session starts =============================
platform win32 -- Python 3.12.2, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\projects\DodgeAI
plugins: anyio-4.12.1
collected 24 items

tests\test_cart.py ...                                                   [ 12%]
tests\test_preprocess.py .....................                           [100%]

============================= 24 passed in 0.13s ==============================
```

### Disagreements

None. Both bugs were confirmed exactly as diagnosed; no scope creep — glyph mapping, `!=` rule, and Task 2's `mask_literals`/`apply_edits` were untouched.

## Fix round 2

### What I changed

One-character fix in `pico8/preprocess.py`: dropped `.` from the negative
lookbehind on `_BIN_RE`.

```
-_BIN_RE = re.compile(r"(?<![\w.])0b([01]+)(?:\.([01]+))?")
+_BIN_RE = re.compile(r"(?<!\w)0b([01]+)(?:\.([01]+))?")
```

The prior fix (round 1) added `(?<![\w.])` to stop `0b` matching inside
identifiers like `x0b101`. Including `.` in that lookbehind also blocked a
binary literal directly after Lua's `..` concat operator, since `..0b1010`
has a `.` immediately before `0b`. Dropping `.` (keeping `\w`) still blocks
the identifier case while allowing the concat-operator case to convert.

### TDD evidence

Added four tests to `tests/test_preprocess.py`:
- `test_binary_literal_after_concat_operator` - `a=a..0b1010` -> `a=a..10`
- `test_binary_literal_after_unary_minus` - `x=-0b1010` -> `x=-10`
- `test_binary_literal_still_not_matched_inside_identifier` - guards
  `local x0b101 = 5` and `local job0b1 = 5` stay unchanged

**RED** (before the fix), `python -m pytest tests/test_preprocess.py -v`:

```
tests/test_preprocess.py::test_binary_literal_after_concat_operator FAILED [ 91%]
tests/test_preprocess.py::test_binary_literal_after_unary_minus PASSED   [ 95%]
tests/test_preprocess.py::test_binary_literal_still_not_matched_inside_identifier PASSED [100%]

================================== FAILURES ===================================
__________________ test_binary_literal_after_concat_operator __________________

    def test_binary_literal_after_concat_operator():
        # regression: the `.` in the lookbehind wrongly blocked `..0b...`
>       assert preprocess("a=a..0b1010") == "a=a..10"
E       AssertionError: assert 'a=a..0b1010' == 'a=a..10'
E
E         - a=a..10
E         + a=a..0b1010
E         ?      ++  ++

tests\test_preprocess.py:105: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_preprocess.py::test_binary_literal_after_concat_operator - ...
======================== 1 failed, 23 passed in 0.22s =========================
```

This matches the predicted regression exactly: the concat-operator case
failed for the stated reason (dot before `0b` suppressed the match), while
the unary-minus case and both identifier guards already passed (they were
never affected by the `.` in the lookbehind).

**GREEN** (after the fix), `python -m pytest tests/test_preprocess.py -v`:

```
============================= 24 passed in 0.08s ==============================
```

All 24 tests in the file passed, including the 4 new ones.

### Full suite

```
python -m pytest
```

```
tests\test_cart.py ...                                                   [ 11%]
tests\test_preprocess.py ........................                        [100%]

============================= 27 passed in 0.13s ==============================
```

### Disagreements

None. Confirmed the diagnosis by reasoning through it independently before
applying the fix: a Lua field access cannot start with a digit (`t.0b101`
is not valid Lua), and a numeric literal cannot contain `b` after a decimal
point (`1.0b1` is not valid either), so there is no legitimate Lua construct
where a dot immediately preceding `0b` should suppress the binary-literal
match. The only realistic source of `.0b` is `..0b` (concat operator
immediately followed by a literal), which must convert. Applied the fix
exactly as specified: drop `.` from the lookbehind, keep `\w`.
