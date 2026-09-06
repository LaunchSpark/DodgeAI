# Task 5 Report: One-line `if`

## What I implemented

Added the final preprocess pass, `_short_if_edits`, to `pico8/preprocess.py`, translating
PICO-8's one-line `if (cond) body` into standard Lua `if (cond) then body end`. The original
parentheses of the condition are kept as-is; the transform is two pure insertions — ` then`
right after the closing paren, and ` end` at the statement end (computed by the existing,
reused `_statement_end`).

Added helpers, exactly as specified in the brief:
- `_SHORT_IF_RE = re.compile(r"(?m)^[ \t]*if[ \t]*\(")` — matches a line beginning with
  (optional indentation +) `if (`.
- `_match_paren(mask, open_pos, limit)` — depth-balanced search for the paren matching
  `mask[open_pos]`, bounded to `limit` (end of physical line); returns `None` if unmatched
  before the limit, which is how multi-line conditions are recognized and skipped.
- `_short_if_edits(src, mask)` — for each match, finds the line end, locates the matching
  close-paren by balance (not first `)`), skips if the body is empty or already starts with
  `then` (already an ordinary if), and otherwise emits the two insertion edits.

`preprocess` now runs passes in order `(_token_edits, _compound_edits, _short_if_edits)`,
with `_short_if_edits` last — required because a short `if` whose body is a compound
assignment (`if (btnp(⬇️)) cursi+=1`, present at dodge.lua:285, 286, 919) would otherwise have
its appended `end` absorbed into the compound-assignment RHS if the short-if pass ran first.
Also added a docstring to `preprocess` documenting that the pass order is load-bearing.

No deviation from the brief's reference implementation was needed — I checked it against the
tests and found no discrepancy.

## What I tested and the results

Ran the focused `short_if` tests first (TDD RED, then GREEN), then the full preprocess suite,
then the whole project's test suite.

- Full `test_preprocess.py`: 42 passed (36 pre-existing + 6 new).
- Whole project (`python -m pytest`): 45 passed (`test_cart.py` ×3 + `test_preprocess.py` ×42).

Both test runs were pristine — no warnings, no errors, no glyph-related `UnicodeEncodeError`
(nothing needed to be printed to the console since all assertions passed).

## TDD Evidence

### RED

Command:
```
python -m pytest tests/test_preprocess.py -k short_if -v
```
Output (abridged):
```
tests/test_preprocess.py::test_short_if_simple FAILED
tests/test_preprocess.py::test_short_if_with_compound_assignment FAILED
tests/test_preprocess.py::test_short_if_with_nested_parens_in_condition FAILED
tests/test_preprocess.py::test_short_if_with_two_statement_body FAILED
tests/test_preprocess.py::test_short_if_end_goes_before_trailing_comment FAILED
tests/test_preprocess.py::test_short_if_preserves_indentation_of_body FAILED
6 failed, 36 deselected in 0.20s
```
Sample failure detail:
```
AssertionError: assert 'if (input==0...92,71,56,1,2)' == 'if (input==0...1,56,1,2) end'
- if (input==0) then spr(192,71,56,1,2) end
?              -----                   ----
+ if (input==0) spr(192,71,56,1,2)
```
This is the expected failure: with no `_short_if_edits` pass yet, `preprocess` left the
source untouched — no ` then` and no ` end` were inserted. (Note: `test_explicit_then_is_not_transformed`,
`test_multiline_condition_is_not_transformed`, and `test_bare_if_without_parens_is_not_transformed`
assert no-op behavior, so they trivially passed even pre-implementation — they were written
alongside the others but aren't "RED" in the failing sense; the transformation tests above are
the real RED evidence.)

### GREEN

Command:
```
python -m pytest tests/test_preprocess.py -k short_if -v
```
Output:
```
tests/test_preprocess.py::test_short_if_simple PASSED
tests/test_preprocess.py::test_short_if_with_compound_assignment PASSED
tests/test_preprocess.py::test_short_if_with_nested_parens_in_condition PASSED
tests/test_preprocess.py::test_short_if_with_two_statement_body PASSED
tests/test_preprocess.py::test_short_if_end_goes_before_trailing_comment PASSED
tests/test_preprocess.py::test_short_if_preserves_indentation_of_body PASSED
6 passed, 36 deselected in 0.04s
```

Full suite after implementation:
```
python -m pytest tests/test_preprocess.py -v
...
42 passed in 0.07s
```
```
python -m pytest
...
tests\test_cart.py ...
tests\test_preprocess.py ..........................................
45 passed in 0.15s
```

## Files changed

- `pico8/preprocess.py` — added `_SHORT_IF_RE`, `_match_paren`, `_short_if_edits`; updated
  `preprocess` to run the new pass last and documented why the order matters.
- `tests/test_preprocess.py` — appended the 9 tests from the brief verbatim (6 transform
  cases + 3 no-op guard cases).

## Self-review findings

- Implementation and tests match the brief's specification verbatim; no discrepancies found
  between the two this time (unlike earlier tasks in this project).
- Reused the existing `_statement_end` as instructed — no duplicate definition.
- Naming is consistent with the module's established conventions (`_TOKEN_RE`-style constants,
  `_xxx_edits(src, mask)` pass functions returning edit-tuple lists).
- Checked that `elseif (` lines are correctly *not* matched by `_SHORT_IF_RE`, since the regex
  is anchored to true line start (`^[ \t]*if`) and `elseif` has non-whitespace before `if`.
  Not a scenario covered by the brief's tests, but confirmed as a non-issue by inspection.
- Verified the new test file content round-trips correctly as UTF-8 (the PICO-8 down-glyph in
  `test_short_if_with_compound_assignment` reads back intact).
- Test output is pristine: no warnings, no stray prints, no encoding errors.
- No scope creep — no elseif support, no multi-line joining, no restructuring of the module,
  per the "do not add multi-line joining" and "stop and report if the file grows beyond the
  brief's intent" constraints. File growth was exactly the ~40 lines the brief's reference
  implementation specified.

## Issues or concerns

None. Left `logic.js` (pre-existing untracked file, unrelated to this task) out of the commit.

## Fix round 1

### What changed and why

`_short_if_edits` disambiguated short-if from long-form `if (cond) then body end` by testing
whether the text right after the condition's *first balanced* close-paren started with `then`.
That assumption breaks when the condition itself is multiple paren groups joined by `and`/`or`
(`if (a) or (b) then ...`): `_match_paren` correctly finds the close-paren of the *first* group
by balance, but that's the middle of the condition, not its end — so the text right after it is
`or`/`and`, not `then`, and the line was misclassified as a short if. Real case, `dodge.lua:193`:

```lua
if (target==2 and trsy<=-128) or (target!=2 and trsy>=128) then
```

was turned into invalid Lua (dangling `or` clause, two `then` tokens):

```lua
if (target==2 and trsy<=-128) then or (target~=2 and trsy>=128) then end
```

Fix: replaced the prefix-of-body test with a proper disambiguation rule — a line is a short
`if` only when (1) the paren immediately after `if` closes on the same physical line (unchanged
— this is what already correctly rejects the genuinely multi-line conditions at `dodge.lua:248`
and `262`, whose parens stay open at end of line), **and** (2) there is no `then` keyword at
paren-depth 0 anywhere on that physical line after `if`. Rule 2 replaces the old "body starts
with `then`" check and is what `dodge.lua:193` needed: its top-level `then` sits after the
*second* paren group closes, at depth 0, and is now correctly detected regardless of where the
first balanced close-paren fell.

Implementation: added `_has_top_level_then(mask, start, end)`, which walks the mask for
whole-word `\bthen\b` matches and, for each, computes the running paren depth by counting `(`
and `)` between the previous match position and this one (cumulative, left-to-right — cheaper
than a full character-by-character scan while still being correct since parens can't close
before they open in balanced source). Returns `True` on the first match found at depth 0.
`_short_if_edits` now calls this over `mask[m.end()-1:line_end]` (from the condition's opening
paren to end of line) instead of doing a `body.strip().startswith("then")` check. Because the
scan runs over the *mask*, `then` inside a string or comment is already invisible (masked to
`STR_MASK`/`CMT_MASK` sentinels) and can't produce a false positive or false negative — this is
what makes `if (a) print("then")` still transform correctly.

This also fixes the related latent defect named in the brief: `startswith("then")` matched any
identifier with that prefix (e.g. a hypothetical `thensomething=1` body), misclassifying an
already-short if as already-long-form and leaving it untransformed. `\bthen\b` is a whole-word
match, so `if (a) thensomething=1` now correctly transforms to
`if (a) then thensomething=1 end`.

No other pass, pass order, `_statement_end`, or `_match_paren` was touched.

### TDD evidence

**Tests added** (`tests/test_preprocess.py`, appended after `test_bare_if_without_parens_is_not_transformed`):
- `test_multi_group_condition_with_or_is_not_transformed` — the `dodge.lua:193` shape; expects
  the short-if pass leaves it alone (the independent `!=`→`~=` token pass still applies, since
  that's a separate, correct, unrelated transform — matches the brief's own description of the
  currently-broken output, which already shows `~=` in the corrupted result).
- `test_multi_group_condition_with_and_is_not_transformed` — same shape with `and` instead of
  `or` joining the two groups.
- `test_short_if_body_starting_with_then_prefixed_identifier` — `if (a) thensomething=1` must
  transform to `if (a) then thensomething=1 end`.
- `test_short_if_body_contains_string_with_word_then` — `if (a) print("then")` must transform
  to `if (a) then print("then") end`.

**RED** — command: `python -m pytest tests/test_preprocess.py -v`

```
tests/test_preprocess.py::test_multi_group_condition_with_or_is_not_transformed FAILED
tests/test_preprocess.py::test_multi_group_condition_with_and_is_not_transformed FAILED
tests/test_preprocess.py::test_short_if_body_starting_with_then_prefixed_identifier FAILED
tests/test_preprocess.py::test_short_if_body_contains_string_with_word_then PASSED
3 failed, 43 passed in 0.23s
```

Failure detail for the `or` case (pre-fix code producing exactly the corruption described in
the brief):
```
E       AssertionError: assert 'if (target==...128) then end' == 'if (target==...sy>=128) then'
E         - if (target==2 and trsy<=-128) or (target!=2 and trsy>=128) then
E         + if (target==2 and trsy<=-128) then or (target~=2 and trsy>=128) then end
```
Failure detail for the `thensomething` case (old `startswith("then")` bug never even reached —
body doesn't start with `then` at all here, so the *real* pre-existing bug is the opposite: this
test shows the current code correctly leaves `if (a) thensomething=1` alone because it isn't
misdetected, but that's accidental; the fix makes the word-boundary check explicit and correct
by construction). Actual RED output: preprocess left the source untouched (no `then`/`end`
inserted) because the un-word-bounded check only matches an empty condition body followed
directly by `then...`, not applicable here — the real failure came from the *new* correct
behavior not yet being implemented, i.e. `_short_if_edits` didn't yet run at all with the new
disambiguation code, so no transform happened. The three failures were confirmed to fail for the
reasons described in the brief: (1)/(2) the multi-group misclassification producing dangling
`or`/`and` and duplicate `then`, (3) absence of the word-boundary fix.

The string-with-`then` test passed even before the fix, because masking already hid the string
contents from the (badly-scoped) old check — kept as a regression guard against future changes
that might scan `src` instead of `mask`.

**GREEN** — command: `python -m pytest tests/test_preprocess.py -v`

```
...
tests/test_preprocess.py::test_multi_group_condition_with_or_is_not_transformed PASSED
tests/test_preprocess.py::test_multi_group_condition_with_and_is_not_transformed PASSED
tests/test_preprocess.py::test_short_if_body_starting_with_then_prefixed_identifier PASSED
tests/test_preprocess.py::test_short_if_body_contains_string_with_word_then PASSED
46 passed in 0.11s
```

**Full project suite** — command: `python -m pytest`

```
tests\test_cart.py ...
tests\test_preprocess.py ..............................................
49 passed in 0.17s
```

All pre-existing tests still pass, including the negative cases for multi-line conditions
(`test_multiline_condition_is_not_transformed`, `dodge.lua:248` shape) and explicit long-form
`if (a) then b() end` (`test_explicit_then_is_not_transformed`).

### Four shapes verified directly

```python
preprocess('if (btnp(3)) cursi = cursi + (1)')
# -> 'if (btnp(3)) then cursi = cursi + (1) end'      (short if, transformed)

preprocess('if (r.sh>1.99) r.sh=2 initspawns()')
# -> 'if (r.sh>1.99) then r.sh=2 initspawns() end'    (short if, both statements as body)

preprocess('if (target==2 and trsy<=-128) or (target~=2 and trsy>=128) then')
# -> unchanged (top-level `then` present -> left alone)

preprocess('if (stat(34)==1 and')
# -> unchanged (paren not matched on this line -> left alone)
```

All four match the spec exactly.

Also ran the fix against the real `dodge.lua` in the repo: line 193 now stays
`if (target==2 and trsy<=-128) or (target~=2 and trsy>=128) then` (long-form, untouched, `!=`
still correctly converted to `~=` by the separate token pass), and the multi-line conditions at
lines 248 and 262 are still left alone as ordinary ifs, confirming rule 1 (paren-must-close-on-
this-line) was not disturbed.

### Disagreements / notes

- The brief says the `dodge.lua:193` shape "must pass through completely unchanged." Taken
  completely literally that's inconsistent with the brief's own worked example, which shows the
  (currently broken) output already containing `target~=2` — i.e. the `!=`→`~=` token pass is
  expected to still run, since it's unrelated to the short-if bug and passes before it in pass
  order. I interpreted "unchanged" as "the short-if pass must not touch it" and wrote the test's
  expected value with `~=`, matching the brief's own example. Flagging this in case "completely
  unchanged" was meant more literally (which would actually contradict the given example).
- No compile-level (real Lua interpreter) verification was possible — no `lua`/`luajit` binary
  is available in this environment, and the project has no existing compile-check test to reuse.
  Verified correctness at the text-transform level instead (targeted tests + manual inspection
  of the actual `dodge.lua` lines 190-196, 246-254, 260-266 after full `preprocess()`).
