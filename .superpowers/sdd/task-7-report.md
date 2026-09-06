# Task 7 Report: Lua prelude — tables, math, seeded PRNG

## What I implemented

- `pico8/prelude.lua` — a pure-Lua chunk defining PICO-8's table/math/PRNG
  builtins: `add del all foreach count mid flr ceil abs min max sqrt sgn
  sin cos rnd srand tostr sub`. Implemented in Lua (not Python) because
  the cart calls these thousands of times per frame; crossing the
  Python/Lua boundary for `add`/`all` would dominate the frame budget.
- `tests/test_prelude.py` — tests each builtin through a real LuaJIT
  runtime (`lupa.luajit21`), following the same convention already used
  in `tests/test_preprocess_cart.py`.

Used the brief's Lua implementation verbatim (Step 3) after verifying by
hand-trace (see below) that its `all()`/`del()` interaction is correct.
Used the brief's test cases verbatim, but with two corrections/additions:

1. **Mandatory correction**: switched `lupa.LuaRuntime()` to
   `lupa.luajit21`'s `LuaRuntime(unpack_returned_tuples=True)`, bound to
   the name `luajit` per dispatch instructions, matching
   `test_preprocess_cart.py`'s existing convention. Lupa 2.8's default
   runtime is Lua 5.5, whose integer/float split and moved `unpack`
   differ from PICO-8's Lua-5.1-like semantics.
2. **Added tests beyond the brief**: the brief's given test for `all()`
   deletion only exercises deleting elements at positions 2 and 4 of a
   5-element list (both "middle-ish", never index 1 or the last index).
   Since the task explicitly calls out tracing first/middle/last by
   hand, I added `test_all_survives_deletion_of_first_element` and
   `test_all_survives_deletion_of_last_element` to turn that hand-trace
   into an executable regression test. I also added
   `test_foreach_calls_function_on_each_element` and
   `test_count_returns_length` since `foreach`/`count` are part of the
   brief's declared interface but the brief's test file never exercised
   them.

I did **not** modify the given implementation in `prelude.lua` or the
given tests — both checked out on inspection (see hand-trace and
semantics verification below).

## What I tested and the results

Ran with `python -m pytest`, Python 3.12, Windows.

- `tests/test_prelude.py` alone: **14 passed**, 0 skipped, 0 warnings
  (verified with `-W error` too — pristine, no stray output).
- Full suite (`python -m pytest -q`): **67 passed** (53 pre-existing +
  14 new), 0 skipped, 0 failed. This is the real number — the brief's
  "11 passed" / "Expected: N passed" figures are stale per the dispatch
  note.

Explicit confirmation tests **executed** rather than skipped: the
`pytest -v` output below shows `PASSED` (not `SKIPPED`) for all 14
tests. Earlier in the RED phase I had accidentally added a
`pytest.mark.skipif(not PRELUDE.exists(), ...)` (copy-pasted from
`test_preprocess_cart.py`'s dodge.lua-not-extracted guard) which caused
all 14 to report `SKIPPED` instead of failing — caught this during RED
verification (a test that skips is not a test that passes) and removed
it, since `prelude.lua` is a file this task creates, not an optional
build artifact.

I also ran extra ad-hoc sanity checks outside the formal suite (not
committed, just verification):
```
rnd(0)        = 0
rnd()         = 0.13153778773875702   (no-arg form, [0,1))
mid(5,5,5)    = 5
sgn(5)        = 1
count(nil)    = 0
tostr(nil)    = '[nil]'
sub("hello",2,3) = "el"
```
All consistent with PICO-8 semantics.

## TDD Evidence

**RED**

Command: `python -m pytest tests/test_prelude.py -v` (before creating
`pico8/prelude.lua`)

```
E       FileNotFoundError: [Errno 2] No such file or directory: 'pico8\\prelude.lua'
...
ERROR tests/test_prelude.py::test_all_survives_deletion_during_iteration - Fi...
... (14 errors total)
============================= 14 errors in 0.80s ==============================
```

This is exactly the failure the brief predicts (Step 2: "FAIL with
`FileNotFoundError: pico8/prelude.lua`") — the test file imports and
reads a Lua file that doesn't exist yet, so every test errors before
the runtime is even constructed. Expected because no implementation
exists.

**GREEN**

Command: `python -m pytest tests/test_prelude.py -v` (after creating
`pico8/prelude.lua`)

```
tests/test_prelude.py::test_all_survives_deletion_during_iteration PASSED [  7%]
tests/test_prelude.py::test_all_survives_deletion_of_first_element PASSED [ 14%]
tests/test_prelude.py::test_all_survives_deletion_of_last_element PASSED [ 21%]
tests/test_prelude.py::test_all_on_empty_and_nil PASSED                  [ 28%]
tests/test_prelude.py::test_add_returns_value_and_appends PASSED         [ 35%]
tests/test_prelude.py::test_del_removes_first_match_only PASSED          [ 42%]
tests/test_prelude.py::test_foreach_calls_function_on_each_element PASSED [ 50%]
tests/test_prelude.py::test_count_returns_length PASSED                  [ 57%]
tests/test_prelude.py::test_mid_clamps_like_pico8 PASSED                 [ 64%]
tests/test_prelude.py::test_sgn_of_zero_is_one PASSED                    [ 71%]
tests/test_prelude.py::test_sin_is_inverted_and_turn_based PASSED        [ 78%]
tests/test_prelude.py::test_rnd_is_deterministic_for_a_seed PASSED       [ 85%]
tests/test_prelude.py::test_rnd_differs_between_seeds PASSED             [ 92%]
tests/test_prelude.py::test_rnd_range_and_table_form PASSED              [100%]
============================= 14 passed in 0.05s ==============================
```

Full suite: `python -m pytest -q` → `67 passed in 0.27s`.

## Hand-trace of `all()` under deletion

Implementation:
```lua
function all(t)
  if t == nil or #t == 0 then return function() end end
  local i, prev = 1, nil
  return function()
    if t[i] == prev then i = i + 1 end
    while i <= #t and t[i] == nil do i = i + 1 end
    prev = t[i]
    return t[i]
  end
end
```
`del(t,v)` removes the first element equal to `v` via `table.remove`,
which shifts every later element down by one index.

**Delete the first element** — `t={1,2,3}`, delete when `v==1`:

| call | i (before) | prev | t before | t[i]==prev? | i (after) | returns | action |
|---|---|---|---|---|---|---|---|
| 1 | 1 | nil | {1,2,3} | no | 1 | 1 | `del(t,1)` → t={2,3} |
| 2 | 1 | 1 | {2,3} | no (t[1]=2) | 1 | 2 | — |
| 3 | 1 | 2 | {2,3} | yes | 2 | 3 | — |
| 4 | 2 | 3 | {2,3} | yes | 3 | nil (3>#t=2) | stop |

Visited: `1,2,3`. Correct — deleting the element just yielded, even
when it's the first one, doesn't skip or repeat anything, because the
`t[i]==prev` check advances past a slot that's already been shifted
into by the deletion.

**Delete a middle element** — `t={1,2,3}`, delete when `v==2`: by the
same mechanics, visits `1,2,3`. (The brief's own test additionally
covers deleting two middle-ish elements — indices 2 and 4 of a 5-element
list — and I confirmed by the same method it also yields `1,2,3,4,5`.)

**Delete the last element** — `t={1,2,3}`, delete when `v==3`:

| call | i (before) | prev | t before | t[i]==prev? | i (after) | t[i] | returns |
|---|---|---|---|---|---|---|---|
| 1 | 1 | nil | {1,2,3} | no | 1 | 1 | 1 |
| 2 | 1 | 1 | {1,2,3} | yes | 2 | 2 | 2 |
| 3 | 2 | 2 | {1,2,3} | yes | 3 | 3 | 3, then `del(t,3)` → t={1,2} |
| 4 | 3 | 3 | {1,2} | t[3]=nil, nil==3 is **false** | 3 (unchanged) | nil (3>#t=2, while skipped) | nil → stop |

Visited: `1,2,3`, then terminates cleanly. The important subtlety here:
after the last element is removed, `t[i]` (`t[3]`) is `nil`, and `nil ==
prev` (`nil == 3`) is false in Lua, so the "advance past consumed slot"
branch does *not* fire — but that's fine, because the subsequent `while
i <= #t and t[i] == nil` loop condition (`3 <= 2`) is already false, so
the function falls through to `prev = t[i]; return t[i]`, both `nil`,
and the `for` loop terminates normally. No index error, no infinite
loop.

**Conclusion**: the brief's implementation and its given test's expected
output (`"1,2,3,4,5"`) are both correct. I did not need to adjust
either. I added two extra tests (first-element and last-element
deletion) to make the hand-trace above an executable, permanent check
rather than something verified only once by inspection.

## `mid()` semantics verification

`mid(a,b,c) = max(min(a,b), min(max(a,b), c))`, used by the cart as
`mid(2, px, 125)` (clamp `px` into `[2,125]`):

- `mid(2,200,125)`: `min(2,200)=2`; `max(2,200)=200`; `min(200,125)=125`;
  `max(2,125)=125` → **125** (value above range clamps to the upper
  bound). Matches expected.
- `mid(2,-5,125)`: `min(2,-5)=-5`; `max(2,-5)=2`; `min(2,125)=2`;
  `max(-5,2)=2` → **2** (value below range clamps to the lower bound).
  Matches expected.
- `mid(2,50,125)`: `min(2,50)=2`; `max(2,50)=50`; `min(50,125)=50`;
  `max(2,50)=50` → **50** (value already inside range passes through
  unchanged). Matches expected.

All three argument-order cases the game actually uses (value inside,
below, and above the clamp bounds) are correct.

## Files changed

- `d:\projects\DodgeAI\pico8\prelude.lua` (new, 85 lines)
- `d:\projects\DodgeAI\tests\test_prelude.py` (new, 114 lines)

Both added and committed; nothing else touched. `logic.js` (pre-existing
untracked file, unrelated to this task) was left alone.

## Self-review findings

- Fixed during RED verification: an incorrectly copy-pasted `skipif`
  marker was silently turning all 14 new tests into skips instead of
  failures/passes. Removed before proceeding — this is exactly the "a
  test that skips is not a test that passes" trap called out in the
  dispatch instructions.
- Verified all 19 names in the brief's declared interface
  (`add del all foreach count mid flr ceil abs min max sqrt sgn sin cos
  rnd srand tostr sub`) are defined in `prelude.lua`, with none extra —
  no scope creep beyond the brief.
- Ran the full suite with `-W error` to confirm no warnings are hiding
  behind pytest's default filters; output is unchanged (67 passed, no
  warnings).
- Confirmed file sizes are in line with the brief's intent (85/114
  lines) — no restructuring needed, no report of `DONE_WITH_CONCERNS`
  for scope reasons.
- Extra ad-hoc checks (`rnd(0)`, no-arg `rnd()`, `mid` with equal args,
  `sgn(5)`, `count(nil)`, `tostr(nil)`, `sub`) all behaved sensibly and
  consistently with PICO-8 semantics, though these aren't part of the
  committed suite (they're edge cases not called out by the brief's
  interface contract, kept out to avoid speculative test bloat beyond
  what's needed).

## Issues or concerns

None. The brief's implementation and given tests were correct as
written (verified by hand-trace and semantics check above), the
mandatory LuaJIT-runtime correction was applied, and coverage was
extended to the interface members (`foreach`, `count`) and edge cases
(first/last-element deletion) the brief's own test file didn't reach.
