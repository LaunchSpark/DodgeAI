# Phase 1 progress ledger

Plan: docs/superpowers/plans/2026-08-11-dodge-phase1-playable.md
Branch: design/dodge-env
Base before Task 1: 65e83fd

Task 1: complete (commits 65e83fd..bad67f5, review clean)

## Minor findings deferred to final review
- T1 tools/extract_cart.py:24 — dead imports (CODE, END, GLYPHS) unused in that file
- T1 gfx_lines() vs sprite_sheet() duplicate the nibble-unpack logic in two forms
- T1 pico8/cart.py — no type hints (verbatim move of untyped code)
- T1 tests/test_cart.py — sprite_sheet test checks shape/range but not known pixel values

## Open decision for the human
- logic.js is untracked, so tests gated on it skip silently in a fresh clone.
  Already raised with the user; not blocking (the file is present locally).

Task 2: complete (commit 759bc54, review clean)
- NOTE: implementer committed to `main` as 730c4ae, orphaned from the branch.
  Recovered: cherry-picked onto design/dodge-env as 759bc54; main reset to df9c98e.
  All later dispatches must forbid branch changes.
- Plan bug found: test_mask_hides_line_comments asserted count("=")==1 on input
  with 2 surviving "=". Implementer amended the fixture; reviewer independently
  verified the arithmetic and approved the correction.
- T2 Minor: no tests for unterminated string / unterminated --[[ / escaped quote
  at boundary (all hand-traced correct); apply_edits has no overlap guard; no type hints.

Task 3: implemented 9539ef7, fixes 43cf6bf (re-review pending)
- Review found 2 real bugs, both inherited from the PLAN's example code:
  (a) `?` shorthand inserted its closing paren after a trailing `--` comment,
      emitting unclosed Lua. Fixed by adding _statement_end() and using it.
  (b) _BIN_RE matched `0b` inside identifiers (`x0b101` -> `x5`). Fixed with
      a (?<![\w.]) lookbehind.
  Neither triggered on dodge.lua; both were latent.
- IMPORTANT for Task 4: `_statement_end(mask, pos)` now ALREADY EXISTS in
  pico8/preprocess.py (added by the Task 3 fix). Task 4's brief presents it as
  new code to add — the Task 4 implementer must reuse it, not redefine it.
- T3 Minor: _SHORTHAND_RE's ([ \t]*) capture group is never read; BUTTONS
  inline comment alignment is inconsistent.
- Fix round 2 (2cac4d7): my `(?<![\w.])` instruction was itself wrong — the `.`
  blocked a literal after Lua's `..` operator (`a=a..0b1010` stayed unconverted).
  Lookbehind narrowed to `(?<!\w)`; identifier guards retained. Re-review pending.

Task 3: complete (commits 759bc54..2cac4d7, review clean after 2 fix rounds)
- T3 Minor: tests/test_preprocess.py has two identical binary-literal identifier
  guard tests (round-1 and round-2 additions duplicate each other) - delete one.

Task 4: complete (commit 53a473b, review clean)
- T4 OPEN DECISION (Important, plan-mandated, LATENT): _compound_edits bounds the
  RHS to end of physical line, so `if a then c+=1 end` becomes
  `if a then c = c + (1 end)`. Reviewer grepped all ~90 compound assignments in
  dodge.lua and confirmed NONE are followed by trailing code, so nothing breaks
  today. Pass ordering (compound before short-if) already protects the 3 short-if
  cases at dodge.lua:285,286,919. Fixing properly needs statement-terminator
  awareness (end/then/;/keyword). Needs a human call: fix now or accept as a
  documented limitation like the two-compounds-per-line case.
- T4 Minor: no automated tests for a[b[c]]+=1 / t[i][j]+=1 (both hand-verified OK).

Task 5: implemented 514a058, fix 9a4df13 (re-review pending)
- Review found a REAL (not latent) bug from the plan's example code: the
  short-if rule used the FIRST balanced close-paren, so dodge.lua:193
  `if (target==2 and trsy<=-128) or (target!=2 and trsy>=128) then`
  became `if (...) then or (...) then end` - invalid Lua. Fixed by requiring
  "no top-level `then` on the physical line" (whole-word) instead.
- CONTROLLER VERIFICATION: preprocess(dodge.lua) now COMPILES under LuaJIT 2.1.
  No residual !=, 0b, compound ops or bare glyphs; HUD glyph preserved in strings.

## CRITICAL for Tasks 6, 7, 8: lupa's default runtime is NOT LuaJIT
`lupa.LuaRuntime()` in lupa 2.8 == Lua 5.5, which has no `loadstring` and
differs from PICO-8 semantics (integer/float split affects tostring, unpack,
etc). Use `from lupa.luajit21 import LuaRuntime` everywhere.
Available: luajit21, luajit20, lua51..lua55. The plan's code says
`lupa.LuaRuntime()` and must be corrected in every remaining dispatch.

Task 5: complete (commits 53a473b..9a4df13, review clean after 1 fix round)
- T5 Minor (latent): `if (a) if (b) then c() end` left untransformed - the
  top-level-`then` rule can't tell a body statement's `then` from the condition's.
  Only `if (`...`then` line in the whole cart is 193 itself.
- USER DECISION: accept the T4 compound-RHS limitation as documented, not fixed.
  Recorded in the spec's Known limitations (commit follows).

Task 6: implemented d5aae44, fixes ed0d9d0 + cd4fec8 (re-review pending)
- Review found the flagship test was VACUOUS: `return loadstring(src)` tail-calls,
  so lupa received a (nil, err) TUPLE, and `assert chunk is not None` could never
  fail. Fixed by capturing `local f, e = ...` and returning only `e`; added
  test_compile_check_rejects_invalid_lua as a permanent guard.
- Controller found suite output was not pristine: LuaJIT uses Windows SEH, so
  pytest's faulthandler dumped a fake "fatal exception" trace when the guard test
  compiled invalid Lua. Suppressed narrowly around that one call.
- CONFIRMED: 53 passed, 0 skipped, no fatal-exception text.
Task 6: complete (commits ad977d6..cd4fec8, review clean after 2 fix rounds)
- T6 Minor: check_lua.py error-line parse `split(":")[1]` fragile if a chunkname
  ever contains a colon (masked in practice); module-level skipif also gates the
  guard test which doesn't need dodge.lua.

Task 7: implemented 01050d5 (review pending)
Task 7: complete (commit 01050d5, review clean)
- T7 OPEN (Important, LATENT): all() skips elements when a list holds DUPLICATE
  scalar values and one is deleted mid-iteration, because the iterator detects
  deletion by value comparison (t[i]==prev). Reviewer confirmed every real call
  site in dodge.lua passes lists of unique table references, so unreachable today.
  NOTE: PICO-8's own all() has this same value-comparison quirk, so documenting
  it is arguably more faithful than "fixing" it. -> document in prelude.lua.
- T7 Minor: tostr/sub have no committed tests; mid()-equal-to-bound, sgn(-0.0)
  and rnd({}) verified correct by the reviewer but not pinned by tests.

Task 8: complete (commit d409f2f, review clean - no Critical/Important)
- THE GAME RUNS HEADLESS. Controller-verified: boots, 66 splash flips during
  execute, enemies spawn (peak 15), idle player dies at frame 194 (seed 1).
- Reviewer mutation-tested the rewritten determinism test: forcing srand(0)
  makes it fail, so it is NOT vacuous.
- Fixing now (relevant to Tasks 9-11): btnp repeat fires 3 frames late
  ([1,19,23..] vs documented [1,16,20..]); draw lambdas early-bind the backend
  while flip late-binds it, so swapping backends post-construction would draw
  to the old one silently.
- T8 note for PHASE 2 (not phase 1): dodge.lua:204-213 backtomenu(1) exits the
  round on btnp(4)/O. An RL action space including bits 4/5 could silently
  leave the game state start_game() installed. Exclude them or re-assert.
- T8 Minor deferred: Machine.flip() doesn't advance the frame counter (latent);
  boot()'s default logic.js path is cwd-relative so pytest from elsewhere skips
  silently; load_sheet is duck-typed rather than on the Backend interface.

Task 9: implemented 24adf91 (review pending)
- Implementer found 2 more real bugs in the plan's example code:
  (a) cls() reset the camera offset, which would have silently defeated screen
      shake (the cart calls camera(shakex,shakey) immediately before cls(12)).
  (b) pygame.transform.scale() into the 32-bit window raises ValueError from an
      8-bit source; fixed by scaling to a fresh surface then blitting.
- CONTROLLER-VERIFIED RENDERING: after 120 frames of real gameplay the surface
  holds exactly the game's signature - 16266 px blue (colour 12) background,
  92 px white (colour 7) entities, 26 px dark-blue (colour 1) drop shadows.
- NEW Minor for final cleanup: suite now reports "1 warning" - third-party
  DeprecationWarning from pygame/pkgdata.py importing pkg_resources. Not our
  code. Add a filterwarnings entry so real warnings stay visible.
Task 9: complete (commit 24adf91, review clean - no Critical/Important)
- Reviewer independently confirmed BOTH plan bugs were real: reproduced the
  transform.scale ValueError, and traced dodge.lua:2245,2258-2259 to prove
  camera() runs immediately before cls() every frame.
- T9 Minor -> folded into Task 10 dispatch: code comment + test docstring in
  backends.py/test_pygame_backend.py misattribute the camera/cls sequence to
  updatemenu(); it is actually the global _draw() (dodge.lua:2214-2262). Also
  they call shake a settings-screen effect when shake+=0.07 fires on real
  gameplay hits (dodge.lua:565,746,875).

Task 10: implemented 39510ff (review pending)
- Text overlay + glyph icons; corrected the T9 misattributed citations.
- Implementer verified by pixel-diff that text renders in the real game
  (2472 px differ on the title screen, 414 px on the score HUD vs a stubbed
  _draw_text), plus saved screenshots.
- Implementer flagged pico8/backends.py as growing large (interface +
  NullBackend + glyph helpers + PygameBackend) - deferred to final review.

## FINAL CLEANUP LIST (for the final whole-branch review fix wave)
1. pygame/pkgdata.py DeprecationWarning -> add a filterwarnings entry so the
   suite output is pristine and real warnings stay visible.
2. Document all()'s duplicate-value limitation in prelude.lua (T7, latent;
   PICO-8's own all() shares the quirk).
3. Consider splitting pico8/backends.py.
4. Delete the duplicated binary-literal identifier guard test (T3).
5. Deferred T8 minors: Machine.flip() doesn't advance the frame counter;
   boot()'s default logic.js path is cwd-relative so a run from another
   directory skips silently; load_sheet is duck-typed off the Backend interface.
6. T7 minors: tostr/sub untested; mid()-at-bound, sgn(-0.0), rnd({}) unpinned.
7. T6 minor: check_lua.py's error-line parse is fragile if a chunkname contains
   a colon.
Task 10: complete (commit 39510ff, review clean - no Critical/Important)
- T10 Minor -> cleanup list: no automated test covers _draw_text itself (the
  brief only specified split_glyphs tests). Advance math, palette lookup and
  camera-adjusted positioning have no regression coverage.

Task 11: implemented 19f25e5 (review pending)
- CONTROLLER-VERIFIED: play.py --frames 300 exits 0 with no traceback; 100 tests
  pass; extract_cart.py still reproduces dodge.lua/dodge.p8 byte-for-byte.
- Captured real frames: title screen shows the Dodge logo sprite, "press [X]"
  with the glyph icon, and all three credits; gameplay shows the score HUD and
  the player. THE GAME IS PLAYABLE.
