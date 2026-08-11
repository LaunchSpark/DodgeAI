# Dodge Phase 1 — Locally Playable Build

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python play.py` opens a window and plays the PICO-8 game `dodge` — menus, settings, gameplay and all — with no PICO-8 licence and no browser.

**Architecture:** The real, unmodified `dodge.lua` runs inside LuaJIT (via `lupa`). A Python layer supplies the PICO-8 API: a Lua prelude for table/math builtins, and Python callbacks for input, memory and drawing. Drawing goes through a `Backend` interface so the same cart runs headless (phase 2) or on screen.

**Tech Stack:** Python 3.12, `lupa` (LuaJIT), `pygame`, `pytest`.

## Global Constraints

- **`dodge.lua` is never edited.** It is a build artefact of `tools/extract_cart.py`. Any behaviour change is made in Python or in an injected Lua chunk.
- **Single RNG stream.** The cart draws gameplay and cosmetic randomness from one stream. Never skip a function that can consume `rnd()` — see `spawntrail` inside `drawgame`.
- **The frame entry point is `_draw()`.** `_update60` exists but is empty; all game logic runs in `_draw`.
- **Python 3.12, `lupa>=2.8`** (prebuilt `cp312-win_amd64` wheel — no compiler needed).
- **Glyph substitution must skip string literals.** Bare `❎` is button 5; `"press ❎"` is HUD text.
- Button numbers: `⬅️`=0, `➡️`=1, `⬆️`=2, `⬇️`=3, `🅾️`=4, `❎`=5.
- PICO-8 palette indices are 0-15; drawing surfaces are 8-bit indexed so `pal(c, c2, 1)` works by swapping the palette.

## File Structure

| File | Responsibility |
| --- | --- |
| `pico8/cart.py` | Read the ROM from `logic.js`, decompress PXA code, decode the sprite sheet |
| `pico8/preprocess.py` | PICO-8 Lua dialect → standard Lua 5.1 |
| `pico8/prelude.lua` | Pure-Lua builtins: tables, math, seeded PRNG |
| `pico8/api.py` | Python-backed builtins: input, memory, drawing dispatch |
| `pico8/backends.py` | `Backend` protocol, `NullBackend`, `PygameBackend` |
| `pico8/machine.py` | Boot the cart, own machine state, step one frame |
| `play.py` | Window, 60 Hz loop, keyboard/mouse → button state |
| `tools/extract_cart.py` | Existing tool; refactored to import from `pico8/cart.py` |

---

### Task 1: Cart loading module

Move ROM reading out of the standalone tool so the runtime and the tool share one decoder.

**Files:**
- Create: `pico8/__init__.py`, `pico8/cart.py`, `requirements.txt`, `tests/__init__.py`, `tests/test_cart.py`
- Modify: `tools/extract_cart.py` (replace its ROM/PXA code with imports)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `pico8.cart.read_rom(js_path: Path) -> tuple[bytes, str]` — 32,768-byte ROM and cart name
  - `pico8.cart.decompress_code(rom: bytes) -> str` — Lua source with P8SCII mapped to UTF-8
  - `pico8.cart.sprite_sheet(rom: bytes) -> list[list[int]]` — 128 rows × 128 palette indices
  - `pico8.cart.GLYPHS: dict[int, str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cart.py
from pathlib import Path
import pytest
from pico8 import cart

ROM = Path("logic.js")
pytestmark = pytest.mark.skipif(not ROM.exists(), reason="logic.js not present")


def test_read_rom_returns_full_cart():
    rom, name = cart.read_rom(ROM)
    assert len(rom) == 32768
    assert name == "dodge.p8"


def test_decompress_code_matches_known_size():
    rom, _ = cart.read_rom(ROM)
    code = cart.decompress_code(rom)
    assert len(code) == 37110          # 37,101 bytes, 9 of them multi-byte glyphs
    assert code.startswith("\npoke(0x5ffe,1)")
    assert "function updategame()" in code


def test_sprite_sheet_shape_and_content():
    rom, _ = cart.read_rom(ROM)
    sheet = cart.sprite_sheet(rom)
    assert len(sheet) == 128
    assert all(len(row) == 128 for row in sheet)
    assert all(0 <= px <= 15 for row in sheet for px in row)
    assert any(px != 0 for row in sheet for px in row)   # sheet is not blank
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cart.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pico8'`

- [ ] **Step 3: Create the package and move the decoder**

Create `requirements.txt`:

```text
lupa>=2.8
pygame>=2.5
pytest>=8.0
```

Install: `pip install -r requirements.txt`

Create `pico8/__init__.py` (empty) and `tests/__init__.py` (empty).

Create `pico8/cart.py` by moving these pieces out of `tools/extract_cart.py` verbatim: the ROM offset constants, `P8SCII`/`GLYPHS`, `read_cartdat` (renamed `read_rom`), `BitReader`, and `decompress_code`. Change `sys.exit(...)` calls to `raise ValueError(...)` so the library never kills the host process. Then add the sprite decoder:

```python
def sprite_sheet(rom):
    """Decode ROM 0x0000-0x1fff into 128 rows of 128 palette indices.

    Two pixels per byte, low nibble first — the same layout the .p8 __gfx__
    section uses.
    """
    rows = []
    for y in range(128):
        base = GFX + y * 64
        row = []
        for b in rom[base:base + 64]:
            row.append(b & 0xF)
            row.append(b >> 4)
        rows.append(row)
    return rows
```

- [ ] **Step 4: Point the tool at the module**

In `tools/extract_cart.py`, delete the moved code and add near the top:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pico8.cart import GFX, MAP, GFF, MUSIC, SFX, CODE, END, GLYPHS, read_rom, decompress_code
```

Replace the `read_cartdat(args.js)` call in `main()` with `read_rom(args.js)`.

- [ ] **Step 5: Run tests and verify the tool still round-trips**

Run: `python -m pytest tests/test_cart.py -v`
Expected: 3 passed

Run: `python tools/extract_cart.py --out /tmp/check`
Expected: prints `dodge.lua  37,110 chars` and `dodge.p8  64,909 chars`

Run: `git diff --stat -- dodge.lua dodge.p8`
Expected: no output — regenerating produces byte-identical files

- [ ] **Step 6: Commit**

```bash
git add pico8 tests requirements.txt tools/extract_cart.py
git commit -m "refactor: extract cart decoding into pico8.cart"
```

---

### Task 2: Literal masking and edit application

The preprocessor's foundation. Every later transform runs its regex against a *mask* where string and comment contents are replaced by sentinels, then applies edits to the original text by index. This is what makes "skip string literals" automatic for every rule.

**Files:**
- Create: `pico8/preprocess.py`, `tests/test_preprocess.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `pico8.preprocess.STR_MASK = "\x00"`, `pico8.preprocess.CMT_MASK = "\x01"`
  - `pico8.preprocess.mask_literals(src: str) -> str` — same length as `src`
  - `pico8.preprocess.apply_edits(src: str, edits: list[tuple[int, int, str]]) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocess.py
from pico8.preprocess import STR_MASK, CMT_MASK, mask_literals, apply_edits


def test_mask_preserves_length_and_newlines():
    src = 'a = "hi"  -- note\nb = 1\n'
    mask = mask_literals(src)
    assert len(mask) == len(src)
    assert mask.count("\n") == src.count("\n")


def test_mask_hides_string_contents():
    mask = mask_literals('print("press x")')
    assert "press" not in mask
    assert mask.startswith("print(")
    assert mask.count(STR_MASK) == len('"press x"')


def test_mask_hides_line_comments():
    mask = mask_literals("x = 1 -- x = 2\ny = 3")
    assert mask.count("=") == 1
    assert mask.endswith("y = 3")


def test_mask_hides_block_comments_but_keeps_newlines():
    src = "a=1\n--[[\n b=2\n]]\nc=3"
    mask = mask_literals(src)
    assert "b=2" not in mask
    assert mask.count("\n") == 4
    assert mask.endswith("c=3")


def test_apply_edits_runs_right_to_left():
    src = "abcdef"
    out = apply_edits(src, [(0, 1, "X"), (4, 6, "YY_Y")])
    assert out == "XbcdYY_Y"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pico8.preprocess'`

- [ ] **Step 3: Implement**

```python
# pico8/preprocess.py
"""Translate PICO-8's Lua dialect into standard Lua 5.1.

Every rule matches against a *mask* of the source in which string and comment
contents are replaced by sentinel characters, then edits the original by index.
That makes "never rewrite inside a literal" a property of the whole module
rather than something each rule has to remember.
"""

STR_MASK = "\x00"
CMT_MASK = "\x01"


def mask_literals(src):
    """Return a same-length copy with literal and comment contents masked.

    Newlines are preserved everywhere, including inside block comments, so
    line-oriented rules still see the real line structure.
    """
    out = list(src)
    i, n = 0, len(src)

    def blank(start, stop, sentinel):
        for k in range(start, min(stop, n)):
            if out[k] != "\n":
                out[k] = sentinel

    while i < n:
        if src.startswith("--[[", i):
            end = src.find("]]", i + 4)
            end = n if end == -1 else end + 2
            blank(i, end, CMT_MASK)
            i = end
        elif src.startswith("--", i):
            end = src.find("\n", i)
            end = n if end == -1 else end
            blank(i, end, CMT_MASK)
            i = end
        elif src[i] == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            blank(i, j + 1, STR_MASK)
            i = j + 1
        else:
            i += 1
    return "".join(out)


def apply_edits(src, edits):
    """Apply (start, end, replacement) edits, latest offset first."""
    for start, end, repl in sorted(edits, key=lambda e: e[0], reverse=True):
        src = src[:start] + repl + src[end:]
    return src
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add pico8/preprocess.py tests/test_preprocess.py
git commit -m "feat: literal masking and indexed edits for the preprocessor"
```

---

### Task 3: Token transforms — glyphs, `!=`, binary literals, `?`

**Files:**
- Modify: `pico8/preprocess.py`, `tests/test_preprocess.py`

**Interfaces:**
- Consumes: `mask_literals`, `apply_edits`, `STR_MASK`, `CMT_MASK`
- Produces: `pico8.preprocess.BUTTONS: dict[str, int]`, and internal
  `_token_edits(src: str, mask: str) -> list[tuple[int, int, str]]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_preprocess.py`:

```python
import pytest
from pico8.preprocess import preprocess


@pytest.mark.parametrize("src,expected", [
    ("if btnp(❎) then x() end", "if btnp(5) then x() end"),
    ("if btnp(\U0001f17e️) then x() end", "if btnp(4) then x() end"),
    ("btnp(⬅️)", "btnp(0)"),
    ("btnp(➡️)", "btnp(1)"),
    ("btnp(⬆️)", "btnp(2)"),
    ("btnp(⬇️)", "btnp(3)"),
])
def test_bare_glyphs_become_button_numbers(src, expected):
    assert preprocess(src) == expected


def test_glyphs_inside_strings_are_left_alone():
    # dodge.lua:511 — HUD text telling the player which button to press
    src = 'print2("press "..(input==1 and "❎" or "  ").." to play again",20,120)'
    assert preprocess(src) == src


def test_not_equal_becomes_lua_inequality():
    assert preprocess("if a!=b then end") == "if a~=b then end"


def test_not_equal_inside_string_is_left_alone():
    assert preprocess('s="a!=b"') == 's="a!=b"'


def test_fixed_point_binary_literal():
    # dodge.lua:1051
    assert preprocess("fillp(0b1010010110100101.1)") == "fillp(42405.5)"


def test_plain_binary_literal():
    assert preprocess("x=0b1010") == "x=10"


def test_question_mark_shorthand():
    # dodge.lua:6
    assert preprocess(" ?_txt,_x,_y+1,1") == " print(_txt,_x,_y+1,1)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: FAIL with `ImportError: cannot import name 'preprocess'`

- [ ] **Step 3: Implement**

Add to `pico8/preprocess.py`:

```python
import re

BUTTONS = {
    "⬅️": 0,          # left
    "➡️": 1,          # right
    "⬆️": 2,          # up
    "⬇️": 3,          # down
    "\U0001f17e️": 4,      # O
    "❎": 5,                # X
}

_GLYPH_RE = re.compile("|".join(re.escape(g) for g in BUTTONS))
_NEQ_RE = re.compile(r"!=")
_BIN_RE = re.compile(r"0b([01]+)(?:\.([01]+))?")
_SHORTHAND_RE = re.compile(r"(?m)^([ \t]*)\?")


def _token_edits(src, mask):
    edits = []

    for m in _GLYPH_RE.finditer(mask):
        edits.append((m.start(), m.end(), str(BUTTONS[m.group(0)])))

    for m in _NEQ_RE.finditer(mask):
        edits.append((m.start(), m.end(), "~="))

    for m in _BIN_RE.finditer(mask):
        value = float(int(m.group(1), 2))
        if m.group(2):
            value += int(m.group(2), 2) / (1 << len(m.group(2)))
        text = str(int(value)) if value.is_integer() else str(value)
        edits.append((m.start(), m.end(), text))

    for m in _SHORTHAND_RE.finditer(mask):
        end = mask.find("\n", m.end())
        end = len(mask) if end == -1 else end
        edits.append((m.end() - 1, m.end(), "print("))
        edits.append((end, end, ")"))

    return edits


def preprocess(src):
    """Translate a PICO-8 Lua source string into standard Lua 5.1."""
    mask = mask_literals(src)
    return apply_edits(src, _token_edits(src, mask))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add pico8/preprocess.py tests/test_preprocess.py
git commit -m "feat: glyph, inequality, binary literal and ? transforms"
```

---

### Task 4: Compound assignment

`lhs op= rhs` becomes `lhs = lhs op (rhs)`. The left-hand side may be a plain name, a field path, or an index expression containing its own operators — `enemiestats[e.p+1]-=2` at line 2222 is the case that defeats a naive regex.

**Files:**
- Modify: `pico8/preprocess.py`, `tests/test_preprocess.py`

**Interfaces:**
- Consumes: `mask_literals`, `apply_edits`, `CMT_MASK`
- Produces: internal `_compound_edits(src: str, mask: str) -> list[tuple[int, int, str]]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_preprocess.py`:

```python
@pytest.mark.parametrize("src,expected", [
    ("cursi+=1", "cursi = cursi + (1)"),
    ("shake*=0.95", "shake = shake * (0.95)"),
    ("_p.age+=1", "_p.age = _p.age + (1)"),
    ("_e.x-=0.5*frz", "_e.x = _e.x - (0.5*frz)"),
    ("enemiestats[i]+=1", "enemiestats[i] = enemiestats[i] + (1)"),
])
def test_compound_assignment_shapes(src, expected):
    assert preprocess(src) == expected


def test_compound_assignment_with_operator_inside_index():
    # dodge.lua:2222 — the index expression contains a '+' of its own
    assert preprocess("enemiestats[e.p+1]-=2") == \
        "enemiestats[e.p+1] = enemiestats[e.p+1] - (2)"


def test_compound_assignment_with_nested_index():
    # dodge.lua:621
    assert preprocess("patterns[ix].prob-=6*patterns[ix].counter") == \
        "patterns[ix].prob = patterns[ix].prob - (6*patterns[ix].counter)"


def test_compound_assignment_stops_before_trailing_comment():
    assert preprocess("size+=0.045 -- ramp") == "size = size + (0.045) -- ramp"


def test_equality_and_comparison_are_untouched():
    assert preprocess("if a==b and c>=d and e<=f then end") == \
        "if a==b and c>=d and e<=f then end"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocess.py -k compound -v`
Expected: FAIL — output still contains `+=`

- [ ] **Step 3: Implement**

Add to `pico8/preprocess.py`:

```python
_COMPOUND_RE = re.compile(r"(?<![=<>~!+\-*/%])([+\-*/%])=(?!=)")


def _lvalue_start(mask, pos):
    """Walk left from an operator to the start of the assignment target.

    Handles plain names, dotted fields, and index expressions whose contents
    may themselves contain operators (enemiestats[e.p+1]).
    """
    i = pos
    while i > 0 and mask[i - 1] in " \t":
        i -= 1
    while i > 0:
        c = mask[i - 1]
        if c == "]":
            depth, j = 0, i
            while j > 0:
                if mask[j - 1] == "]":
                    depth += 1
                elif mask[j - 1] == "[":
                    depth -= 1
                    if depth == 0:
                        j -= 1
                        break
                j -= 1
            i = j
        elif c.isalnum() or c in "_.":
            i -= 1
        else:
            break
    return i


def _statement_end(mask, pos):
    """End of the statement: end of line, excluding any trailing comment."""
    end = mask.find("\n", pos)
    end = len(mask) if end == -1 else end
    while end > pos and (mask[end - 1] == CMT_MASK or mask[end - 1] in " \t"):
        end -= 1
    return end


def _compound_edits(src, mask):
    edits = []
    for m in _COMPOUND_RE.finditer(mask):
        op = m.group(1)
        start = _lvalue_start(mask, m.start())
        end = _statement_end(mask, m.end())
        lhs = src[start:m.start()].strip()
        rhs = src[m.end():end].strip()
        if not lhs or not rhs:
            continue
        edits.append((start, end, f"{lhs} = {lhs} {op} ({rhs})"))
    return edits
```

Extend `preprocess` to run both passes. They must be separate passes over freshly computed masks, because the first pass changes offsets:

```python
def preprocess(src):
    """Translate a PICO-8 Lua source string into standard Lua 5.1."""
    for pass_fn in (_token_edits, _compound_edits):
        src = apply_edits(src, pass_fn(src, mask_literals(src)))
    return src
```

**Known limitation:** because the right-hand side runs to end of line, two compound
assignments on one line would produce overlapping edits. No line in `dodge.lua` has two, so
this is left unhandled rather than solved speculatively.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: 24 passed

- [ ] **Step 5: Commit**

```bash
git add pico8/preprocess.py tests/test_preprocess.py
git commit -m "feat: compound assignment transform"
```

---

### Task 5: One-line `if`

`if (cond) body` becomes `if (cond) then body end`. Keeping the original parentheses means the condition never has to be reconstructed — only two insertions are needed.

The rule that separates a short `if` from an ordinary multi-line one: **the opening paren must be matched on the same physical line**, with non-`then` content following it. Lines 248 and 262 leave their parens open at end of line, so they are skipped without any lookahead.

**Files:**
- Modify: `pico8/preprocess.py`, `tests/test_preprocess.py`

**Interfaces:**
- Consumes: `mask_literals`, `apply_edits`, `CMT_MASK`
- Produces: internal `_short_if_edits(src: str, mask: str) -> list[tuple[int, int, str]]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_preprocess.py`:

```python
def test_short_if_simple():
    assert preprocess("if (input==0) spr(192,71,56,1,2)") == \
        "if (input==0) then spr(192,71,56,1,2) end"


def test_short_if_with_compound_assignment():
    # dodge.lua:285 — glyph, short if and compound assignment on one line
    assert preprocess("if (btnp(⬇️)) cursi+=1") == \
        "if (btnp(3)) then cursi = cursi + (1) end"


def test_short_if_with_nested_parens_in_condition():
    # dodge.lua:397 — closing paren must be found by balance, not first match
    src = 'if ((isdead or not hasplayed) or n!="gameplay") print2("<  >",65,y)'
    assert preprocess(src) == \
        'if ((isdead or not hasplayed) or n~="gameplay") then print2("<  >",65,y) end'


def test_short_if_with_two_statement_body():
    # dodge.lua:2131 — the body is everything to end of line
    assert preprocess("if (r.sh>1.99) r.sh=2 initspawns()") == \
        "if (r.sh>1.99) then r.sh=2 initspawns() end"


def test_short_if_end_goes_before_trailing_comment():
    assert preprocess("if (a) b() -- why") == "if (a) then b() end -- why"


def test_short_if_preserves_indentation_of_body():
    assert preprocess("  if (a) b()") == "  if (a) then b() end"


def test_explicit_then_is_not_transformed():
    assert preprocess("if (a) then b() end") == "if (a) then b() end"


def test_multiline_condition_is_not_transformed():
    # dodge.lua:248 — parens stay open at end of line, so this is an ordinary if
    src = "if (stat(34)==1 and\n    70>stat(32)) or\n    (btnp(0) and cursi==i2)\n then\n  x()\n end"
    assert preprocess(src) == src


def test_bare_if_without_parens_is_not_transformed():
    assert preprocess("if a then b() end") == "if a then b() end"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocess.py -k short_if -v`
Expected: FAIL — no `then` inserted

- [ ] **Step 3: Implement**

Add to `pico8/preprocess.py`:

```python
_SHORT_IF_RE = re.compile(r"(?m)^[ \t]*if[ \t]*\(")


def _match_paren(mask, open_pos, limit):
    """Index of the paren matching mask[open_pos], or None if not before limit."""
    depth = 0
    for i in range(open_pos, limit):
        if mask[i] == "(":
            depth += 1
        elif mask[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _short_if_edits(src, mask):
    edits = []
    for m in _SHORT_IF_RE.finditer(mask):
        line_end = mask.find("\n", m.end())
        line_end = len(mask) if line_end == -1 else line_end

        close = _match_paren(mask, m.end() - 1, line_end)
        if close is None:
            continue                      # condition spans lines: ordinary if

        body = mask[close + 1:line_end]
        if not body.strip() or body.strip().startswith("then"):
            continue

        body_end = _statement_end(mask, close + 1)
        edits.append((close + 1, close + 1, " then"))
        edits.append((body_end, body_end, " end"))
    return edits
```

Add the pass to `preprocess` **last**. The order matters and is not arbitrary: three lines
in the cart (285, 286, 919) are a short `if` whose body is a compound assignment, such as
`if (btnp(⬇️)) cursi+=1`. Compound assignment takes its right-hand side to end of line, so
if the short-`if` ran first the `end` it appends would be swallowed into the expression,
yielding `cursi = cursi + (1 end)`.

```python
def preprocess(src):
    """Translate a PICO-8 Lua source string into standard Lua 5.1.

    Pass order is load-bearing: compound assignment must run before the
    short-if rule, or the appended `end` lands inside the expression.
    """
    for pass_fn in (_token_edits, _compound_edits, _short_if_edits):
        src = apply_edits(src, pass_fn(src, mask_literals(src)))
    return src
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: 33 passed

- [ ] **Step 5: Commit**

```bash
git add pico8/preprocess.py tests/test_preprocess.py
git commit -m "feat: one-line if transform"
```

---

### Task 6: Whole-cart syntax check

The unit tests cover the rules in isolation. This proves the combination parses — the single highest-value test in the phase, because a syntax error anywhere in 2,263 lines blocks everything downstream.

**Files:**
- Create: `tests/test_preprocess_cart.py`, `tools/check_lua.py`

**Interfaces:**
- Consumes: `pico8.cart.read_rom`, `pico8.cart.decompress_code`, `pico8.preprocess.preprocess`
- Produces: nothing

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocess_cart.py
from pathlib import Path
import pytest
from pico8 import cart
from pico8.preprocess import preprocess

lupa = pytest.importorskip("lupa")
SOURCE = Path("dodge.lua")
pytestmark = pytest.mark.skipif(not SOURCE.exists(), reason="dodge.lua not extracted")


@pytest.fixture(scope="module")
def translated():
    return preprocess(SOURCE.read_text(encoding="utf-8"))


def test_whole_cart_is_valid_lua(translated):
    """Compile (do not run) the translated cart under LuaJIT."""
    runtime = lupa.LuaRuntime()
    loader = runtime.eval("function(src) return loadstring(src) end")
    chunk = loader(translated)
    assert chunk is not None, "translated cart failed to compile"


def test_no_pico8_dialect_remains(translated):
    from pico8.preprocess import mask_literals, BUTTONS
    mask = mask_literals(translated)
    assert "!=" not in mask
    assert "0b" not in mask
    for glyph in BUTTONS:
        assert glyph not in mask, f"bare glyph {glyph!r} survived"
    for op in ("+=", "-=", "*=", "/=", "%="):
        assert op not in mask, f"compound assignment {op} survived"


def test_hud_glyphs_survive_inside_strings(translated):
    # These are button prompts drawn to screen, not constants
    assert '"❎"' in translated
    assert "to play again" in translated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocess_cart.py -v`
Expected: FAIL — `loadstring` returns `None`, or a dialect assertion trips

- [ ] **Step 3: Fix whatever the cart exposes**

The unit tests do not cover every line of a real 2,263-line cart. If compilation fails, get the actual message and fix the responsible rule in `pico8/preprocess.py` — then add the offending line to `tests/test_preprocess.py` as a new case before fixing it.

To see the compiler's message and the line it points at, create `tools/check_lua.py`:

```python
#!/usr/bin/env python3
"""Report the first syntax error in the translated cart, with context."""

import pathlib
import sys

import lupa

from pico8.preprocess import preprocess

translated = preprocess(pathlib.Path("dodge.lua").read_text(encoding="utf-8"))
runtime = lupa.LuaRuntime()
error = runtime.eval("function(s) local f, e = loadstring(s) return e end")(translated)

if error is None:
    print("ok: translated cart compiles")
    sys.exit(0)

print(error)
lines = translated.split("\n")
number = int(str(error).split(":")[1])
for i in range(max(0, number - 3), min(len(lines), number + 2)):
    print(f"{i + 1:5}{'>' if i + 1 == number else ' '} {lines[i]}")
sys.exit(1)
```

Run: `python tools/check_lua.py`
Expected: `ok: translated cart compiles`. Until then it prints the message and the five lines around it. Note the reported line number is in the *translated* source; the transforms are line-preserving, so it maps directly onto `dodge.lua`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_preprocess_cart.py tools/check_lua.py pico8/preprocess.py
git commit -m "test: whole-cart translation compiles under LuaJIT"
```

---

### Task 7: Lua prelude — tables, math, seeded PRNG

Implemented in Lua rather than Python because these are called thousands of times per frame; crossing the language boundary for `add`/`all` would dominate the frame budget. The PRNG is Lehmer/Park-Miller: not PICO-8's exact generator, but deterministic and seedable, which is what phase 2 needs.

`all()` must tolerate deletion during iteration — the cart does `del(part,_p)` inside `for _p in all(part)`.

**Files:**
- Create: `pico8/prelude.lua`, `tests/test_prelude.py`

**Interfaces:**
- Consumes: nothing
- Produces: a Lua chunk defining `add del all foreach count mid flr ceil abs min max sqrt sgn sin cos rnd srand tostr sub`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prelude.py
from pathlib import Path
import pytest

lupa = pytest.importorskip("lupa")
PRELUDE = Path("pico8/prelude.lua")


@pytest.fixture
def lua():
    runtime = lupa.LuaRuntime(unpack_returned_tuples=True)
    runtime.execute(PRELUDE.read_text(encoding="utf-8"))
    return runtime


def test_all_survives_deletion_during_iteration(lua):
    result = lua.execute("""
        local t = {1,2,3,4,5}
        local seen = {}
        for v in all(t) do
            seen[#seen+1] = v
            if v % 2 == 0 then del(t, v) end
        end
        return table.concat(seen, ",")
    """)
    assert result == "1,2,3,4,5"


def test_all_on_empty_and_nil(lua):
    assert lua.execute("local n=0 for v in all({}) do n=n+1 end return n") == 0
    assert lua.execute("local n=0 for v in all(nil) do n=n+1 end return n") == 0


def test_add_returns_value_and_appends(lua):
    assert lua.execute("local t={} add(t,7) add(t,8) return t[1]..','..t[2]") == "7,8"


def test_del_removes_first_match_only(lua):
    assert lua.execute("local t={1,2,1} del(t,1) return #t..':'..t[1]..','..t[2]") == "2:2,1"


def test_mid_clamps_like_pico8(lua):
    assert lua.execute("return mid(2,200,125)") == 125
    assert lua.execute("return mid(2,-5,125)") == 2
    assert lua.execute("return mid(2,50,125)") == 50


def test_sgn_of_zero_is_one(lua):
    assert lua.execute("return sgn(0)") == 1
    assert lua.execute("return sgn(-3)") == -1


def test_sin_is_inverted_and_turn_based(lua):
    assert lua.execute("return sin(0)") == pytest.approx(0, abs=1e-9)
    assert lua.execute("return sin(0.25)") == pytest.approx(-1, abs=1e-9)
    assert lua.execute("return cos(0)") == pytest.approx(1, abs=1e-9)
    assert lua.execute("return cos(0.25)") == pytest.approx(0, abs=1e-9)


def test_rnd_is_deterministic_for_a_seed(lua):
    first = lua.execute("srand(7) local s='' for i=1,5 do s=s..flr(rnd(100))..',' end return s")
    again = lua.execute("srand(7) local s='' for i=1,5 do s=s..flr(rnd(100))..',' end return s")
    assert first == again


def test_rnd_differs_between_seeds(lua):
    a = lua.execute("srand(1) return rnd(1000)")
    b = lua.execute("srand(2) return rnd(1000)")
    assert a != b


def test_rnd_range_and_table_form(lua):
    assert lua.execute("srand(1) local m=0 for i=1,200 do m=math.max(m,rnd(4)) end return m") < 4
    assert lua.execute("srand(1) return rnd({9,9,9})") == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prelude.py -v`
Expected: FAIL with `FileNotFoundError: pico8/prelude.lua`

- [ ] **Step 3: Implement**

```lua
-- pico8/prelude.lua
-- PICO-8 builtins that are pure Lua. Implemented here rather than in Python
-- because they run thousands of times per frame.

function add(t, v)
  if t ~= nil then t[#t + 1] = v end
  return v
end

function del(t, v)
  if t == nil then return end
  for i = 1, #t do
    if t[i] == v then
      table.remove(t, i)
      return v
    end
  end
end

-- Tolerates deletion of the current element during iteration, which the cart
-- relies on (`del(part,_p)` inside `for _p in all(part)`).
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

function foreach(t, f)
  for v in all(t) do f(v) end
end

function count(t)
  if t == nil then return 0 end
  return #t
end

flr = math.floor
ceil = math.ceil
abs = math.abs
min = math.min
max = math.max
sqrt = math.sqrt
sub = string.sub

function mid(a, b, c)
  return math.max(math.min(a, b), math.min(math.max(a, b), c))
end

function sgn(x)
  if x < 0 then return -1 end
  return 1
end

-- PICO-8 angles are turns, and its sine is inverted relative to math.sin.
local TAU = 6.283185307179586
function sin(x) return -math.sin(x * TAU) end
function cos(x) return math.cos(x * TAU) end

function tostr(v)
  if v == nil then return "[nil]" end
  return tostring(v)
end

-- Lehmer / Park-Miller. Not PICO-8's generator, but deterministic and seedable.
local seed = 1
function srand(s)
  seed = math.floor(s) % 2147483647
  if seed <= 0 then seed = seed + 2147483646 end
end

function rnd(x)
  seed = (seed * 16807) % 2147483647
  local r = (seed - 1) / 2147483646
  if type(x) == "table" then
    if #x == 0 then return nil end
    return x[math.floor(r * #x) + 1]
  end
  if x == nil then x = 1 end
  return r * x
end
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prelude.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add pico8/prelude.lua tests/test_prelude.py
git commit -m "feat: Lua prelude with tables, math and seeded PRNG"
```

---

### Task 8: Backend protocol, NullBackend, and the machine

The machine owns input state, memory and the Lua runtime, and steps exactly one frame. The headless smoke test at the end is the moment the real game first runs.

**Files:**
- Create: `pico8/backends.py`, `pico8/api.py`, `pico8/machine.py`, `tests/test_machine.py`

**Interfaces:**
- Consumes: `pico8.cart.read_rom`, `pico8.cart.decompress_code`, `pico8.cart.sprite_sheet`, `pico8.preprocess.preprocess`, `pico8/prelude.lua`
- Produces:
  - `pico8.backends.Backend` — methods `cls(col)`, `pset(x,y,c)`, `rect(x0,y0,x1,y1,c)`, `rectfill(x0,y0,x1,y1,c)`, `circfill(x,y,r,c)`, `ovalfill(x0,y0,x1,y1,c)`, `line(x0,y0,x1,y1,c)`, `spr(n,x,y,w,h)`, `print_(text,x,y,c)`, `pal(c0,c1,mode)`, `fillp(pattern)`, `camera(x,y)`, `flip()`
  - `pico8.backends.NullBackend()`
  - `pico8.machine.Machine(rom, code, backend, seed=0)` with `.step()`, `.set_buttons(mask: int)`, `.set_mouse(x, y, buttons)`, `.globals`, `.frame`
  - `pico8.machine.boot(js_path=Path("logic.js"), backend=None, seed=0) -> Machine`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_machine.py
from pathlib import Path
import pytest

pytest.importorskip("lupa")
from pico8 import machine
from pico8.backends import NullBackend

pytestmark = pytest.mark.skipif(not Path("logic.js").exists(), reason="logic.js not present")


@pytest.fixture(scope="module")
def booted():
    return machine.boot(backend=NullBackend(), seed=1)


def test_cart_boots_and_defines_its_entry_points(booted):
    g = booted.globals
    assert g._draw is not None
    assert g.updategame is not None
    assert g.drawgame is not None


def test_stepping_runs_frames_without_error(booted):
    start = booted.frame
    for _ in range(120):
        booted.step()
    assert booted.frame == start + 120


def test_gameplay_state_advances_and_kills_the_idle_player():
    m = machine.boot(backend=NullBackend(), seed=3)
    m.start_game()
    assert m.globals.isdead is False
    for _ in range(3600):          # a minute of standing still
        m.step()
        if m.globals.isdead:
            break
    assert m.globals.isdead is True, "an idle player should eventually be hit"


def test_enemies_actually_spawn():
    m = machine.boot(backend=NullBackend(), seed=5)
    m.start_game()
    for _ in range(600):
        m.step()
        if len(m.globals.enemies) > 0:
            break
    assert len(m.globals.enemies) > 0


def test_same_seed_gives_the_same_trajectory():
    def run(seed):
        m = machine.boot(backend=NullBackend(), seed=seed)
        m.start_game()
        trace = []
        for _ in range(300):
            m.step()
            trace.append((round(m.globals.px, 6), round(m.globals.py, 6),
                          len(m.globals.enemies)))
        return trace

    assert run(11) == run(11)
    assert run(11) != run(12)


def test_buttons_move_the_player():
    m = machine.boot(backend=NullBackend(), seed=7)
    m.start_game()
    m.set_buttons(1 << 1)          # hold right
    x0 = m.globals.px
    for _ in range(20):
        m.step()
    assert m.globals.px > x0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_machine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pico8.backends'`

- [ ] **Step 3: Implement the backends**

```python
# pico8/backends.py
"""Drawing targets for the PICO-8 machine.

The machine always calls every drawing entry point. NullBackend discards the
output rather than skipping the call, because the cart's draw path consumes
randomness from the same stream as gameplay -- see spawntrail() inside
drawgame(). Skipping draws would change which enemies spawn.
"""


class Backend:
    """Interface every backend implements. Coordinates are PICO-8 pixels."""

    def cls(self, col): raise NotImplementedError
    def pset(self, x, y, c): raise NotImplementedError
    def rect(self, x0, y0, x1, y1, c): raise NotImplementedError
    def rectfill(self, x0, y0, x1, y1, c): raise NotImplementedError
    def circfill(self, x, y, r, c): raise NotImplementedError
    def ovalfill(self, x0, y0, x1, y1, c): raise NotImplementedError
    def line(self, x0, y0, x1, y1, c): raise NotImplementedError
    def spr(self, n, x, y, w, h): raise NotImplementedError
    def print_(self, text, x, y, c): raise NotImplementedError
    def pal(self, c0, c1, mode): raise NotImplementedError
    def fillp(self, pattern): raise NotImplementedError
    def camera(self, x, y): raise NotImplementedError
    def flip(self): raise NotImplementedError


class NullBackend(Backend):
    """Accepts and discards every draw call."""

    def cls(self, col): pass
    def pset(self, x, y, c): pass
    def rect(self, x0, y0, x1, y1, c): pass
    def rectfill(self, x0, y0, x1, y1, c): pass
    def circfill(self, x, y, r, c): pass
    def ovalfill(self, x0, y0, x1, y1, c): pass
    def line(self, x0, y0, x1, y1, c): pass
    def spr(self, n, x, y, w, h): pass
    def print_(self, text, x, y, c): pass
    def pal(self, c0, c1, mode): pass
    def fillp(self, pattern): pass
    def camera(self, x, y): pass
    def flip(self): pass
```

- [ ] **Step 4: Implement the Python-backed API**

```python
# pico8/api.py
"""PICO-8 builtins that need Python state: input, memory and drawing.

Everything here is installed as a Lua global. Pure-Lua builtins live in
prelude.lua instead.
"""

# PICO-8 button repeat: first repeat after 15 frames, then every 4.
BTNP_DELAY = 15
BTNP_INTERVAL = 4


def _num(value, default=0):
    """Lua passes nil for omitted arguments."""
    return default if value is None else value


def install(machine, lua):
    g = lua.globals()

    def btn(i=None):
        if i is None:
            return machine.buttons
        return bool(machine.buttons & (1 << int(i)))

    def btnp(i=None):
        if i is None:
            return machine.btnp_mask
        return bool(machine.btnp_mask & (1 << int(i)))

    def stat(n):
        n = int(n)
        if n == 32:
            return machine.mouse_x
        if n == 33:
            return machine.mouse_y
        if n == 34:
            return machine.mouse_buttons
        return 0

    def peek(addr):
        return machine.mem[int(addr) & 0xFFFF]

    def poke(addr, value):
        machine.mem[int(addr) & 0xFFFF] = int(value) & 0xFF

    def time():
        return machine.frame / 60.0

    def noop(*args):
        return None

    b = machine.backend
    g.btn = btn
    g.btnp = btnp
    g.stat = stat
    g.peek = peek
    g.poke = poke
    g.time = time
    g.t = time
    g.sfx = noop
    g.music = noop
    g.printh = noop

    g.cls = lambda col=None: b.cls(int(_num(col)))
    g.pset = lambda x, y, c=None: b.pset(x, y, c)
    g.rect = lambda x0, y0, x1, y1, c=None: b.rect(x0, y0, x1, y1, c)
    g.rectfill = lambda x0, y0, x1, y1, c=None: b.rectfill(x0, y0, x1, y1, c)
    g.circfill = lambda x, y, r=None, c=None: b.circfill(x, y, _num(r, 4), c)
    g.ovalfill = lambda x0, y0, x1, y1, c=None: b.ovalfill(x0, y0, x1, y1, c)
    g.line = lambda x0, y0, x1, y1, c=None: b.line(x0, y0, x1, y1, c)
    g.spr = lambda n, x, y, w=None, h=None, *rest: b.spr(int(n), x, y, _num(w, 1), _num(h, 1))
    g.print = lambda text, x=None, y=None, c=None: b.print_(text, _num(x), _num(y), c)
    g.pal = lambda c0=None, c1=None, mode=None: b.pal(c0, c1, int(_num(mode)))
    g.palt = noop
    g.fillp = lambda pattern=None: b.fillp(pattern)
    g.camera = lambda x=None, y=None: b.camera(_num(x), _num(y))
    g.flip = lambda: machine.flip()
```

- [ ] **Step 5: Implement the machine**

```python
# pico8/machine.py
"""Boots the cart and steps it one frame at a time."""

from pathlib import Path

from lupa import LuaRuntime

from . import api, cart
from .backends import NullBackend
from .preprocess import preprocess

PRELUDE = Path(__file__).with_name("prelude.lua")

# smallsettings indices used by the cart: [theme, difficulty, patterns, powerups]
DIFFICULTY = {"normal": 1, "hard": 2, "easy": 3}


class Machine:
    def __init__(self, rom, code, backend, seed=0):
        self.rom = rom
        self.backend = backend
        self.sheet = cart.sprite_sheet(rom)
        self.mem = bytearray(0x10000)
        self.frame = 0

        self.buttons = 0
        self.btnp_mask = 0
        self._held = [0] * 6
        self.mouse_x, self.mouse_y, self.mouse_buttons = 64, 64, 0

        if hasattr(backend, "load_sheet"):
            backend.load_sheet(self.sheet)

        self.lua = LuaRuntime(unpack_returned_tuples=True)
        api.install(self, self.lua)
        self.lua.execute(PRELUDE.read_text(encoding="utf-8"))
        self.lua.execute(f"srand({seed})")
        self.lua.execute(preprocess(code))
        self.globals = self.lua.globals()

    # -- input ------------------------------------------------------------
    def set_buttons(self, mask):
        self.buttons = mask & 0x3F

    def set_mouse(self, x, y, buttons):
        self.mouse_x, self.mouse_y, self.mouse_buttons = x, y, buttons

    def _update_btnp(self):
        """PICO-8 btnp: true on the press frame, then repeating while held."""
        mask = 0
        for i in range(6):
            if self.buttons & (1 << i):
                self._held[i] += 1
                held = self._held[i]
                if held == 1 or (held > api.BTNP_DELAY and
                                 (held - api.BTNP_DELAY) % api.BTNP_INTERVAL == 0):
                    mask |= 1 << i
            else:
                self._held[i] = 0
        self.btnp_mask = mask

    # -- stepping ---------------------------------------------------------
    def flip(self):
        """Called by the cart's own flip(); also ends a frame during boot."""
        self.backend.flip()

    def step(self):
        self.frame += 1
        self._update_btnp()
        if self.globals._update60 is not None:
            self.globals._update60()
        self.globals._draw()
        self.backend.flip()

    def start_game(self):
        """Skip the menus and begin a round immediately."""
        g = self.globals
        g._upd, g._drw = g.updategame, g.drawgame
        g.reset()


def boot(js_path=Path("logic.js"), backend=None, seed=0):
    rom, _ = cart.read_rom(Path(js_path))
    code = cart.decompress_code(rom)
    return Machine(rom, code, backend or NullBackend(), seed=seed)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_machine.py -v`
Expected: 6 passed

If `test_gameplay_state_advances_and_kills_the_idle_player` fails, check that `smallsettings` is populated — `updatesettings()` runs once at cart load and fills it. If enemies never spawn, print `m.globals.spd` and `m.globals.est` after `start_game()`; both must be non-nil.

- [ ] **Step 7: Commit**

```bash
git add pico8/backends.py pico8/api.py pico8/machine.py tests/test_machine.py
git commit -m "feat: PICO-8 machine running the cart headless"
```

---

### Task 9: Pygame rendering

An 8-bit indexed surface is what makes `pal(c, c2, 1)` cheap: the cart draws with palette *indices*, and the display palette is applied by swapping the surface's palette before scaling. Without that the game renders flat blue, because every draw call in `drawgame` uses colour 12 or 1.

**Files:**
- Create: `pico8/palette.py`, `tests/test_pygame_backend.py`
- Modify: `pico8/backends.py`

**Interfaces:**
- Consumes: `pico8.backends.Backend`
- Produces:
  - `pico8.palette.PICO8: list[tuple[int, int, int]]` — 16 RGB triples
  - `pico8.backends.PygameBackend(scale=5)` with `.load_sheet(rows)`, `.surface`, `.window`, and the `Backend` methods

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pygame_backend.py
import os
import pytest

pygame = pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pico8.backends import PygameBackend
from pico8.palette import PICO8


@pytest.fixture
def backend():
    b = PygameBackend(scale=1)
    yield b
    pygame.quit()


def test_palette_has_sixteen_colours():
    assert len(PICO8) == 16
    assert PICO8[0] == (0, 0, 0)
    assert PICO8[7] == (255, 241, 232)


def test_cls_fills_with_the_index(backend):
    backend.cls(12)
    assert backend.surface.get_at((0, 0))[:3] == PICO8[12]
    assert backend.surface.get_at((127, 127))[:3] == PICO8[12]


def test_rectfill_uses_the_requested_index(backend):
    backend.cls(0)
    backend.rectfill(2, 2, 5, 5, 8)
    assert backend.surface.get_at((3, 3))[:3] == PICO8[8]
    assert backend.surface.get_at((10, 10))[:3] == PICO8[0]


def test_camera_offsets_subsequent_draws(backend):
    backend.cls(0)
    backend.camera(10, 0)
    backend.rectfill(10, 0, 12, 2, 8)
    assert backend.surface.get_at((0, 0))[:3] == PICO8[8]


def test_draw_palette_remaps_colours(backend):
    backend.cls(0)
    backend.pal(8, 12, 0)
    backend.rectfill(0, 0, 4, 4, 8)
    assert backend.surface.get_at((1, 1))[:3] == PICO8[12]


def test_display_palette_does_not_change_the_drawn_index(backend):
    backend.cls(0)
    backend.rectfill(0, 0, 4, 4, 12)
    backend.pal(12, 3, 1)
    # the surface still holds index 12; the swap happens at flip time
    assert backend.surface.get_at((1, 1))[:3] == PICO8[3]


def test_spr_blits_from_the_sheet(backend):
    sheet = [[0] * 128 for _ in range(128)]
    sheet[0][0] = 9              # sprite 0, top-left pixel
    backend.load_sheet(sheet)
    backend.cls(0)
    backend.spr(0, 20, 30, 1, 1)
    assert backend.surface.get_at((20, 30))[:3] == PICO8[9]


def test_colour_zero_is_transparent_in_sprites(backend):
    sheet = [[0] * 128 for _ in range(128)]
    backend.load_sheet(sheet)
    backend.cls(7)
    backend.spr(0, 0, 0, 1, 1)
    assert backend.surface.get_at((2, 2))[:3] == PICO8[7]


def test_backends_produce_identical_trajectories():
    """The two backends must simulate identically.

    This is not a formality: drawgame() calls spawntrail(), which draws from
    the same RNG stream as gameplay. If rendering consumed randomness that
    headless did not, the trained policy and the watched policy would be
    playing different games.
    """
    import pathlib
    if not pathlib.Path("logic.js").exists():
        pytest.skip("logic.js not present")

    from pico8.machine import boot
    from pico8.backends import NullBackend

    def run(backend):
        m = boot(backend=backend, seed=21)
        m.start_game()
        trace = []
        for _ in range(900):        # long enough for powerups to appear
            m.step()
            trace.append((round(m.globals.px, 6), round(m.globals.py, 6),
                          round(m.globals.score, 6), len(m.globals.enemies)))
        return trace

    headless = run(NullBackend())
    rendered = run(PygameBackend(scale=1))
    pygame.quit()
    assert headless == rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pygame_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pico8.palette'`

- [ ] **Step 3: Implement the palette**

```python
# pico8/palette.py
"""The 16 PICO-8 colours, in index order."""

PICO8 = [
    (0, 0, 0),         # 0  black
    (29, 43, 83),      # 1  dark blue
    (126, 37, 83),     # 2  dark purple
    (0, 135, 81),      # 3  dark green
    (171, 82, 54),     # 4  brown
    (95, 87, 79),      # 5  dark grey
    (194, 195, 199),   # 6  light grey
    (255, 241, 232),   # 7  white
    (255, 0, 77),      # 8  red
    (255, 163, 0),     # 9  orange
    (255, 236, 39),    # 10 yellow
    (0, 228, 54),      # 11 green
    (41, 173, 255),    # 12 blue
    (131, 118, 156),   # 13 lavender
    (255, 119, 168),   # 14 pink
    (255, 204, 170),   # 15 peach
]
```

- [ ] **Step 4: Implement the backend**

Append to `pico8/backends.py`:

```python
import pygame

from .palette import PICO8


class PygameBackend(Backend):
    """Draws to a 128x128 indexed surface, scaled to the window on flip.

    Colour values passed in are PICO-8 palette indices, and the surface stores
    them directly. That makes pal(c, c2, 1) a palette swap rather than a
    per-pixel rewrite -- which matters because the cart draws the entire game
    in colours 12 and 1 and remaps them at the end of every frame.
    """

    SIZE = 128

    def __init__(self, scale=5, caption="dodge"):
        pygame.init()
        self.scale = scale
        self.window = pygame.display.set_mode((self.SIZE * scale, self.SIZE * scale))
        pygame.display.set_caption(caption)

        self.surface = pygame.Surface((self.SIZE, self.SIZE), depth=8)
        self.surface.set_palette(PICO8)
        self.sheet = None

        self.draw_pal = list(range(16))     # pal(c, c1, 0)
        self.display_pal = list(range(16))  # pal(c, c1, 1)
        self.cam_x = self.cam_y = 0
        self.pen = 6
        self.text_queue = []

    # -- helpers ----------------------------------------------------------
    def load_sheet(self, rows):
        """Build a transparent surface from 128x128 palette indices."""
        sheet = pygame.Surface((self.SIZE, self.SIZE), depth=8)
        sheet.set_palette(PICO8)
        for y, row in enumerate(rows):
            for x, index in enumerate(row):
                sheet.set_at((x, y), index)
        sheet.set_colorkey(0)               # PICO-8 treats colour 0 as transparent
        self.sheet = sheet

    def _colour(self, c):
        if c is None:
            c = self.pen
        else:
            self.pen = int(c) & 15
        return self.draw_pal[int(c) & 15]

    def _pt(self, x, y):
        return int(x) - self.cam_x, int(y) - self.cam_y

    # -- Backend ----------------------------------------------------------
    def cls(self, col=0):
        self.surface.fill(self.draw_pal[int(col) & 15])
        self.cam_x = self.cam_y = 0

    def pset(self, x, y, c=None):
        px, py = self._pt(x, y)
        if 0 <= px < self.SIZE and 0 <= py < self.SIZE:
            self.surface.set_at((px, py), self._colour(c))

    def rect(self, x0, y0, x1, y1, c=None):
        colour = self._colour(c)
        ax, ay = self._pt(x0, y0)
        bx, by = self._pt(x1, y1)
        pygame.draw.rect(self.surface, colour,
                         pygame.Rect(min(ax, bx), min(ay, by),
                                     abs(bx - ax) + 1, abs(by - ay) + 1), width=1)

    def rectfill(self, x0, y0, x1, y1, c=None):
        colour = self._colour(c)
        ax, ay = self._pt(x0, y0)
        bx, by = self._pt(x1, y1)
        pygame.draw.rect(self.surface, colour,
                         pygame.Rect(min(ax, bx), min(ay, by),
                                     abs(bx - ax) + 1, abs(by - ay) + 1))

    def circfill(self, x, y, r=4, c=None):
        colour = self._colour(c)
        px, py = self._pt(x, y)
        radius = int(r)
        if radius <= 0:
            if 0 <= px < self.SIZE and 0 <= py < self.SIZE:
                self.surface.set_at((px, py), colour)
            return
        pygame.draw.circle(self.surface, colour, (px, py), radius)

    def ovalfill(self, x0, y0, x1, y1, c=None):
        colour = self._colour(c)
        ax, ay = self._pt(x0, y0)
        bx, by = self._pt(x1, y1)
        rect = pygame.Rect(min(ax, bx), min(ay, by), abs(bx - ax) + 1, abs(by - ay) + 1)
        pygame.draw.ellipse(self.surface, colour, rect)

    def line(self, x0, y0, x1, y1, c=None):
        colour = self._colour(c)
        pygame.draw.line(self.surface, colour, self._pt(x0, y0), self._pt(x1, y1))

    def spr(self, n, x, y, w=1, h=1):
        if self.sheet is None:
            return
        n = int(n)
        sx, sy = (n % 16) * 8, (n // 16) * 8
        area = pygame.Rect(sx, sy, int(w) * 8, int(h) * 8)
        self.surface.blit(self.sheet, self._pt(x, y), area)

    def print_(self, text, x=0, y=0, c=None):
        self.text_queue.append((str(text), int(x) - self.cam_x,
                                int(y) - self.cam_y, self._colour(c)))

    def pal(self, c0=None, c1=None, mode=0):
        if c0 is None:
            self.draw_pal = list(range(16))
            self.display_pal = list(range(16))
            self._apply_display_palette()
            return
        if int(mode) == 1:
            self.display_pal[int(c0) & 15] = int(c1) & 15
            self._apply_display_palette()
        else:
            self.draw_pal[int(c0) & 15] = int(c1) & 15

    def _apply_display_palette(self):
        self.surface.set_palette([PICO8[self.display_pal[i]] for i in range(16)])

    def fillp(self, pattern=None):
        pass                                 # approximated: solid fills

    def camera(self, x=0, y=0):
        self.cam_x, self.cam_y = int(x), int(y)

    def flip(self):
        self._apply_display_palette()
        pygame.transform.scale(self.surface, self.window.get_size(), self.window)
        self._draw_text()
        pygame.display.flip()
        self.text_queue.clear()

    def _draw_text(self):
        pass                                 # implemented in Task 10
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_pygame_backend.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add pico8/palette.py pico8/backends.py tests/test_pygame_backend.py
git commit -m "feat: pygame backend with indexed surface and palette swaps"
```

---

### Task 10: Text overlay and button glyphs

PICO-8 glyphs are 3×5 px, which no system font renders legibly at native scale. Text is therefore queued during the frame and drawn onto the *upscaled* window, where there is room for a real font. Button glyphs inside strings (`"press ❎ to play again"`) become small drawn icons.

**Files:**
- Modify: `pico8/backends.py`
- Create: `tests/test_text_overlay.py`

**Interfaces:**
- Consumes: `PygameBackend.text_queue`, `pico8.preprocess.BUTTONS`
- Produces: `pico8.backends.split_glyphs(text: str) -> list[tuple[str, str]]` — `("text"|"glyph", chunk)` pairs

- [ ] **Step 1: Write the failing test**

```python
# tests/test_text_overlay.py
from pico8.backends import split_glyphs


def test_plain_text_is_one_chunk():
    assert split_glyphs("score:12") == [("text", "score:12")]


def test_glyph_is_split_out():
    assert split_glyphs("press ❎ to play") == [
        ("text", "press "), ("glyph", "❎"), ("text", " to play"),
    ]


def test_adjacent_glyphs():
    assert split_glyphs("⬅️➡️") == [
        ("glyph", "⬅️"), ("glyph", "➡️"),
    ]


def test_empty_string():
    assert split_glyphs("") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_text_overlay.py -v`
Expected: FAIL with `ImportError: cannot import name 'split_glyphs'`

- [ ] **Step 3: Implement**

Add to `pico8/backends.py`:

```python
import re

from .preprocess import BUTTONS

_GLYPH_SPLIT = re.compile("(" + "|".join(re.escape(g) for g in BUTTONS) + ")")

# What each button glyph is drawn as, since no system font has them.
GLYPH_LABEL = {
    "⬅️": "<", "➡️": ">", "⬆️": "^",
    "⬇️": "v", "\U0001f17e️": "O", "❎": "X",
}


def split_glyphs(text):
    """Split a PICO-8 string into ('text'|'glyph', chunk) pairs."""
    return [("glyph" if chunk in BUTTONS else "text", chunk)
            for chunk in _GLYPH_SPLIT.split(text) if chunk]
```

Replace the placeholder `_draw_text` on `PygameBackend` with the real one, and initialise the font in `__init__` after `pygame.init()`:

```python
        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace",
                                        max(8, 4 * scale))
```

```python
    def _draw_text(self):
        """Draw queued text onto the upscaled window.

        PICO-8 characters occupy a 4x6 cell, so a chunk of n characters is
        n*4 pixels wide in game space; that is the advance used here.
        """
        s = self.scale
        for text, x, y, colour in self.text_queue:
            rgb = PICO8[self.display_pal[colour]]
            cursor = x
            for kind, chunk in split_glyphs(text):
                if kind == "glyph":
                    label = GLYPH_LABEL[chunk]
                    box = pygame.Rect(cursor * s, y * s, 7 * s, 5 * s)
                    pygame.draw.rect(self.window, rgb, box, width=max(1, s // 2))
                    glyph = self.font.render(label, True, rgb)
                    self.window.blit(glyph, glyph.get_rect(center=box.center))
                    cursor += 8
                else:
                    self.window.blit(self.font.render(chunk, True, rgb),
                                     (cursor * s, y * s))
                    cursor += len(chunk) * 4
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add pico8/backends.py tests/test_text_overlay.py
git commit -m "feat: scaled text overlay with drawn button glyphs"
```

---

### Task 11: The playable entry point

**Files:**
- Create: `play.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `pico8.machine.boot`, `pico8.backends.PygameBackend`
- Produces: `python play.py` — a playable window

- [ ] **Step 1: Write the entry point**

The window must exist *before* the cart is booted: the cart's top-level code runs a 66-frame splash animation calling `flip()`, so the backend has to be ready to present.

```python
#!/usr/bin/env python3
"""Play the extracted PICO-8 cart locally.

    python play.py                 # normal
    python play.py --scale 4       # smaller window
    python play.py --seed 3        # fixed RNG seed

Controls: arrows or WASD to move, Z/C for O, X/V for X, Escape to quit.
The mouse also works -- the cart switches control mode on its own, exactly
as the browser build does.
"""

import argparse
import sys

import pygame

from pico8.backends import PygameBackend
from pico8.machine import boot

# PICO-8 button index -> keys that press it
KEYMAP = {
    0: (pygame.K_LEFT, pygame.K_a),
    1: (pygame.K_RIGHT, pygame.K_d),
    2: (pygame.K_UP, pygame.K_w),
    3: (pygame.K_DOWN, pygame.K_s),
    4: (pygame.K_z, pygame.K_c),
    5: (pygame.K_x, pygame.K_v),
}


def button_mask(pressed):
    mask = 0
    for index, keys in KEYMAP.items():
        if any(pressed[k] for k in keys):
            mask |= 1 << index
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=5, help="window scale (default 5)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument("--js", default="logic.js", help="path to the HTML5 export")
    args = parser.parse_args()

    backend = PygameBackend(scale=args.scale)
    machine = boot(js_path=args.js, backend=backend, seed=args.seed)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        pressed = pygame.key.get_pressed()
        machine.set_buttons(button_mask(pressed))

        mx, my = pygame.mouse.get_pos()
        buttons = pygame.mouse.get_pressed()
        machine.set_mouse(mx // args.scale, my // args.scale,
                          (1 if buttons[0] else 0) | (2 if buttons[2] else 0))

        machine.step()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Handle events during the boot splash**

The splash calls `flip()` 66 times before `main()` reaches its loop, so the window would be unresponsive and Windows may mark it "not responding". Add an event pump to `PygameBackend.flip`, just before `pygame.display.flip()`:

```python
        pygame.event.pump()
```

- [ ] **Step 3: Play it**

Run: `python play.py`

Expected, in order:
1. A CoolMathGames splash animation (an expanding oval and circle) for about a second
2. The title screen with "press X" and the three credited names
3. `X` starts a round; the player is a white circle, enemies grow from the corners
4. Arrow keys move with visible momentum — the player drifts after keys are released
5. `O` from the title screen opens Settings; left/right change theme, and the colours change immediately

Troubleshooting:

- **Everything renders blue.** `pal(c, c1, 1)` is not reaching the surface palette. Check that `_apply_display_palette` runs in `flip`.
- **The player does not respond to keys.** The cart picks its control mode from `input = anybtn() and 1 or anymouse() and 0 or input`, so a moving mouse can hold it in mouse mode. Move the mouse off the window and press a direction key.
- **Sprites are missing but shapes draw.** `load_sheet` was not called — confirm `PygameBackend` is passed to `boot` before the machine is constructed.

- [ ] **Step 4: Update the README**

```markdown
# DodgeAI

A local, editable copy of the PICO-8 game `dodge`, extracted from its HTML5
export, plus (eventually) a training environment for it.

## Play it

    pip install -r requirements.txt
    python play.py

Arrows or WASD to move, Z/C and X/V for the O and X buttons, Escape to quit.

## Re-extract the cart

`dodge.lua` and `dodge.p8` are generated from `logic.js`:

    python tools/extract_cart.py

## Layout

| Path | What |
| --- | --- |
| `logic.js` | the original HTML5 export (PICO-8 runtime + 32KB cart ROM) |
| `dodge.lua` | the game's Lua source, extracted |
| `dodge.p8` | the full cartridge in PICO-8's text format |
| `pico8/` | a partial PICO-8 runtime in Python, hosting the real Lua under LuaJIT |
| `tools/` | the cart extractor |
| `docs/superpowers/` | design spec and implementation plan |
```

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -v`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add play.py pico8/backends.py README.md
git commit -m "feat: playable local build"
```

---

## Done when

- `python play.py` opens a window, plays the splash, shows the menu, and a round is playable with the keyboard
- The settings screen changes theme, difficulty, patterns and powerups, and the changes take effect
- `python -m pytest tests/ -v` is green
- `python tools/extract_cart.py` still reproduces `dodge.lua` and `dodge.p8` byte-for-byte
- `dodge.lua` has not been modified
