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

