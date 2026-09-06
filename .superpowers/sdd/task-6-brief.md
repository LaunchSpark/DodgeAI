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

