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

