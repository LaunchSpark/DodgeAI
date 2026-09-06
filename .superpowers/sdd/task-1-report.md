# Task 1 Report: Cart loading module

## What I implemented

- `pico8/__init__.py` — empty, marks `pico8` as a package.
- `pico8/cart.py` — new library module holding the ROM/PXA decoder, moved verbatim
  out of `tools/extract_cart.py`:
  - ROM offset constants (`GFX, MAP, GFF, MUSIC, SFX, CODE, END`)
  - `P8SCII` / `GLYPHS`
  - `read_rom(js_path)` (renamed from `read_cartdat`) — pulls the `_cartdat` array
    and cart name out of the JS export
  - `BitReader` — LSB-first bit reader used by the PXA decompressor
  - `decompress_code(rom)` — LZ77 + move-to-front decompressor for the code section
  - Both `sys.exit(...)` calls became `raise ValueError(...)` so the library
    never kills a host process (the only intentional behaviour change).
  - New `sprite_sheet(rom)` — decodes ROM `0x0000-0x1fff` into 128 rows of 128
    palette indices (two pixels per byte, low nibble first), added verbatim
    from the brief.
- `requirements.txt` — `lupa>=2.8`, `pygame>=2.5`, `pytest>=8.0` (verbatim from brief).
- `tests/__init__.py` — empty.
- `tests/test_cart.py` — the three tests from the brief, verbatim.
- `tools/extract_cart.py` — deleted the moved code, replaced with:
  ```python
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  from pico8.cart import GFX, MAP, GFF, MUSIC, SFX, CODE, END, GLYPHS, read_rom, decompress_code
  ```
  and changed the `read_cartdat(args.js)` call in `main()` to `read_rom(args.js)`.
  Everything else in the tool (`gfx_lines`, `map_lines`, `gff_lines`, `sfx_lines`,
  `music_lines`, `trim`, `build_p8`, `main`, CLI flags) is untouched.
- `.gitignore` — added (`__pycache__/`, `*.pyc`, `.pytest_cache/`). Not requested
  by the brief, but `git add pico8 tests ...` was about to sweep in compiled
  `.pyc` files from `pico8/__pycache__` and `tests/__pycache__`; ignoring them
  seemed like the obviously-correct minimal fix rather than something to flag
  and leave broken.

## What I tested and the results

- Focused test: `python -m pytest tests/test_cart.py -v` — 3 passed.
- Full suite: `python -m pytest -v` — same 3 tests (it's the whole suite), all pass.
- Tool still round-trips: `python tools/extract_cart.py --out <tmpdir>` printed
  `dodge.lua  37,110 chars` and `dodge.p8  64,909 chars`, matching the brief exactly.
- Regenerated in place (`python tools/extract_cart.py --out .`) and ran
  `git diff --stat -- dodge.lua dodge.p8` — no output, confirming byte-identical
  output before and after the refactor.

## TDD Evidence

**RED**

Command: `python -m pytest tests/test_cart.py -v`

```
ERROR collecting tests/test_cart.py
ImportError: cannot import name 'cart' from 'pico8' (unknown location)
```

Note: the brief predicted `ModuleNotFoundError: No module named 'pico8'`. I got
`ImportError: cannot import name 'cart' from 'pico8'` instead, because I had
already created the empty `pico8/` directory before writing the test (Python
treats it as an implicit namespace package). The underlying cause is identical
— `pico8.cart` did not exist yet — so this is still a correct RED result, just
with a slightly different exception subtype/message than predicted.

**GREEN**

Command: `python -m pytest tests/test_cart.py -v`

```
tests/test_cart.py::test_read_rom_returns_full_cart PASSED               [ 33%]
tests/test_cart.py::test_decompress_code_matches_known_size PASSED       [ 66%]
tests/test_cart.py::test_sprite_sheet_shape_and_content PASSED           [100%]

============================== 3 passed in 0.12s ==============================
```

## Files changed

- Created: `pico8/__init__.py`, `pico8/cart.py`, `requirements.txt`,
  `tests/__init__.py`, `tests/test_cart.py`, `.gitignore`
- Modified: `tools/extract_cart.py`
- Unchanged (verified byte-identical): `dodge.lua`, `dodge.p8`

Commit: `bad67f5` — "refactor: extract cart decoding into pico8.cart"

## Self-review findings

- Interfaces match the brief exactly: `read_rom(js_path) -> (bytes, str)`,
  `decompress_code(rom) -> str`, `sprite_sheet(rom) -> list[list[int]]`,
  `GLYPHS: dict[int, str]`, all importable from `pico8.cart`.
- Diffed `tools/extract_cart.py` against its pre-refactor version: the only
  changes are the deleted block replaced by the two-line import, and
  `read_cartdat` → `read_rom` in `main()`. No other logic touched.
- Confirmed no stray `sys.exit` calls remain in `pico8/cart.py`; `sys` is not
  imported there since nothing else needs it.
- Confirmed `pico8/__init__.py` and `tests/__init__.py` are empty, as specified.
- Test output is pristine — no warnings, no deprecation notices.
- Caught and fixed: `git add` was about to stage `pico8/__pycache__/*.pyc` and
  `tests/__pycache__/*.pyc` (no `.gitignore` existed before this task). Removed
  them from the index, deleted the directories, and added a `.gitignore` before
  committing.
- Confirmed `dodge.lua`/`dodge.p8` are byte-identical pre- and post-refactor via
  `git diff --stat`.

## Issues or concerns

None. The one deviation from the brief (RED step's exact exception message)
is cosmetic and explained above; the failure was for the correct underlying
reason. No scope was added beyond the brief except the `.gitignore`, which
was necessary to avoid committing build artefacts that the `git add` command
in Step 6 would otherwise have swept in.
