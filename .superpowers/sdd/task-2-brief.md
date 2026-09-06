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

