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

