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

