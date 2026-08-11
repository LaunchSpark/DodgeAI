"""Translate PICO-8's Lua dialect into standard Lua 5.1.

Every rule matches against a *mask* of the source in which string and comment
contents are replaced by sentinel characters, then edits the original by index.
That makes "never rewrite inside a literal" a property of the whole module
rather than something each rule has to remember.
"""

import re

STR_MASK = "\x00"
CMT_MASK = "\x01"

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
_BIN_RE = re.compile(r"(?<!\w)0b([01]+)(?:\.([01]+))?")
_SHORTHAND_RE = re.compile(r"(?m)^([ \t]*)\?")


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


def _statement_end(mask, pos):
    """End of the statement: end of line, excluding any trailing comment."""
    end = mask.find("\n", pos)
    end = len(mask) if end == -1 else end
    while end > pos and (mask[end - 1] == CMT_MASK or mask[end - 1] in " \t"):
        end -= 1
    return end


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
        end = _statement_end(mask, m.end())
        edits.append((m.end() - 1, m.end(), "print("))
        edits.append((end, end, ")"))

    return edits


def preprocess(src):
    """Translate a PICO-8 Lua source string into standard Lua 5.1."""
    for pass_fn in (_token_edits, _compound_edits):
        src = apply_edits(src, pass_fn(src, mask_literals(src)))
    return src
