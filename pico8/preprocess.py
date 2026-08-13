"""Translate PICO-8's Lua dialect into standard Lua 5.1.

Every rule matches against a *mask* of the source in which string and comment
contents are replaced by sentinel characters, then edits the original by index.
That makes "never rewrite inside a literal" a property of the whole module
rather than something each rule has to remember.
"""

from functools import lru_cache
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


_THEN_RE = re.compile(r"\bthen\b")


def _has_top_level_then(mask, start, end):
    """True if a whole-word `then` appears at paren-depth 0 in mask[start:end].

    Depth is tracked by walking parens left-to-right so a `then` that sits
    inside a still-open group (there shouldn't be one, since callers only
    pass ranges with balanced parens) doesn't count, and a `then` after a
    closed group -- e.g. "(a) or (b) then" -- does.
    """
    depth = 0
    i = start
    for m in _THEN_RE.finditer(mask, start, end):
        depth += mask.count("(", i, m.start()) - mask.count(")", i, m.start())
        i = m.start()
        if depth == 0:
            return True
    return False


def _short_if_edits(src, mask):
    edits = []
    for m in _SHORT_IF_RE.finditer(mask):
        line_end = mask.find("\n", m.end())
        line_end = len(mask) if line_end == -1 else line_end

        close = _match_paren(mask, m.end() - 1, line_end)
        if close is None:
            continue                      # condition spans lines: ordinary if

        body = mask[close + 1:line_end]
        if not body.strip():
            continue

        # Distinguish short-if from long-form: a `then` at paren-depth 0
        # anywhere on this line (after `if`) means the condition is a
        # multi-group expression (e.g. "(a) or (b) then") and the `then`
        # we're seeing is the real long-form keyword, not part of a short
        # if's body.
        if _has_top_level_then(mask, m.end() - 1, line_end):
            continue

        body_end = _statement_end(mask, close + 1)
        edits.append((close + 1, close + 1, " then"))
        edits.append((body_end, body_end, " end"))
    return edits


@lru_cache(maxsize=4)
def preprocess(src):
    """Translate a PICO-8 Lua source string into standard Lua 5.1.

    Pass order is load-bearing: compound assignment must run before the
    short-if rule, or the appended `end` lands inside the expression.

    Cached because translating a cart is pure and not cheap -- three passes
    over 37,000 characters -- while a training run rebuilds its machine on
    every episode. Without the cache that translation dominated the step
    budget. The cache holds a handful of carts; keying on the source means a
    re-extracted or edited cart translates again on its own.
    """
    for pass_fn in (_token_edits, _compound_edits, _short_if_edits):
        src = apply_edits(src, pass_fn(src, mask_literals(src)))
    return src
