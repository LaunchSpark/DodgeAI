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
