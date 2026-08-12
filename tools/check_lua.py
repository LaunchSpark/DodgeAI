#!/usr/bin/env python3
"""Report the first syntax error in the translated cart, with context.

Compiles (does not run) the PICO-8-to-Lua translation of dodge.lua under
LuaJIT and, if it fails, prints the compiler's error message plus the five
lines of *translated* source around the reported line. The preprocess
transforms are line-preserving, so that line number maps directly onto
dodge.lua.

Usage:
    python tools/check_lua.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lupa.luajit21 import LuaRuntime

from pico8.preprocess import preprocess


def main():
    translated = preprocess(pathlib.Path("dodge.lua").read_text(encoding="utf-8"))
    runtime = LuaRuntime()
    error = runtime.eval("function(s) local f, e = loadstring(s) return e end")(translated)

    if error is None:
        print("ok: translated cart compiles")
        return 0

    print(error)
    lines = translated.split("\n")
    number = int(str(error).split(":")[1])
    for i in range(max(0, number - 3), min(len(lines), number + 2)):
        print(f"{i + 1:5}{'>' if i + 1 == number else ' '} {lines[i]!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
