#!/usr/bin/env python3
"""Extract the PICO-8 cartridge embedded in the HTML5 export (logic.js).

logic.js is a PICO-8 cart exported to JS: ~97% of it is the Emscripten-compiled
PICO-8 runtime, and the actual game is a 32,768-byte ROM sitting in the
`_cartdat` array near the top of the file.

This pulls that ROM out and turns it back into editable sources:

    dodge.p8    full cartridge in PICO-8's plain-text format (loadable in PICO-8)
    dodge.lua   just the Lua, for reading/diffing
    dodge.rom   the raw 32KB ROM, with --rom

Usage:
    python tools/extract_cart.py                    # reads ./logic.js, writes ./
    python tools/extract_cart.py --js x.js --out b/ --rom
"""

import argparse
import json
import re
import sys
from pathlib import Path

# --- ROM layout -------------------------------------------------------------
# 0x0000  gfx        sprites 0-127   (sprite sheet rows 0-63)
# 0x1000  gfx2       sprites 128-255 (rows 64-127; shared with map rows 32-63)
# 0x2000  map        map rows 0-31
# 0x3000  gff        sprite flags
# 0x3100  music
# 0x3200  sfx
# 0x4300  code       PXA-compressed Lua
# 0x8000  end
GFX, MAP, GFF, MUSIC, SFX, CODE, END = 0x0000, 0x2000, 0x3000, 0x3100, 0x3200, 0x4300, 0x8000

# PICO-8 stores its glyphs as single bytes 0x80+ (P8SCII); .p8 files carry them
# as UTF-8. Kept as a list because several are multi-codepoint emoji, so a plain
# string would not index one glyph per byte.
P8SCII = [
    "█", "▒", "🐱", "⬇️", "░", "✽", "●", "♥",
    "☉", "웃", "⌂", "⬅️", "😐", "♪", "🅾️", "◆",
    "…", "➡️", "★", "⧗", "⬆️", "ˇ", "∧", "❎",
    "▤", "▥",
]
GLYPHS = {0x80 + i: ch for i, ch in enumerate(P8SCII)}


def read_cartdat(js_path):
    """Pull the `_cartdat = [...]` array out of the JS export."""
    text = js_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"_cartdat\s*=\s*(\[.*?\])\s*;", text, re.S)
    if not m:
        sys.exit(f"no _cartdat array found in {js_path}")
    rom = bytes(json.loads(m.group(1)))
    if len(rom) != END:
        sys.exit(f"expected a {END}-byte ROM, got {len(rom)}")

    name = re.search(r"_cartname\s*=\s*\[\s*[`'\"](.*?)[`'\"]", text)
    return rom, (name.group(1) if name else "cart.p8")


class BitReader:
    """PXA reads bits LSB-first within each byte."""

    def __init__(self, data):
        self.data, self.pos, self.bit = data, 0, 0

    def read_bit(self):
        v = (self.data[self.pos] >> self.bit) & 1
        self.bit += 1
        if self.bit == 8:
            self.bit, self.pos = 0, self.pos + 1
        return v

    def read_bits(self, n):
        v = 0
        for i in range(n):
            v |= self.read_bit() << i
        return v


def decompress_code(rom):
    """Decompress the code section (LZ77 + move-to-front literals)."""
    section = rom[CODE:END]
    if section[:4] != b"\x00pxa":
        sys.exit("code section is not PXA-compressed; unsupported")

    unc_size = (section[4] << 8) | section[5]
    br = BitReader(section[8:])
    mtf = list(range(256))
    out = bytearray()

    while len(out) < unc_size:
        if br.read_bit():
            # literal: unary-prefixed index into the move-to-front table
            extra = 0
            while br.read_bit():
                extra += 1
            idx = br.read_bits(4 + extra) + (((1 << extra) - 1) << 4)
            ch = mtf.pop(idx)
            mtf.insert(0, ch)
            out.append(ch)
        else:
            # back-reference: 5, 10 or 15 bits of offset
            offset_bits = (5 if br.read_bit() else 10) if br.read_bit() else 15
            offset = br.read_bits(offset_bits) + 1
            if offset == 1 and offset_bits != 5:
                # escape hatch: raw bytes until a NUL
                while True:
                    ch = br.read_bits(8)
                    if ch == 0:
                        break
                    out.append(ch)
            else:
                count = 3
                while True:
                    part = br.read_bits(3)
                    count += part
                    if part != 7:
                        break
                for _ in range(count):
                    out.append(out[-offset])

    return "".join(GLYPHS.get(b, chr(b)) for b in out)


def gfx_lines(rom):
    """128 rows of 128 pixels; each byte holds two pixels, low nibble first."""
    for row in range(128):
        base = GFX + row * 64
        yield "".join(f"{b & 0xf:x}{b >> 4:x}" for b in rom[base:base + 64])


def map_lines(rom):
    for row in range(32):
        base = MAP + row * 128
        yield rom[base:base + 128].hex()


def gff_lines(rom):
    for half in range(2):
        base = GFF + half * 128
        yield rom[base:base + 128].hex()


def sfx_lines(rom):
    """Each sfx is 68 bytes: 32 notes of 2 bytes, then mode/speed/loop."""
    for i in range(64):
        base = SFX + i * 68
        notes, mode, speed, lstart, lend = rom[base:base + 64], *rom[base + 64:base + 68]
        row = f"{mode:02x}{speed:02x}{lstart:02x}{lend:02x}"
        for n in range(32):
            lo, hi = notes[n * 2], notes[n * 2 + 1]
            pitch = lo & 0x3F
            waveform = ((lo >> 6) & 0x3) | ((hi & 0x1) << 2) | (0x8 if hi & 0x80 else 0)
            volume = (hi >> 1) & 0x7
            effect = (hi >> 4) & 0x7
            row += f"{pitch:02x}{waveform:x}{volume:x}{effect:x}"
        yield row


def music_lines(rom):
    """Each pattern is 4 bytes; the top bit of the first three carries flags."""
    for i in range(64):
        c = rom[MUSIC + i * 4:MUSIC + i * 4 + 4]
        flags = ((c[0] >> 7) & 1) | ((c[1] >> 7) & 1) << 1 | ((c[2] >> 7) & 1) << 2
        yield f"{flags:02x} " + "".join(f"{b & 0x7f:02x}" for b in c)


def trim(lines, empty):
    """PICO-8 omits trailing empty rows in the sfx/music sections."""
    lines = list(lines)
    while lines and lines[-1] == empty:
        lines.pop()
    return lines


def build_p8(rom, code):
    out = ["pico-8 cartridge // http://www.pico-8.com/version 42", "__lua__", code.rstrip("\n")]
    for header, lines in (
        ("__gfx__", gfx_lines(rom)),
        ("__gff__", gff_lines(rom)),
        ("__map__", map_lines(rom)),
        ("__sfx__", trim(sfx_lines(rom), "0" * 4 + "00000" * 32)),
        ("__music__", trim(music_lines(rom), "00 41424344")),
    ):
        lines = [ln for ln in lines]
        if any(set(ln) - {"0", " "} for ln in lines):
            out.append(header)
            out.extend(lines)
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--js", type=Path, default=Path("logic.js"), help="the HTML5 export (default: logic.js)")
    ap.add_argument("--out", type=Path, default=Path("."), help="output directory (default: .)")
    ap.add_argument("--rom", action="store_true", help="also dump the raw 32KB ROM")
    args = ap.parse_args()

    rom, cartname = read_cartdat(args.js)
    stem = Path(cartname).stem
    code = decompress_code(rom)
    args.out.mkdir(parents=True, exist_ok=True)

    def write(name, data):
        path = args.out / name
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8", newline="\n")
        else:
            path.write_bytes(data)
        print(f"{path}  {len(data):,} {'chars' if isinstance(data, str) else 'bytes'}")

    write(f"{stem}.lua", code)
    write(f"{stem}.p8", build_p8(rom, code))
    if args.rom:
        write(f"{stem}.rom", rom)


if __name__ == "__main__":
    main()
