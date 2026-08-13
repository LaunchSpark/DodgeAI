"""Decode PICO-8 cartridges from ``.p8.png`` or HTML5 exports.

PICO-8 cartridge PNGs hide one ROM byte in the low two bits of each ARGB
pixel. HTML5 exports carry the same 32,768-byte ROM in a ``_cartdat`` array.

This module pulls that ROM out and decodes it: the compressed Lua code
section and the sprite sheet. Shared by tools/extract_cart.py and the runtime.
"""

import json
from pathlib import Path
import re

from PIL import Image

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


LEGACY_CODE_TABLE = b"#\n 0123456789abcdefghijklmnopqrstuvwxyz!#%(){}[]<>+=/*:;.,~_"


def _read_png_rom(path):
    with Image.open(path) as source:
        image = source.convert("RGBA")
        if image.size != (160, 205):
            raise ValueError(f"expected a 160x205 PICO-8 cart PNG, got {image.size}")
        pixels = (image.get_flattened_data() if hasattr(image, "get_flattened_data")
                  else image.getdata())
        rom = bytes(
            ((a & 3) << 6) | ((r & 3) << 4) | ((g & 3) << 2) | (b & 3)
            for r, g, b, a in pixels[:END]
        )
    return rom, path.name.removesuffix(".png")


def read_rom(path):
    """Read a PICO-8 ROM from a cartridge PNG or an HTML5 JS export."""
    path = Path(path)
    if path.suffix.lower() == ".png":
        return _read_png_rom(path)

    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"_cartdat\s*=\s*(\[.*?\])\s*;", text, re.S)
    if not m:
        raise ValueError(f"no _cartdat array found in {path}")
    rom = bytes(json.loads(m.group(1)))
    if len(rom) != END:
        raise ValueError(f"expected a {END}-byte ROM, got {len(rom)}")

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


def _decompress_legacy(section):
    output_size = (section[4] << 8) | section[5]
    source = 8
    output = bytearray()
    while len(output) < output_size:
        value = section[source]
        source += 1
        if value == 0:
            output.append(section[source])
            source += 1
        elif value <= 0x3B:
            output.append(LEGACY_CODE_TABLE[value])
        else:
            packed = section[source]
            source += 1
            offset = (value - 0x3C) * 16 + (packed & 0x0F)
            length = (packed >> 4) + 2
            for _ in range(length):
                output.append(output[-offset])
    return bytes(output[:output_size])


def decompress_code(rom):
    """Decompress the cartridge's legacy or PXA-compressed Lua section."""
    section = rom[CODE:END]
    if section[:4] == b":c:\x00":
        output = _decompress_legacy(section)
        return "".join(GLYPHS.get(b, chr(b)) for b in output)
    if section[:4] != b"\x00pxa":
        output = section.split(b"\x00", 1)[0]
        return "".join(GLYPHS.get(b, chr(b)) for b in output)

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
