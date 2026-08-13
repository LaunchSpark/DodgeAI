#!/usr/bin/env python3
"""Extract a PICO-8 cartridge from a ``.p8.png`` or HTML5 export.

Cartridge PNGs encode a 32,768-byte ROM in their pixels. The extractor also
accepts older HTML5 exports that store the same ROM in a ``_cartdat`` array.

This pulls that ROM out and turns it back into editable sources:

    dodge.p8    full cartridge in PICO-8's plain-text format (loadable in PICO-8)
    dodge.lua   just the Lua, for reading/diffing
    dodge.rom   the raw 32KB ROM, with --rom

Usage:
    python tools/extract_cart.py                    # reads ./dodge.p8.png, writes ./
    python tools/extract_cart.py --cart x.p8.png --out b/ --rom
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pico8.cart import GFX, MAP, GFF, MUSIC, SFX, CODE, END, GLYPHS, read_rom, decompress_code


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
    ap.add_argument("--cart", "--js", dest="cart", type=Path,
                    default=Path("dodge.p8.png"),
                    help="the cartridge PNG or HTML5 export (default: dodge.p8.png)")
    ap.add_argument("--out", type=Path, default=Path("."), help="output directory (default: .)")
    ap.add_argument("--rom", action="store_true", help="also dump the raw 32KB ROM")
    args = ap.parse_args()

    rom, cartname = read_rom(args.cart)
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
