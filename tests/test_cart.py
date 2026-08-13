from pathlib import Path
import pytest
from pico8 import cart

ROM = Path("dodge.p8.png")
pytestmark = pytest.mark.skipif(not ROM.exists(), reason="dodge.p8.png not present")


def test_read_rom_returns_full_cart():
    rom, name = cart.read_rom(ROM)
    assert len(rom) == 32768
    assert name == "dodge.p8"


def test_decompress_code_matches_known_size():
    rom, _ = cart.read_rom(ROM)
    code = cart.decompress_code(rom)
    assert len(code) == 36813          # 36,804 bytes plus expanded P8SCII glyphs
    assert code.startswith("-- dodge\n-- a game by artridge")
    assert "function updategame()" in code


def test_sprite_sheet_shape_and_content():
    rom, _ = cart.read_rom(ROM)
    sheet = cart.sprite_sheet(rom)
    assert len(sheet) == 128
    assert all(len(row) == 128 for row in sheet)
    assert all(0 <= px <= 15 for row in sheet for px in row)
    assert any(px != 0 for row in sheet for px in row)   # sheet is not blank
