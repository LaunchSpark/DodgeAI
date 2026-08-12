from pathlib import Path

import pytest

from pico8.preprocess import preprocess

luajit = pytest.importorskip("lupa.luajit21")
SOURCE = Path("dodge.lua")
pytestmark = pytest.mark.skipif(not SOURCE.exists(), reason="dodge.lua not extracted")


@pytest.fixture(scope="module")
def translated():
    return preprocess(SOURCE.read_text(encoding="utf-8"))


def test_whole_cart_is_valid_lua(translated):
    """Compile (do not run) the translated cart under LuaJIT."""
    runtime = luajit.LuaRuntime()
    compile_error = runtime.eval("function(s) local f, e = loadstring(s) return e end")
    error = compile_error(translated)
    assert error is None, f"translated cart failed to compile: {error}"


def test_compile_check_rejects_invalid_lua():
    """Guard against a vacuous assertion: the check must be able to fail."""
    runtime = luajit.LuaRuntime()
    compile_error = runtime.eval("function(s) local f, e = loadstring(s) return e end")
    error = compile_error("this is not valid lua $$$")
    assert error is not None
    assert isinstance(error, str)


def test_no_pico8_dialect_remains(translated):
    from pico8.preprocess import mask_literals, BUTTONS
    mask = mask_literals(translated)
    assert "!=" not in mask
    assert "0b" not in mask
    for glyph in BUTTONS:
        assert glyph not in mask, f"bare glyph {glyph!r} survived"
    for op in ("+=", "-=", "*=", "/=", "%="):
        assert op not in mask, f"compound assignment {op} survived"


def test_hud_glyphs_survive_inside_strings(translated):
    # These are button prompts drawn to screen, not constants
    assert '"❎"' in translated
    assert "to play again" in translated
