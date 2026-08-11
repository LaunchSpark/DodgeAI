import pytest

from pico8.preprocess import STR_MASK, CMT_MASK, mask_literals, apply_edits
from pico8.preprocess import preprocess


def test_mask_preserves_length_and_newlines():
    src = 'a = "hi"  -- note\nb = 1\n'
    mask = mask_literals(src)
    assert len(mask) == len(src)
    assert mask.count("\n") == src.count("\n")


def test_mask_hides_string_contents():
    mask = mask_literals('print("press x")')
    assert "press" not in mask
    assert mask.startswith("print(")
    assert mask.count(STR_MASK) == len('"press x"')


def test_mask_hides_line_comments():
    mask = mask_literals("x = 1 -- x = 2\nz 3")
    assert mask.count("=") == 1
    assert mask.endswith("z 3")


def test_mask_hides_block_comments_but_keeps_newlines():
    src = "a=1\n--[[\n b=2\n]]\nc=3"
    mask = mask_literals(src)
    assert "b=2" not in mask
    assert mask.count("\n") == 4
    assert mask.endswith("c=3")


def test_apply_edits_runs_right_to_left():
    src = "abcdef"
    out = apply_edits(src, [(0, 1, "X"), (4, 6, "YY_Y")])
    assert out == "XbcdYY_Y"


@pytest.mark.parametrize("src,expected", [
    ("if btnp(❎) then x() end", "if btnp(5) then x() end"),
    ("if btnp(\U0001f17e️) then x() end", "if btnp(4) then x() end"),
    ("btnp(⬅️)", "btnp(0)"),
    ("btnp(➡️)", "btnp(1)"),
    ("btnp(⬆️)", "btnp(2)"),
    ("btnp(⬇️)", "btnp(3)"),
])
def test_bare_glyphs_become_button_numbers(src, expected):
    assert preprocess(src) == expected


def test_glyphs_inside_strings_are_left_alone():
    # dodge.lua:511 — HUD text telling the player which button to press
    src = 'print2("press "..(input==1 and "❎" or "  ").." to play again",20,120)'
    assert preprocess(src) == src


def test_not_equal_becomes_lua_inequality():
    assert preprocess("if a!=b then end") == "if a~=b then end"


def test_not_equal_inside_string_is_left_alone():
    assert preprocess('s="a!=b"') == 's="a!=b"'


def test_fixed_point_binary_literal():
    # dodge.lua:1051
    assert preprocess("fillp(0b1010010110100101.1)") == "fillp(42405.5)"


def test_plain_binary_literal():
    assert preprocess("x=0b1010") == "x=10"


def test_question_mark_shorthand():
    # dodge.lua:6
    assert preprocess(" ?_txt,_x,_y+1,1") == " print(_txt,_x,_y+1,1)"


def test_question_mark_shorthand_with_trailing_comment():
    src = " ?_txt,_x,_y+1,1 -- draw label\n"
    expected = " print(_txt,_x,_y+1,1) -- draw label\n"
    assert preprocess(src) == expected


def test_question_mark_shorthand_with_trailing_comment_no_newline():
    src = " ?_txt,_x,_y+1,1 -- draw label"
    expected = " print(_txt,_x,_y+1,1) -- draw label"
    assert preprocess(src) == expected


def test_binary_literal_not_matched_inside_identifier():
    assert preprocess("local x0b101 = 5") == "local x0b101 = 5"
    assert preprocess("local job0b1 = 5") == "local job0b1 = 5"


def test_fixed_point_binary_literal_still_converts():
    # guard against over-correcting the identifier-boundary fix
    assert preprocess("0b1010010110100101.1") == "42405.5"
