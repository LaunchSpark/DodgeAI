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


def test_binary_literal_after_concat_operator():
    # regression: the `.` in the lookbehind wrongly blocked `..0b...`
    assert preprocess("a=a..0b1010") == "a=a..10"


def test_binary_literal_after_unary_minus():
    assert preprocess("x=-0b1010") == "x=-10"


def test_binary_literal_still_not_matched_inside_identifier():
    # guard against over-correcting the concat-operator fix
    assert preprocess("local x0b101 = 5") == "local x0b101 = 5"
    assert preprocess("local job0b1 = 5") == "local job0b1 = 5"


@pytest.mark.parametrize("src,expected", [
    ("cursi+=1", "cursi = cursi + (1)"),
    ("shake*=0.95", "shake = shake * (0.95)"),
    ("_p.age+=1", "_p.age = _p.age + (1)"),
    ("_e.x-=0.5*frz", "_e.x = _e.x - (0.5*frz)"),
    ("enemiestats[i]+=1", "enemiestats[i] = enemiestats[i] + (1)"),
])
def test_compound_assignment_shapes(src, expected):
    assert preprocess(src) == expected


def test_compound_assignment_with_operator_inside_index():
    # dodge.lua:2222 — the index expression contains a '+' of its own
    assert preprocess("enemiestats[e.p+1]-=2") == \
        "enemiestats[e.p+1] = enemiestats[e.p+1] - (2)"


def test_compound_assignment_with_nested_index():
    # dodge.lua:621
    assert preprocess("patterns[ix].prob-=6*patterns[ix].counter") == \
        "patterns[ix].prob = patterns[ix].prob - (6*patterns[ix].counter)"


def test_compound_assignment_stops_before_trailing_comment():
    assert preprocess("size+=0.045 -- ramp") == "size = size + (0.045) -- ramp"


def test_equality_and_comparison_are_untouched():
    assert preprocess("if a==b and c>=d and e<=f then end") == \
        "if a==b and c>=d and e<=f then end"


def test_short_if_simple():
    assert preprocess("if (input==0) spr(192,71,56,1,2)") == \
        "if (input==0) then spr(192,71,56,1,2) end"


def test_short_if_with_compound_assignment():
    # dodge.lua:285 — glyph, short if and compound assignment on one line
    assert preprocess("if (btnp(⬇️)) cursi+=1") == \
        "if (btnp(3)) then cursi = cursi + (1) end"


def test_short_if_with_nested_parens_in_condition():
    # dodge.lua:397 — closing paren must be found by balance, not first match
    src = 'if ((isdead or not hasplayed) or n!="gameplay") print2("<  >",65,y)'
    assert preprocess(src) == \
        'if ((isdead or not hasplayed) or n~="gameplay") then print2("<  >",65,y) end'


def test_short_if_with_two_statement_body():
    # dodge.lua:2131 — the body is everything to end of line
    assert preprocess("if (r.sh>1.99) r.sh=2 initspawns()") == \
        "if (r.sh>1.99) then r.sh=2 initspawns() end"


def test_short_if_end_goes_before_trailing_comment():
    assert preprocess("if (a) b() -- why") == "if (a) then b() end -- why"


def test_short_if_preserves_indentation_of_body():
    assert preprocess("  if (a) b()") == "  if (a) then b() end"


def test_explicit_then_is_not_transformed():
    assert preprocess("if (a) then b() end") == "if (a) then b() end"


def test_multiline_condition_is_not_transformed():
    # dodge.lua:248 — parens stay open at end of line, so this is an ordinary if
    src = "if (stat(34)==1 and\n    70>stat(32)) or\n    (btnp(0) and cursi==i2)\n then\n  x()\n end"
    assert preprocess(src) == src


def test_bare_if_without_parens_is_not_transformed():
    assert preprocess("if a then b() end") == "if a then b() end"


def test_multi_group_condition_with_or_is_not_transformed():
    # dodge.lua:193 — first balanced close-paren is mid-condition, not the
    # end of it; the top-level `then` after the second group means this is
    # already long-form and must be left untouched by the short-if pass.
    # The `!=` -> `~=` rewrite is a separate, unrelated pass and still
    # applies normally.
    src = "if (target==2 and trsy<=-128) or (target!=2 and trsy>=128) then"
    expected = "if (target==2 and trsy<=-128) or (target~=2 and trsy>=128) then"
    assert preprocess(src) == expected


def test_multi_group_condition_with_and_is_not_transformed():
    src = "if (target==2 and trsy<=-128) and (target!=2 and trsy>=128) then"
    expected = "if (target==2 and trsy<=-128) and (target~=2 and trsy>=128) then"
    assert preprocess(src) == expected


def test_short_if_body_starting_with_then_prefixed_identifier():
    # "thensomething" is not the keyword `then` — word-bounded match required
    assert preprocess("if (a) thensomething=1") == \
        "if (a) then thensomething=1 end"


def test_short_if_body_contains_string_with_word_then():
    assert preprocess('if (a) print("then")') == \
        'if (a) then print("then") end'
