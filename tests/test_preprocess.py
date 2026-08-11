from pico8.preprocess import STR_MASK, CMT_MASK, mask_literals, apply_edits


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
