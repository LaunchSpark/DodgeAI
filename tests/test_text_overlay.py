from pico8.backends import split_glyphs


def test_plain_text_is_one_chunk():
    assert split_glyphs("score:12") == [("text", "score:12")]


def test_glyph_is_split_out():
    assert split_glyphs("press ❎ to play") == [
        ("text", "press "), ("glyph", "❎"), ("text", " to play"),
    ]


def test_adjacent_glyphs():
    assert split_glyphs("⬅️➡️") == [
        ("glyph", "⬅️"), ("glyph", "➡️"),
    ]


def test_empty_string():
    assert split_glyphs("") == []
