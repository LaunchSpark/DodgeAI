"""The game window: palette lookup, and a danger field behind the game."""

import os

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from dodge.colormap import LEVELS, VIRIDIS, colorize
from dodge.display import BACKGROUND_INDEX, GameWindow
from dodge.palette import PICO8


def test_palette_has_sixteen_colours():
    assert len(PICO8) == 16
    assert PICO8[0] == (0, 0, 0)
    assert PICO8[7] == (255, 241, 232)


def test_viridis_is_a_full_ramp_from_dark_blue_to_yellow():
    assert VIRIDIS.shape == (LEVELS, 3)
    assert VIRIDIS.dtype == np.uint8
    assert tuple(VIRIDIS[0]) == (68, 1, 84)
    assert tuple(VIRIDIS[-1]) == (253, 231, 37)
    # Perceptually uniform means monotonic in lightness, which is what makes a
    # gradient read as a gradient rather than as bands.
    lightness = VIRIDIS.astype(int).sum(axis=1)
    # Allow a rounding step: interpolating ten control points to 256 uint8
    # entries puts a -2 at two of the joins, which is quantisation and not a
    # dip in the ramp.
    assert np.all(np.diff(lightness) >= -2)
    assert lightness[-1] > 3 * lightness[0], "and it must brighten overall"


def test_colorize_clamps_outside_the_given_range():
    values = np.array([-5.0, 0.0, 0.5, 1.0, 5.0])
    rgb = colorize(values, 0.0, 1.0)
    np.testing.assert_array_equal(rgb[0], VIRIDIS[0])
    np.testing.assert_array_equal(rgb[-1], VIRIDIS[-1])
    assert tuple(rgb[2]) not in {tuple(VIRIDIS[0]), tuple(VIRIDIS[-1])}


@pytest.fixture
def window():
    yield GameWindow(scale=2)
    pygame.quit()


def _frame():
    """A cleared screen with a white block drawn on it."""
    pixels = np.full((128, 128), BACKGROUND_INDEX, dtype=np.uint8)
    pixels[40:61, 40:61] = 7
    return pixels


def test_frames_are_drawn_in_the_pico8_palette(window):
    rgb = window.compose(_frame())
    assert tuple(rgb[0, 0]) == PICO8[BACKGROUND_INDEX]
    assert tuple(rgb[50, 50]) == PICO8[7]


def test_the_field_replaces_only_background_pixels(window):
    """Everything the cart draws sits on top; only cleared pixels are taken."""
    window.set_field(np.add.outer(np.arange(32), np.arange(32)).astype(float))
    rgb = window.compose(_frame())

    assert tuple(rgb[0, 0]) == (68, 1, 84), "background takes the low colour"
    assert tuple(rgb[127, 127]) == (253, 231, 37), "and the high one"
    assert tuple(rgb[50, 50]) == PICO8[7], "drawn pixels are not overwritten"


def test_a_coarse_field_is_scaled_without_interpolating(window):
    """32x32 into 128x128 stays blocky, which is honest about its resolution."""
    field = np.zeros((32, 32))
    field[0, 0] = 1.0
    window.set_field(field)
    block = window._field_rgb[0:4, 0:4]
    assert len({tuple(c) for c in block.reshape(-1, 3)}) == 1, "one cell, one colour"
    np.testing.assert_array_equal(window._field_rgb[0, 0], VIRIDIS[-1])


def test_the_colour_range_is_smoothed_across_frames(window):
    """Rescaling per frame makes the picture flash every time an enemy moves."""
    window.set_field(np.zeros((32, 32)))
    window.set_field(np.full((32, 32), 100.0))
    high = window._field_range[-1]
    assert 0.0 < high < 100.0, "the range should lag, not snap"


def test_clearing_the_field_restores_the_plain_frame(window):
    window.set_field(np.zeros((32, 32)))
    assert window._field_rgb is not None
    window.set_field(None)
    assert window._field_rgb is None
    window.show(_frame())      # must not raise, and must take the plain path
    assert tuple(window.compose(_frame())[0, 0]) == PICO8[BACKGROUND_INDEX]


def test_an_overlay_draws_its_boxes_and_paths_over_the_frame(window):
    from dodge.overlay import Overlay
    overlay = Overlay(boxes=[(10, 10, 20, 20, (255, 0, 77))],
                      paths=[([(100, 5), (100, 120)], (255, 241, 232))])
    window.show(_frame(), overlay)
    # scale 2: the box's top-left corner and a point on the vertical path
    assert tuple(window.window.get_at((20, 20)))[:3] == (255, 0, 77)
    assert tuple(window.window.get_at((200, 100)))[:3] == (255, 241, 232)
    assert tuple(window.window.get_at((2, 2)))[:3] == PICO8[BACKGROUND_INDEX], "nothing else drawn"


def test_the_field_mean_takes_the_middle_colour(window):
    """A field that is mostly high would otherwise be nearly all one colour."""
    field = np.ones((32, 32))
    field[0, :4] = 0.0                     # a few low cells; the mean is near 1
    window.set_field(field)
    low, middle, high = window._field_range
    assert low == 0.0 and high == 1.0 and 0.99 < middle < 1.0
    np.testing.assert_array_equal(window._field_rgb[0, 0], VIRIDIS[0])
    np.testing.assert_array_equal(window._field_rgb[127, 127], VIRIDIS[-1])
    rgb = colorize(np.array([0.5]), 0.0, 1.0)[0]
    field2 = np.linspace(0.0, 1.0, 32 * 32).reshape(32, 32) ** 4   # mean 0.2
    window._field_range = None
    window.set_field(field2)
    mean_cell = np.unravel_index(np.argmin(np.abs(field2 - field2.mean())), field2.shape)
    row, column = mean_cell[0] * 4, mean_cell[1] * 4
    assert np.abs(window._field_rgb[row, column].astype(int) - rgb.astype(int)).max() <= 12
