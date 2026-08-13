"""Drawing a danger field behind the game, in place of the background."""

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")

from pico8.backends import BACKGROUND_INDEX, PygameBackend
from pico8.colormap import LEVELS, VIRIDIS, colorize


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


def _backend():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    return PygameBackend(scale=2)


def test_the_field_replaces_only_background_pixels():
    """Everything the cart draws sits on top; only cls() pixels are taken."""
    backend = _backend()
    backend.cls(BACKGROUND_INDEX)
    backend.rectfill(40, 40, 60, 60, 7)          # a white block

    backend.set_field(np.add.outer(np.arange(32), np.arange(32)).astype(float))
    pixels = pygame.surfarray.array3d(backend._composited())

    assert tuple(pixels[0, 0]) == (68, 1, 84), "background takes the low colour"
    assert tuple(pixels[127, 127]) == (253, 231, 37), "and the high one"
    drawn = tuple(pixels[50, 50])
    assert drawn not in {tuple(c) for c in VIRIDIS}, f"drawn pixel overwritten: {drawn}"
    pygame.quit()


def test_a_coarse_field_is_scaled_without_interpolating():
    """32x32 into 128x128 stays blocky, which is honest about its resolution."""
    backend = _backend()
    field = np.zeros((32, 32))
    field[0, 0] = 1.0
    backend.set_field(field)
    block = backend._field_rgb[0:4, 0:4]
    assert len({tuple(c) for c in block.reshape(-1, 3)}) == 1, "one cell, one colour"
    np.testing.assert_array_equal(backend._field_rgb[0, 0], VIRIDIS[-1])
    pygame.quit()


def test_the_colour_range_is_smoothed_across_frames():
    """Rescaling per frame makes the picture flash every time an enemy moves."""
    backend = _backend()
    backend.set_field(np.zeros((32, 32)))
    backend.set_field(np.full((32, 32), 100.0))
    _, high = backend._field_range
    assert 0.0 < high < 100.0, "the range should lag, not snap"
    pygame.quit()


def test_clearing_the_field_restores_the_plain_frame():
    backend = _backend()
    backend.set_field(np.zeros((32, 32)))
    assert backend._field_rgb is not None
    backend.set_field(None)
    assert backend._field_rgb is None
    backend.flip()      # must not raise, and must take the plain path
    pygame.quit()
