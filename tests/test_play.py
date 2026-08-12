"""Tests for the button_mask() key-to-button mapping in play.py.

button_mask() is a plain function of a pygame key-state sequence -- it opens
no window and boots no cart, so it can be exercised directly without a
display or an event loop.
"""

import os
from collections import defaultdict

import pytest

pygame = pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from play import KEYMAP, button_mask


def _pressed(*keys):
    """Fake a pygame.key.get_pressed() result with only `keys` down.

    The real return value is a ScancodeWrapper indexed directly by the large
    pygame.K_* constants (e.g. K_LEFT == 1073741904), not a small 0..N list,
    so a plain fixed-size list can't stand in for it, and button_mask() may
    probe any key in KEYMAP regardless of which ones are "down" here.
    """
    state = defaultdict(bool)
    for k in keys:
        state[k] = True
    return state


def test_no_keys_pressed_gives_zero_mask():
    assert button_mask(_pressed()) == 0


@pytest.mark.parametrize("button_index", sorted(KEYMAP))
def test_each_mapped_key_sets_only_its_own_bit(button_index):
    for key in KEYMAP[button_index]:
        mask = button_mask(_pressed(key))
        assert mask == 1 << button_index, (
            f"key {pygame.key.name(key)!r} should set only bit {button_index}, "
            f"got mask {mask:#04x}"
        )


def test_arrow_and_wasd_variants_agree():
    # Every button in KEYMAP is bound to two keys (e.g. left arrow and 'a');
    # either one alone must produce the identical mask.
    for button_index, keys in KEYMAP.items():
        assert len(keys) == 2
        masks = {button_mask(_pressed(k)) for k in keys}
        assert masks == {1 << button_index}


def test_multiple_buttons_combine_with_bitwise_or():
    left, right = pygame.K_LEFT, pygame.K_x
    mask = button_mask(_pressed(left, right))
    assert mask == (1 << 0) | (1 << 5)


def test_unmapped_keys_are_ignored():
    assert button_mask(_pressed(pygame.K_SPACE, pygame.K_RETURN)) == 0


def test_holding_both_keys_of_the_same_button_is_still_one_bit():
    mask = button_mask(_pressed(pygame.K_LEFT, pygame.K_a))
    assert mask == 1 << 0
