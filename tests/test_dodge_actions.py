import pytest

from dodge.actions import ACTION_MASKS, DOWN, LEFT, RIGHT, UP, action_mask


def test_nine_actions_have_the_expected_button_masks():
    assert ACTION_MASKS == (
        0, LEFT, RIGHT, UP, DOWN,
        UP | LEFT, UP | RIGHT, DOWN | LEFT, DOWN | RIGHT,
    )


def test_invalid_action_is_rejected():
    with pytest.raises(ValueError):
        action_mask(9)
