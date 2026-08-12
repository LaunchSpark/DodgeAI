"""Discrete PPO actions mapped to the cart's four direction buttons."""

ACTION_NAMES = (
    "idle",
    "left",
    "right",
    "up",
    "down",
    "up-left",
    "up-right",
    "down-left",
    "down-right",
)

LEFT = 1 << 0
RIGHT = 1 << 1
UP = 1 << 2
DOWN = 1 << 3

ACTION_MASKS = (
    0,
    LEFT,
    RIGHT,
    UP,
    DOWN,
    UP | LEFT,
    UP | RIGHT,
    DOWN | LEFT,
    DOWN | RIGHT,
)


def action_mask(action):
    action = int(action)
    if not 0 <= action < len(ACTION_MASKS):
        raise ValueError(f"action must be in 0..{len(ACTION_MASKS) - 1}")
    return ACTION_MASKS[action]
