"""What a player can ask the game window to draw on top of the game.

The window knows nothing about any particular bot: a bot that wants to show
its thinking builds an `Overlay` from its last decision, and the window draws
whatever is in it. Everything is in game coordinates (0-128, x right, y down).
"""

from dataclasses import dataclass, field as dataclass_field

import numpy as np

# PICO-8 colours, so overlays sit naturally on the cart's own palette.
THREAT = (255, 0, 77)       # something that kills on contact
HAZARD = (255, 163, 0)      # pattern rectangles and blasts
PICKUP = (0, 228, 54)       # powerups
PLAYER = (255, 241, 232)    # the player's own plan, in the player's white
MUTED = (131, 118, 156)     # predicted to be gone, or not yet dangerous


@dataclass
class Overlay:
    """One frame of a player's own picture of the game.

    field: a scalar map behind the action, rows = y and columns = x, at any
        resolution; drawn on the background pixels only, coloured low to high.
    boxes: (x0, y0, x1, y1, rgb) outlines.
    paths: (points, rgb) polylines, points as [(x, y), ...].
    """
    field: np.ndarray | None = None
    boxes: list = dataclass_field(default_factory=list)
    paths: list = dataclass_field(default_factory=list)
