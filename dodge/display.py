"""A pygame window showing the engine's frames, optionally over a danger field.

The native engine renders the cartridge's own 128x128 screen as palette
indices, so drawing a frame is a palette lookup and an upscale.
"""

import numpy as np
import pygame

from .colormap import colorize
from .palette import PICO8

SIZE = 128

# What the cart clears the screen to, and therefore the pixels a field overlay
# is allowed to replace.
BACKGROUND_INDEX = 12

_LOOKUP = np.array(PICO8, dtype=np.uint8)


class GameWindow:
    def __init__(self, scale=5, caption="Dodge"):
        pygame.init()
        self.window = pygame.display.set_mode((SIZE * scale, SIZE * scale))
        pygame.display.set_caption(caption)
        self._field_rgb = None
        self._field_range = None

    def set_caption(self, caption):
        pygame.display.set_caption(caption)

    def set_field(self, field, smoothing=0.1):
        """Show a scalar field behind the game, in place of the background.

        The cart clears to palette index 12 and draws everything else over it,
        so replacing exactly those pixels puts the field behind the action.
        The score bar's backing is also index 12, so it picks up the field's
        colours too, which is harmless.

        The colour range follows the field's own range through an exponential
        average rather than being rescaled per frame. Per-frame rescaling makes
        the whole picture flash whenever the hottest cell changes, which is
        every time an enemy moves.
        """
        if field is None:
            self._field_rgb = None
            return
        field = np.asarray(field, dtype=np.float32)
        low, high = float(field.min()), float(field.max())
        if self._field_range is None:
            self._field_range = (low, high)
        else:
            previous_low, previous_high = self._field_range
            self._field_range = (
                previous_low + smoothing * (low - previous_low),
                previous_high + smoothing * (high - previous_high),
            )
        rgb = colorize(field, *self._field_range)
        # Nearest-neighbour to 128x128: the field is coarse and honest about
        # it, and interpolating would imply a resolution it does not have.
        rows = np.arange(SIZE) * field.shape[0] // SIZE
        columns = np.arange(SIZE) * field.shape[1] // SIZE
        self._field_rgb = rgb[rows][:, columns]

    def compose(self, pixels):
        """One frame of palette indices, (row, column), as RGB."""
        pixels = np.asarray(pixels)
        rgb = _LOOKUP[pixels & 15]
        if self._field_rgb is not None:
            background = pixels == BACKGROUND_INDEX
            rgb[background] = self._field_rgb[background]
        return rgb

    def show(self, pixels):
        # Surfarray is (x, y); frames are (row, column).
        surface = pygame.surfarray.make_surface(self.compose(pixels).transpose(1, 0, 2))
        self.window.blit(pygame.transform.scale(surface, self.window.get_size()), (0, 0))
        pygame.display.flip()
