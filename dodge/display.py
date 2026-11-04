"""A pygame window showing the engine's frames, optionally over a danger field
and under a player's Overlay (dodge.overlay).

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
        current = (float(field.min()), float(field.mean()), float(field.max()))
        if self._field_range is None:
            self._field_range = current
        else:
            self._field_range = tuple(
                previous + smoothing * (value - previous)
                for previous, value in zip(self._field_range, current))
        # The mean takes the middle colour, so a field that is mostly near one
        # end (most cells safe, say) still spreads across the whole ramp.
        low, middle, high = self._field_range
        below = 0.5 * (field - low) / max(middle - low, 1e-6)
        above = 0.5 + 0.5 * (field - middle) / max(high - middle, 1e-6)
        rgb = colorize(np.where(field < middle, below, above), 0.0, 1.0)
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

    def show(self, pixels, overlay=None):
        """Draw a frame, and an Overlay's boxes and paths over it if given.

        The overlay's field is not taken from here: it goes through set_field,
        which keeps the colour range steady across frames.
        """
        # Surfarray is (x, y); frames are (row, column).
        surface = pygame.surfarray.make_surface(self.compose(pixels).transpose(1, 0, 2))
        self.window.blit(pygame.transform.scale(surface, self.window.get_size()), (0, 0))
        if overlay is not None:
            self._draw_shapes(overlay)
        pygame.display.flip()

    def _draw_shapes(self, overlay):
        # Drawn on the scaled window, so lines stay one screen pixel thin
        # rather than one game pixel.
        scale = self.window.get_width() / SIZE
        width = max(1, round(scale / 3))
        for x0, y0, x1, y1, colour in overlay.boxes:
            rect = pygame.Rect(round(x0 * scale), round(y0 * scale),
                               max(round((x1 - x0) * scale), 1), max(round((y1 - y0) * scale), 1))
            pygame.draw.rect(self.window, colour, rect, width=width)
        for points, colour in overlay.paths:
            if len(points) >= 2:
                pygame.draw.lines(self.window, colour, False,
                                  [(x * scale, y * scale) for x, y in points], width)
