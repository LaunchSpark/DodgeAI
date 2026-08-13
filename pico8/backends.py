"""Drawing targets for the PICO-8 machine.

The cart's draw path consumes randomness from the same stream as gameplay --
spawntrail() inside drawgame() draws two values per powerup enemy per frame --
so the draw traversal itself can never be skipped: doing so would change which
enemies spawn. What can go is the work at the far end of it. A backend that
declares ``draws = False`` gets the drawing builtins bound as Lua no-ops
instead of Python callbacks (see api.install), which leaves Lua walking the
same code and evaluating the same arguments while removing roughly nineteen
Lua-to-Python crossings per frame. Gameplay is identical frame for frame; the
Lua interpreter runs about 2.3x faster.
"""


class Backend:
    """Interface every backend implements. Coordinates are PICO-8 pixels."""

    # Whether drawing calls should reach Python at all.
    draws = True

    def cls(self, col): raise NotImplementedError
    def pset(self, x, y, c): raise NotImplementedError
    def rect(self, x0, y0, x1, y1, c): raise NotImplementedError
    def rectfill(self, x0, y0, x1, y1, c): raise NotImplementedError
    def circfill(self, x, y, r, c): raise NotImplementedError
    def ovalfill(self, x0, y0, x1, y1, c): raise NotImplementedError
    def line(self, x0, y0, x1, y1, c): raise NotImplementedError
    def spr(self, n, x, y, w, h): raise NotImplementedError
    def print_(self, text, x, y, c): raise NotImplementedError
    def pal(self, c0, c1, mode): raise NotImplementedError
    def fillp(self, pattern): raise NotImplementedError
    def camera(self, x, y): raise NotImplementedError
    def flip(self): raise NotImplementedError


class NullBackend(Backend):
    """Discards drawing entirely -- the builtins never cross into Python.

    The methods below are kept so the class still satisfies Backend and so a
    caller that reaches for one directly gets a no-op rather than an error.
    """

    draws = False

    def cls(self, col): pass
    def pset(self, x, y, c): pass
    def rect(self, x0, y0, x1, y1, c): pass
    def rectfill(self, x0, y0, x1, y1, c): pass
    def circfill(self, x, y, r, c): pass
    def ovalfill(self, x0, y0, x1, y1, c): pass
    def line(self, x0, y0, x1, y1, c): pass
    def spr(self, n, x, y, w, h): pass
    def print_(self, text, x, y, c): pass
    def pal(self, c0, c1, mode): pass
    def fillp(self, pattern): pass
    def camera(self, x, y): pass
    def flip(self): pass


import re

import numpy as np
import pygame

from .colormap import colorize
from .palette import PICO8
from .preprocess import BUTTONS

# What the cart clears the screen to (dodge.lua:182), and therefore the pixels
# a field overlay is allowed to replace.
BACKGROUND_INDEX = 12

_GLYPH_SPLIT = re.compile("(" + "|".join(re.escape(g) for g in BUTTONS) + ")")

# What each button glyph is drawn as, since no system font has them. Built
# from BUTTONS itself (rather than retyped) so the keys here are guaranteed
# to be the exact same glyph strings split_glyphs() ever hands back.
_LABEL_BY_BUTTON = {0: "<", 1: ">", 2: "^", 3: "v", 4: "O", 5: "X"}
GLYPH_LABEL = {glyph: _LABEL_BY_BUTTON[button] for glyph, button in BUTTONS.items()}


def split_glyphs(text):
    """Split a PICO-8 string into ('text'|'glyph', chunk) pairs."""
    return [("glyph" if chunk in BUTTONS else "text", chunk)
            for chunk in _GLYPH_SPLIT.split(text) if chunk]


class PygameBackend(Backend):
    """Draws to a 128x128 indexed surface, scaled to the window on flip.

    Colour values passed in are PICO-8 palette indices, and the surface stores
    them directly. That makes pal(c, c2, 1) a palette swap rather than a
    per-pixel rewrite -- which matters because the cart draws the entire game
    in colours 12 and 1 and remaps them at the end of every frame.
    """

    SIZE = 128

    def __init__(self, scale=5, caption="dodge"):
        pygame.init()
        self.scale = scale
        self.window = pygame.display.set_mode((self.SIZE * scale, self.SIZE * scale))
        pygame.display.set_caption(caption)
        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace",
                                        max(8, 4 * scale))

        self.surface = pygame.Surface((self.SIZE, self.SIZE), depth=8)
        self.surface.set_palette(PICO8)
        self.sheet = None

        self.draw_pal = list(range(16))     # pal(c, c1, 0)
        self.display_pal = list(range(16))  # pal(c, c1, 1)
        self.cam_x = self.cam_y = 0
        self.pen = 6
        self.text_queue = []
        self._field_rgb = None
        self._field_range = None

    # -- helpers ----------------------------------------------------------
    def load_sheet(self, rows):
        """Build a transparent surface from 128x128 palette indices."""
        sheet = pygame.Surface((self.SIZE, self.SIZE), depth=8)
        sheet.set_palette(PICO8)
        for y, row in enumerate(rows):
            for x, index in enumerate(row):
                sheet.set_at((x, y), index)
        sheet.set_colorkey(0)               # PICO-8 treats colour 0 as transparent
        self.sheet = sheet

    def _colour(self, c):
        if c is None:
            c = self.pen
        else:
            self.pen = int(c) & 15
        return self.draw_pal[int(c) & 15]

    def _pt(self, x, y):
        return int(x) - self.cam_x, int(y) - self.cam_y

    # -- Backend ----------------------------------------------------------
    def cls(self, col=0):
        self.surface.fill(self.draw_pal[int(col) & 15])
        # Note: real PICO-8's cls() does not touch the camera offset. See
        # dodge.lua's global _draw() (dodge.lua:2214-2262): camera(shakex,
        # shakey) is called, then cls(12), then _drw() -- if cls() reset the
        # camera here, the screen-shake effect applied on real gameplay hits
        # (shake += 0.07 in dodge.lua:565, 746, 875) would be silently
        # discarded on every frame it's active.

    def pset(self, x, y, c=None):
        px, py = self._pt(x, y)
        if 0 <= px < self.SIZE and 0 <= py < self.SIZE:
            self.surface.set_at((px, py), self._colour(c))

    def rect(self, x0, y0, x1, y1, c=None):
        colour = self._colour(c)
        ax, ay = self._pt(x0, y0)
        bx, by = self._pt(x1, y1)
        pygame.draw.rect(self.surface, colour,
                         pygame.Rect(min(ax, bx), min(ay, by),
                                     abs(bx - ax) + 1, abs(by - ay) + 1), width=1)

    def rectfill(self, x0, y0, x1, y1, c=None):
        colour = self._colour(c)
        ax, ay = self._pt(x0, y0)
        bx, by = self._pt(x1, y1)
        pygame.draw.rect(self.surface, colour,
                         pygame.Rect(min(ax, bx), min(ay, by),
                                     abs(bx - ax) + 1, abs(by - ay) + 1))

    def circfill(self, x, y, r=4, c=None):
        colour = self._colour(c)
        px, py = self._pt(x, y)
        radius = int(r)
        if radius <= 0:
            if 0 <= px < self.SIZE and 0 <= py < self.SIZE:
                self.surface.set_at((px, py), colour)
            return
        pygame.draw.circle(self.surface, colour, (px, py), radius)

    def ovalfill(self, x0, y0, x1, y1, c=None):
        colour = self._colour(c)
        ax, ay = self._pt(x0, y0)
        bx, by = self._pt(x1, y1)
        rect = pygame.Rect(min(ax, bx), min(ay, by), abs(bx - ax) + 1, abs(by - ay) + 1)
        pygame.draw.ellipse(self.surface, colour, rect)

    def line(self, x0, y0, x1, y1, c=None):
        colour = self._colour(c)
        pygame.draw.line(self.surface, colour, self._pt(x0, y0), self._pt(x1, y1))

    def spr(self, n, x, y, w=1, h=1):
        if self.sheet is None:
            return
        n = int(n)
        sx, sy = (n % 16) * 8, (n // 16) * 8
        area = pygame.Rect(sx, sy, int(w) * 8, int(h) * 8)
        self.surface.blit(self.sheet, self._pt(x, y), area)

    def print_(self, text, x=0, y=0, c=None):
        self.text_queue.append((str(text), int(x) - self.cam_x,
                                int(y) - self.cam_y, self._colour(c)))

    def pal(self, c0=None, c1=None, mode=0):
        if c0 is None:
            self.draw_pal = list(range(16))
            self.display_pal = list(range(16))
            self._apply_display_palette()
            return
        if int(mode) == 1:
            self.display_pal[int(c0) & 15] = int(c1) & 15
            self._apply_display_palette()
        else:
            self.draw_pal[int(c0) & 15] = int(c1) & 15

    def _apply_display_palette(self):
        self.surface.set_palette([PICO8[self.display_pal[i]] for i in range(16)])

    def fillp(self, pattern=None):
        pass                                 # approximated: solid fills

    def camera(self, x=0, y=0):
        self.cam_x, self.cam_y = int(x), int(y)

    # -- danger field overlay ---------------------------------------------
    def set_field(self, field, smoothing=0.1):
        """Show a scalar field behind the game, in place of the background.

        The cart clears to palette index 12 and draws everything else over it,
        so replacing exactly those pixels puts the field behind the action with
        no changes to any drawing primitive. Only two things in the cart use
        index 12 -- cls() and the score bar's backing (dodge.lua:484) -- so the
        score readout picks up the field's colours too, which is harmless.

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
        rows = np.arange(self.SIZE) * field.shape[0] // self.SIZE
        columns = np.arange(self.SIZE) * field.shape[1] // self.SIZE
        # Surfarray is (x, y), the field is (row, column) -- hence the swap.
        self._field_rgb = np.ascontiguousarray(
            rgb[rows][:, columns].transpose(1, 0, 2)
        )

    def _composited(self):
        """The game frame as RGB, with background pixels showing the field."""
        indices = pygame.surfarray.array2d(self.surface)
        lookup = np.array([PICO8[self.display_pal[i]] for i in range(16)],
                          dtype=np.uint8)
        rgb = lookup[indices]
        background = indices == BACKGROUND_INDEX
        rgb[background] = self._field_rgb[background]
        return pygame.surfarray.make_surface(rgb)

    def flip(self):
        self._apply_display_palette()
        # pygame.transform.scale() into a destination surface requires the
        # destination to match the source's format exactly; our indexed
        # 8-bit surface can't be scaled directly into the 32-bit display
        # surface (raises ValueError: "Source and destination surfaces need
        # the same format."). Scale into a fresh surface instead and blit
        # that -- blit converts formats, transform.scale-into-dest does not.
        source = self.surface if self._field_rgb is None else self._composited()
        scaled = pygame.transform.scale(source, self.window.get_size())
        self.window.blit(scaled, (0, 0))
        self._draw_text()
        # The cart's own boot splash calls flip() ~66 times from top-level
        # code before play.py's main loop ever runs, so without pumping the
        # event queue here the window would sit unresponsive that whole time
        # and Windows may mark it "not responding".
        pygame.event.pump()
        pygame.display.flip()
        self.text_queue.clear()

    def _draw_text(self):
        """Draw queued text onto the upscaled window.

        PICO-8 characters occupy a 4x6 cell, so a chunk of n characters is
        n*4 pixels wide in game space; that is the advance used here.
        """
        s = self.scale
        for text, x, y, colour in self.text_queue:
            rgb = PICO8[self.display_pal[colour]]
            cursor = x
            for kind, chunk in split_glyphs(text):
                if kind == "glyph":
                    label = GLYPH_LABEL[chunk]
                    box = pygame.Rect(cursor * s, y * s, 7 * s, 5 * s)
                    pygame.draw.rect(self.window, rgb, box, width=max(1, s // 2))
                    glyph = self.font.render(label, True, rgb)
                    self.window.blit(glyph, glyph.get_rect(center=box.center))
                    cursor += 8
                else:
                    self.window.blit(self.font.render(chunk, True, rgb),
                                     (cursor * s, y * s))
                    cursor += len(chunk) * 4
